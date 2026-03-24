import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import gymnasium as gym
import time
import .core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from collections import defaultdict, deque
import random, math
from .stats import *
import copy
import heapq

def NeuralHashQ(env_fn, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):

    setup_pytorch_for_mpi()

    q_table = defaultdict(lambda: 0)
    frequency_table = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: 0
            ))) # state -> action -> state -> count
    min_q_table = defaultdict(lambda: 0)
    visit_count = defaultdict(lambda: 1)
    alpha = 0.5
    gamma = 0.1

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n
    stack_size = 10
    repeats = 1
    factor = 2    

    latent_dim = 16
    criterion = nn.BCEWithLogitsLoss()

    autoencoder = core.BinarySaliencyAutoencoder(latent_dim=latent_dim, input_dim=repeats * stack_size * obs_dim[0], embed_dim=128, n_queries=16, hidden_size=2048)
    anchor_autoencoder = copy.deepcopy(autoencoder)
    healing_rate = 0.5

    sync_params(autoencoder)

    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-2)
    
    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())

    # Prepare for interaction with environment
    start_time = time.time()
    frame_stacker = core.FrameStacker(stack_size=stack_size)
    input_scale = 1
    state_scale = 1
    output_scale = 1000000
    o, info = env.reset()
    o_copy = [i for i in o]
    o = frame_stacker.get_stack(o)
    o = o * input_scale

    autoencoder.eval()
    with torch.no_grad():
        # _, bits, _, _ = autoencoder(o)
        # s = core.bits_to_int(bits).item()
        s = int(autoencoder.get_int(o).item()) // output_scale
    
    ep_ret, ep_len = 0, 0
    total_vae_loss = 0
    total_world_model_loss = 0
    frame = 0
    exploration = -1

    general_trajectory_buffer = deque()
    q_trajectory_buffer = deque()
    max_trajectory_heap = []
    entry_count = 0

    null_state = torch.tensor([[0,0,0,0,0,0,0,0]], dtype=torch.float32)

    update_mode = -1

    prev_state_stats = None
    state_stats = defaultdict(lambda: [])

    for epoch in range(epochs):
        # ROLLOUT
        # exploration = (exploration + 1) % 3
        exploration = 0
        trajectory = []
        print(f"Exploration Mode: {exploration}")
        total_ep_return = 0
        total_episodes = 0
        for t in range(local_steps_per_epoch): 
            # print(len(q_table))
            # print(s)
            state_stats[s].append(o)

            q_vals = [q_table[(s, _action)] for _action in range(act_dim)]
            # prob_dist = safe_softmax(np.array(q_vals))
            # a = random.choices(range(act_dim), weights=prob_dist, k=1)[0]
            a = q_vals.index(max(q_vals))
            # print("q_val: ", q_vals)

            next_o, r, terminated, _, _ = env.step(a)
            next_o_copy = [i for i in next_o]
            next_o = frame_stacker.get_stack(next_o)
            next_o = next_o * input_scale
            
            # if not terminated:
            #     r = 0
                
            autoencoder.eval()
            with torch.no_grad():
                # _, bits_prime, _, _ = autoencoder(next_o)
                # s_prime = core.bits_to_int(bits_prime).item()
                s_prime = int(autoencoder.get_int(next_o).item()) // output_scale

                frequency_table[s][a][s_prime] += 1
                current_transition_count = frequency_table[s][a][s_prime]
                all_counts = sum([_ for _ in frequency_table[s][a].values()])

                transition_weight = float(current_transition_count) / float(all_counts)

                # print(current_transition_count, all_counts, transition_weight)

                q_vals_update = [q_table[(s_prime, _action)] for _action in range(act_dim)]
                q_table[(s, a)] += alpha * transition_weight * (r + gamma * (1-terminated) * max(q_vals_update) - q_table[(s, a)])

                # print(s, s_prime, s==s_prime) #, q_vals)
                if next_o_copy[6] + next_o_copy[7] > 0:
                    print("legs touching: ", s_prime)
                if terminated:
                    print("terminal state: ", s, s_prime)

            # S(o, b, s), A, R, S'(o, b, s)
            # where o is raw observation, b is bit latent, and s is integer latent
            trajectory.append((o, None, s, a, r, next_o, None, s_prime, terminated))

            ep_ret += r
            ep_len += 1
            timeout = ep_len == max_ep_len
            terminal = terminated or timeout
            epoch_ended = t==local_steps_per_epoch-1
            
            if terminal or epoch_ended:
                general_trajectory_buffer.append((ep_ret, trajectory))
                q_trajectory_buffer.append((ep_ret, trajectory))
                heapq.heappush(max_trajectory_heap, [-ep_ret, entry_count, trajectory, 5])
                entry_count += 1
                
                print("trajectory length: ", len(trajectory), "Q table size: ", len(q_table))
                trajectory = []

                frame_stacker = core.FrameStacker(stack_size=stack_size)
                o, info = env.reset()
                o = frame_stacker.get_stack(o)
                o = o * input_scale
                autoencoder.eval()
                with torch.no_grad():
                    # _, bits, _, _ = autoencoder(o)
                    # s = core.bits_to_int(bits).item()
                    s = int(autoencoder.get_int(o).item()) // output_scale

                total_ep_return += ep_ret
                total_episodes += 1
                ep_ret, ep_len = 0, 0

            else:
                o = next_o
                s = s_prime

        print(f"--- Episodes Summary [Epoch {epoch}] ---")
        print(f"Avg Episode Return: {total_ep_return/total_episodes:.6f}")
        print(f"--------------------------")
        

        # RUN STATS
        aggregate = defaultdict(lambda: [None, None])
        if prev_state_stats is not None:
            for state in prev_state_stats.keys():
                vectors = np.array(prev_state_stats[state])
                centroid, std = get_vector_centroid_std(vectors)
                aggregate[state][0] = (centroid, std)
            for state in state_stats.keys():
                vectors = np.array(state_stats[state])
                centroid, std = get_vector_centroid_std(vectors)
                aggregate[state][1] = (centroid, std)
            
            stability_count = 0
            average_shifts = 0
            for state, stats in aggregate.items():
                # print(stats[0])
                # print(stats[1])
                if all(x is not None for x in stats):
                    average_shifts += np.linalg.norm(stats[0][0] - stats[1][0])
                    stability_count += 1
            
            # print("stability count: ", stability_count, float(stability_count)/float(len(aggregate)))
            if stability_count > 0:
                print("average shift: ", float(average_shifts)/float(stability_count))

        # quit()

        # get average deviations from respective centroids
        total_distance = 0
        for state in state_stats.keys():
            vectors = np.array(state_stats[state])
            mean_distance, std_distance = get_vector_distance_mean_std(vectors)
            total_distance += mean_distance
        print("average deviations from centroids: ", float(total_distance)/len(state_stats))

        prev_state_stats = state_stats


        # EXPERIENCE REPLAY TRAINING
        print(f"Buffer Count: {len(general_trajectory_buffer)}")
        ## MAIN TRAINING LOOP (VAE)
        # Tracking variables for logging
        rp_vae_loss, rp_wm_loss, rp_vae_value_loss = 0, 0, 0
        total_steps = 0

        autoencoder_optimizer.zero_grad()
        autoencoder.train()
        
        # num_samples = min(len(general_trajectory_buffer), 20)
        # random_indices = random.sample(range(len(general_trajectory_buffer)), num_samples)

        obs_list = []
        soft_bits_list = []
        anchor_bits_list = []
        q_val_list = []

        # for _ in range(5):
        #     if len(max_trajectory_heap) == 0:
        #         break
        #     best_trajectory = heapq.heappop(max_trajectory_heap)
        #     if best_trajectory[3] > 1:
        #         best_trajectory[3] -= 1
        #         heapq.heappush(max_trajectory_heap, best_trajectory)
            
        #     general_trajectory_buffer.append((-best_trajectory[0], best_trajectory[2]))
        #     q_trajectory_buffer.append((-best_trajectory[0], best_trajectory[2]))

        losses = []
        total_loss = 0
        # for _trajectory in general_trajectory_buffer:
        # # for _trajectory in [general_trajectory_buffer[idx] for idx in random_indices]:
        #     total_buffer_reward, osa_trajectory = _trajectory
        #     # print(len(buffer[2]), [view_state for _, view_state, _ in trajectory_buffer])
        #     # print(len(_trajectory[2]))

        #     for i in range(len(osa_trajectory)-1, -1, -1):
        #         S_o_h, S_b_h, S_s_h, A_h, R_h, S_p_o_h, S_p_b_h, S_p_s_h, terminated = osa_trajectory[i]
                
        #         # 1. TRAIN VAE
        #         obs_list.append(S_o_h)
        #         recon, b1, soft_bits, logits = autoencoder(S_o_h)
        #         # recon_loss = F.mse_loss(recon, S_o_h)
        #         # entropy_loss = core.binary_entropy_loss(soft_bits)
        #         # vae_loss = recon_loss + entropy_loss
        #         # vae_loss = entropy_loss
        #         # rp_vae_loss += vae_loss.item()
        #         # total_loss += core.hinge_reconstruction_loss(recon, S_o_h)
        #         anchor_autoencoder.eval()
        #         with torch.no_grad():
        #             _, _, anchor_bits, _ = anchor_autoencoder(S_o_h)
        #             anchor_bits = anchor_bits.detach()
        #             anchor_bits_list.append(anchor_bits)

        #         soft_bits_list.append(soft_bits)

        #         s_q = core.bits_to_int(soft_bits).item()
        #         q_vals = [q_table[(s_q, _action)] for _action in range(act_dim)]
        #         q_val = max(abs(max(q_vals)), abs(min(q_vals))) #sum(q_vals) / float(act_dim)
        #         q_val_list.append(q_val)
                
        #         total_steps += 1
            
        #     # _, _, _, _, _, _, _, _, _terminated_1 = osa_trajectory[-2]
        #     # _, _, _, _, _, _, _, _, _terminated_2 = osa_trajectory[-1]
        #     # print(_terminated_1, _terminated_2)
        
        # for key in state_stats.keys():
        #     positions = state_stats[key]
        #     vectors = np.array(positions)
        #     v_array = np.array(vectors)
        #     centroid = np.mean(v_array, axis=0)
        #     distances = np.linalg.norm(v_array - centroid, axis=1)
        #     mean_distance = np.mean(distances)
        #     std_distance = np.std(distances)

        #     int_list = []
        #     obs_list = []
        #     for position in positions:
        #         obs_list.append(position)
        #         int_list.append(autoencoder.get_int(position).unsqueeze(0).unsqueeze(0).float())
        #         total_steps += 1
        #     int_batch = torch.stack(int_list)
        #     obs_batch = torch.stack(obs_list)
        #     total_loss += 10*core.pairwise_repulsion(int_batch, None, None, base_margin=100.0, lambda_repulsion=mean_distance, lambda_anchor=1.0)

        #     # total_loss += 1000*core.snap_boundary_loss(int_batch, obs_batch)

        all_obs = []

        for key in state_stats.keys():
            all_obs.extend(state_stats[key])

            #### BREAD AND BUTTER
            vectors = np.array(state_stats[key])
            centroid = np.mean(vectors, axis=0)
            distances = np.linalg.norm(vectors - centroid, axis=1)
            centroid_id = autoencoder.get_int(torch.tensor(centroid)).detach()
            # print(centroid)
            # print("centroid size: ", len(centroid))
            # print(torch.tensor(centroid))
            # print(distances, len(distances), len(vectors))
            # quit()
            for i, vector in enumerate(vectors):
                loss = (1.0/(distances[i]+1e-6)) * (autoencoder.get_int(torch.tensor(vector)) - centroid_id)**2
                total_loss += loss
                total_steps += 1
                # print(loss)
        
        # all_obs = torch.tensor(np.array(all_obs))
        # # print(all_obs, all_obs.size(), all_obs.dim())
        # # quit()
        # # print(autoencoder.get_int(all_obs))
        # obs_dist_mat = torch.cdist(all_obs, all_obs, p=2)+1e-6
        # # print(obs_dist_mat)
        # states = autoencoder.get_int_batch(all_obs)
        # states_dist_mat = torch.cdist(states, states, p=2)
        # mask = 1.0 - torch.eye(obs_dist_mat.size(0))
        # # print(mask)
        # loss = mask.detach() * states_dist_mat / obs_dist_mat.detach()
        # total_loss += 10000000 * loss.mean()
        # # print(loss, loss.size())
        
        # # print(states_dist_mat, states_dist_mat.size())
        # # quit()
            
        # obs_batch = torch.stack(obs_list)
        # soft_bits_batch = torch.stack(soft_bits_list)
        # anchor_bits_batch = torch.stack(anchor_bits_list)
        # q_val_batch = torch.tensor(q_val_list)
        # total_loss += core.pairwise_repulsion(soft_bits_batch, anchor_bits_batch, q_val_batch, base_margin=1.0, lambda_repulsion=1.0, lambda_anchor=1.0)
        # total_loss += 100 * core.snap_boundary_loss(soft_bits_batch, obs_batch)
        rp_vae_loss += total_loss.item()
        # total_steps = loss.size()[0]**2
        
        total_loss.backward()
        autoencoder_optimizer.step()
        general_trajectory_buffer.clear()
        state_stats = defaultdict(lambda: [])

        # # with torch.no_grad():
        # #     for live_param, anchor_param in zip(autoencoder.parameters(), anchor_autoencoder.parameters()):
        # #         # Pull the live weight slightly back to the original
        # #         live_param.data.copy_(
        # #             (1.0 - healing_rate) * live_param.data + (healing_rate) * anchor_param.data
        # #         )
                

        # # Q-TABLE TRAINING LOOP / Alternative to Bellman updates
        # print("updating")
        # # q_table = defaultdict(lambda: 0)
        # # min_q_table = defaultdict(lambda: 0)

        # trajectory_length = 0
        # autoencoder.eval()
        # for _trajectory in q_trajectory_buffer:
        #     total_buffer_reward, osa_trajectory = _trajectory
        #     trajectory_length += len(osa_trajectory)
        #     for i in range(len(osa_trajectory)-1, -1, -1):
        #         S_o_h, S_b_h, S_s_h, A_h, R_h, S_p_o_h, S_p_b_h, S_p_s_h, terminated = osa_trajectory[i]

        #         with torch.no_grad():
        #             _, bits, _, _ = autoencoder(S_o_h)
        #             s_new = core.bits_to_int(bits).item()
        #             _, bits, _, _ = autoencoder(S_p_o_h)
        #             s_prime_new = core.bits_to_int(bits).item()

        #         q_vals_update = [q_table[(s_prime_new, action)] for action in range(act_dim)]
        #         q_table[(s_new, A_h)] += alpha * (R_h + gamma * (1-terminated) * max(q_vals_update) - q_table[(s_new, A_h)])

        #         # q_vals_update = [min_q_table[(s_prime_new, _action)] for _action in range(act_dim)]
        #         # q_table[(s_new, a)] += alpha * (r + gamma * (1-terminated) * max(q_vals_update) - q_table[(s_new, a)])
        #         # min_q_vals_update = [q_table[(s_prime_new, _action)] for _action in range(act_dim)]
        #         # min_q_table[(s_new, a)] += alpha * (r + gamma * (1-terminated) * min(min_q_vals_update) - min_q_table[(s_new, a)])

        #         # if terminated:
        #         #     print(S_s_h, S_p_s_h)
        #         #     print("terminated!: ", R_h)

        # print("q table size: ", len(q_table))
        # q_trajectory_buffer.clear()
        
        avg_vae = rp_vae_loss / total_steps
        # # avg_q = rp_q_loss / q_total_steps
        # print(f"--- Replay [Epoch {epoch}] ---")
        # print(f"Buffer Count: {len(general_trajectory_buffer)}")
        print(f"Avg Replay VAE Loss: {avg_vae:.6f}")
        # # print(f"Avg Replay Q Loss:  {avg_q:.6f}")
        # print(f"--------------------------")

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            pass

    
    # # print(state_stats)
    # for key in state_stats.keys():
    #     positions = state_stats[key]
        
    #     vectors = np.array(positions)
    #     # print(vectors.tolist()[0])
    #     v_array = np.array(vectors)

    #     centroid = np.mean(v_array, axis=0)

    #     distances = np.linalg.norm(v_array - centroid, axis=1)

    #     mean_distance = np.mean(distances)
    #     std_distance = np.std(distances)

    #     print(key, len(positions), len(positions[0]), mean_distance, std_distance)

    # positions = []
    # for position_list in state_stats.values():
    #     for position in position_list:
    #         positions.append(position)

    # vectors = np.array(positions)
    # # print(vectors.tolist()[0])
    # v_array = np.array(vectors)

    # centroid = np.mean(v_array, axis=0)

    # distances = np.linalg.norm(v_array - centroid, axis=1)

    # mean_distance = np.mean(distances)
    # std_distance = np.std(distances)

    # print("overall: ", len(positions), mean_distance, std_distance)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='vae_q')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    NeuralHashQ(lambda : gym.make(args.env, render_mode="human"),
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)

    # vae_q(lambda : gym.make(args.env),
    #     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
    #     seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
    #     logger_kwargs=logger_kwargs)