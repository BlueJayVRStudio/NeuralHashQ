import numpy as np
import scipy.signal
from gymnasium.spaces import Box, Discrete
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class FrameStacker:
    def __init__(self, stack_size=10):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def get_stack(self, new_frame):
        new_frame = torch.tensor(new_frame).squeeze(0)
        if len(self.frames) == 0:
            for _ in range(self.stack_size):
                self.frames.append(new_frame)
        else:
            self.frames.append(new_frame)
        
        # print(torch.cat(list(self.frames), dim=-1))

        return torch.cat(list(self.frames), dim=-1).detach()

def bits_to_int(bits_tensor):
    if bits_tensor.dtype != torch.int:
        bits_tensor = (bits_tensor > 0.5).int()
    
    powers = 2 ** torch.arange(bits_tensor.shape[-1] - 1, -1, -1).to(bits_tensor.device)
    
    return (bits_tensor * powers).sum(dim=-1)

def binary_sigmoid(logits, alpha=5.0, dim=-1):
    y_rel = (logits - logits.mean(dim=-1, keepdim=True)) / (logits.std(dim=-1, keepdim=True) + 1e-6)
    y_soft = torch.sigmoid(y_rel * alpha)
    y_hard = (y_soft > 0.5).float()
    return (y_hard - y_soft).detach() + y_soft, y_soft

class NeuralHasher(nn.Module):
    def __init__(self, latent_dim=8, hidden_size=32, input_dim=8, n_tokens=8, n_queries=4, embed_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
        )
        self.to_logits = nn.Linear(hidden_size//4, latent_dim)
        self.places = 10
        self.to_int = nn.Linear(hidden_size, self.places)
        

    def decode(self, z):
        return self.decoder(z)
    
    def get_int(self, x):
        logits = torch.sigmoid(5 * self.to_int(self.encoder(x)))
        digits = torch.floor(logits * 10 - 1e-6)
        powers = (10 ** torch.arange(digits.size(0), device=digits.device))
        total = torch.sum(digits * powers)
        return total

    def get_int_batch(self, x):
        logits = torch.sigmoid(5 * self.to_int(self.encoder(x))).flatten()
        digits = torch.floor(logits * 10 - 1e-6).view(x.size()[0], self.places)
        powers = (10 ** torch.arange(self.places)).repeat(x.size()[0]).view(x.size()[0], self.places)
        total = torch.sum(digits * powers, dim=1).unsqueeze(1)
        return total
    
    def forward(self, x):
        logits = self.to_logits(self.encoder(x))
        bits, soft_bits = binary_sigmoid(logits)
        recon = self.decoder(bits)
        
        return recon, bits, soft_bits, logits
