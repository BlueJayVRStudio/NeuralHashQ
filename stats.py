import numpy as np

def get_vector_distance_mean_std(positions):
    v_array = np.array(positions)
    centroid = np.mean(v_array, axis=0)
    distances = np.linalg.norm(v_array - centroid, axis=1)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    return mean_distance, std_distance

def get_vector_centroid_std(positions):
    v_array = np.array(positions)
    centroid = np.mean(v_array, axis=0)
    std = np.std(v_array, axis=0)
    return centroid, std

if __name__ == '__main__':
    v = np.array([[1.1, 1.9], [1, 2], [1, 2], [1, 2], [1, 2]])
    a, b = get_vector_distance_mean_std(v)
    c, d = get_vector_centroid_std(v)

    print(a, b)
    print(c, d)
    print(c.shape)
