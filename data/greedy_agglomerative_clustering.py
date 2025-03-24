import numpy as np
from utils import *
import faiss
import random
import time
from tqdm import tqdm


def pair_hamming_distances(arr1, arr2):
    """
    Compute the Hamming distance between two sets of uint8 arrays.

    Parameters:
    arr1 (ndarray): Shape (num_bit_vectors, dimension_bit_vector)
    arr2 (ndarray): Shape (num_bit_vectors, dimension_bit_vector)

    Returns:
    ndarray: Shape (num_bit_vectors,) - the Hamming distances between corresponding vectors in arr1 and arr2.
    """
    # Step 1: XOR the two arrays
    xor_result = np.bitwise_xor(arr1, arr2)

    # Step 2: Count the number of differing bits for each element
    # Use `bin(x).count('1')` to count the number of 1s in the binary representation
    hamming_distance = np.vectorize(lambda x: bin(x).count('1'))(xor_result)

    # Step 3: Sum all the bit differences
    total_hamming_distance = np.sum(hamming_distance, axis=1)

    return total_hamming_distance


def greedy_agglomerative_clustering_hamming(data, num_clusters, sampling_size=10):
    """
    Perform Greedy Agglomerative Clustering on packed binary data using Hamming distance.
    Each cluster is represented by the original point of the first cluster in the merge.

    Args:
        data: NumPy array of shape (N, D), where N is the number of points and D is the packed dimensionality (e.g., 4 for 32-bit vectors).
        num_clusters: Desired number of clusters.
        sampling_size: Number of random cluster pairs to sample in each iteration.

    Returns:
        points_assign_cluster: Array indicating the cluster assigned to each point.
        representatives: Array where each row is the representative of the cluster assigned to a point.
        clusters_points: Dictionary mapping cluster IDs to the list of all points in each cluster.
    """
    # Step 1: Initialize variables
    n_points = data.shape[0]
    
    points_assign_cluster = np.arange(n_points)  # Each point starts in its own cluster
    representatives = data.copy()  # Each point initially represents its own cluster
    clusters_points = {i: [i] for i in range(n_points)}  # Each cluster starts with a single point
    existing_clusters = set(range(n_points))  # Set of existing cluster IDs
    number_of_existing_clusters = n_points  # Number of existing clusters
    
    number_loop = n_points - num_clusters  # Number of iterations to reach the desired number of clusters

    rng = np.random.default_rng()
    
    print("Number of clusters:", len(clusters_points))
    print("Number of points:", n_points)
    print("Start clustering...")

    # Initialize tqdm progress bar
    with tqdm(total=number_loop, desc="Clustering Progress") as pbar:
        i = 0
        while i < number_loop:
            # Timing dictionary to store times for each step
            timings = {
                "sampling_clusters": 0,
                "compute_distances": 0,
                "merge_clusters": 0,
                "update_clusters": 0,
            }
            
            # Step 2: Randomly sample cluster pairs
            start_time = time.time()
            
            # ============== Option 1: Sample from all points ==============
            # sample cluster pairs from all points
            all_sampled = rng.choice(n_points, 2 * sampling_size, replace=False)
            
            # ============== Option 2: Sample from existing clusters ==============
            # sample cluster pairs from existing clusters
            # all_sampled = rng.choice(list(existing_clusters), 2 * sampling_size, replace=False)
            
            sampled_1 = all_sampled[:sampling_size] # first half
            sampled_2 = all_sampled[sampling_size:] # second half
            
            timings["sampling_clusters"] = time.time() - start_time

            # Step 3: Find the pair with the smallest Hamming distance (based on representatives)
            start_time = time.time()
            distances = pair_hamming_distances(representatives[sampled_1], representatives[sampled_2])
            min_distance = np.min(distances)
            min_sampled_index = np.argmin(distances)
            timings["compute_distances"] = time.time() - start_time

            # Step 4: Merge the selected pair of clusters
            start_time = time.time()
            c1 = sampled_1[min_sampled_index]
            c2 = sampled_2[min_sampled_index]
            
            # Skip if the two points are already in the same cluster
            if points_assign_cluster[c1] == points_assign_cluster[c2]:
                # print("continue")
                pbar.update(1)
                i += 1
                continue
            
            # # Check number of points in each cluster
            if len(clusters_points[points_assign_cluster[c1]]) < len(clusters_points[points_assign_cluster[c2]]):
                # swap c1 and c2 such that the new representative is from the larger cluster
                tmp = c1
                c1 = c2
                c2 = tmp

            cluster_id_c1 = points_assign_cluster[c1]
            cluster_id_c2 = points_assign_cluster[c2]
            
            # Add c2's points to c1
            clusters_points[cluster_id_c1].extend(clusters_points[cluster_id_c2])
            
            # Update the cluster assignment of each point in c2
            points_assign_cluster[clusters_points[cluster_id_c2]] = cluster_id_c1

            # Update the representative of c2
            representatives[clusters_points[cluster_id_c2]] = representatives[c1]
            
            # Clear c2's points
            clusters_points[cluster_id_c2] = []
            
            # Remove c2 from the set of existing clusters
            existing_clusters.remove(cluster_id_c2)
            
            # Update the number of existing clusters
            number_of_existing_clusters -= 1

            # Delete c2 from clusters_points (no deletion of representatives since they are index-based)
            timings["merge_clusters"] = time.time() - start_time
            
            if timings["merge_clusters"] > 1:
                print("Timings (in seconds):", timings)

            # Update progress bar
            pbar.update(1)
            
            # Print timings for this iteration (optional, can slow down overall runtime)
            # print("Timings (in seconds):", timings)
            
            i += 1

    # Update representatives for all points based on their assigned clusters
    representatives = np.array([representatives[points_assign_cluster[i]] for i in range(n_points)])

    return points_assign_cluster, representatives, clusters_points


# Example Usage
if __name__ == "__main__":
    # # ============== Example Usage ==============
    # # Example packed binary data: shape (200000, 4) for 200,000 vectors (32 bits each packed into 4 uint8)
    # np.random.seed(42)
    # n_points = 10000000 # For testing, you can increase this to a larger dataset
    # n_bits = 32
    # packed_dim = n_bits // 8  # Each uint8 holds 8 bits

    # # Generate random binary data (32 bits per row) and pack it into uint8
    # data = np.random.randint(0, 2, (n_points, n_bits), dtype='uint8')
    # packed_data = np.packbits(data, axis=1)  # Shape: (n_points, packed_dim)
    # # print(packed_data.shape)  # [rows, cols]
    # # ===========================================
    
    anno_files_list = get_all_anno_files(".")
    anno_bit_vectors = load_all_annotation_bit_vectors(anno_files_list).T
    print(anno_bit_vectors.shape) # [rows, cols]
    
    packed_anno_bit_vectors = np.packbits(anno_bit_vectors, axis=1)
    packed_anno_bit_vectors = np.ascontiguousarray(packed_anno_bit_vectors)
    print(packed_anno_bit_vectors.shape) # [rows, cols/8]
    
    n_points = packed_anno_bit_vectors.shape[0]
    
    packed_data = packed_anno_bit_vectors
    
    # Perform clustering
    num_clusters = 50000
    points_assign_cluster, representatives, clusters_points = greedy_agglomerative_clustering_hamming(
        packed_data, num_clusters, sampling_size=100
    )

    # Number of clusters
    set_clusters = set(points_assign_cluster)
    print(f"Number of clusters: {len(set_clusters)}")
    
    # Check if all points has been assigned to a cluster
    # for loop cluster_points, add all points to a set
    # check if the set contains all points
    all_points = set(range(n_points))
    assigned_points = set()
    for cluster_points in clusters_points.values():
        assigned_points.update(cluster_points)
    print("All points assigned:", all_points == assigned_points)
    print("Number of assigned points:", len(assigned_points))