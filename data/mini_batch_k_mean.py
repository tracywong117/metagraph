import numpy as np
from utils import *
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import Counter
import sys

def compute_centroid(cluster):
    """
    Compute the centroid (bitwise majority) of a cluster of dim-bit binary codes (dim/8 uint8).
    
    Args:
        cluster (list of np.ndarray): List of (dim/8,) binary codes as `uint8` arrays.
        
    Returns:
        np.ndarray: Centroid as a (dim/8,) `uint8` array.
    """
    if not cluster:
        return None
    
    # Convert cluster to a 2D numpy array (each row is a uint8 array)
    cluster_bits = np.array(cluster, dtype=np.uint8)
    
    # Compute bitwise majority for each bit in the uint8 arrays
    # First, count the number of 1s per bit position
    # Since each uint8 has 8 bits, we need to count these bitwise
    majority_bits = np.zeros_like(cluster_bits[0], dtype=np.uint8)
    for i in range(8):  # Iterate over all 8 bits in each uint8
        bit_counts = (cluster_bits >> i) & 1  # Isolate the i-th bit for all elements
        majority_bit = (bit_counts.sum(axis=0) >= (len(cluster) / 2)).astype(np.uint8)  # Majority vote
        majority_bits |= (majority_bit << i)  # Set the i-th bit in the centroid
    
    return majority_bits


def compute_hamming_distances(args):
    """
    Compute the Hamming distances between a point and all centroids.
    
    Args:
        args (tuple): (point, centroids)
        
    Returns:
        tuple: (point, index of closest centroid, distance to closest centroid)
    """
    point, centroids = args
    distances = hamming_distances_1_to_n(point, np.array(centroids))
    cluster_idx = np.argmin(distances)
    return point, cluster_idx


def hamming_distances_1_to_n(point, array):
    """
    point: numpy array of uint8 (dim/8)
    array: numpy array of uint8 (m, dim/8)
    
    return: numpy array of uint8 (m)
    """
    return np.unpackbits(np.bitwise_xor(point, array), axis=1).sum(axis=1)


# def mini_batch_kmeans_hamming(dataset, k, batch_size=1000, max_iters=10):
#     """
#     Perform Mini-Batch K-means clustering using Hamming distance for 64-bit binary codes.
    
#     Args:
#         dataset (list of int): List of 64-bit binary codes as integers.
#         k (int): Number of clusters.
#         batch_size (int): Size of each mini-batch.
#         max_iters (int): Maximum number of iterations.
        
#     Returns:
#         list: Cluster assignments for each data point.
#         list: Final centroids for each cluster.
#     """
#     n = len(dataset)
    
#     # Step 1: Initialize centroids (randomly select k points from the dataset)
#     # dataset is (n, dim/8)
#     # random select k points from the dataset
#     rng = np.random.default_rng()
#     centroids = rng.choice(dataset, k, replace=False)
    
#     # Initialize cluster assignments
#     cluster_assignments = [None] * n
#     cluster_counts = [0] * k  # Track number of points assigned to each cluster
#     print("Initialization complete.")
    
#     # Step 2: Iterate until convergence or max iterations
#     for iteration in tqdm(range(max_iters)):
#         # Step 2.1: Sample a mini-batch from the dataset
#         mini_batch = dataset[rng.choice(n, batch_size, replace=False)]
        
#         # Step 2.2: Assign points in the mini-batch to the nearest centroid
#         clusters = [[] for _ in range(k)]
        
#         # Use multiprocessing to compute distances in parallel
#         with Pool(cpu_count()) as pool:
#             results = list(pool.imap(compute_hamming_distances, [(point, centroids) for point in mini_batch]))
#         # Single-threaded version
#         # results = [compute_hamming_distances((point, centroids)) for point in mini_batch]
        
#         # Update cluster assignments and counts based on mini-batch results
#         for point, cluster_idx in results:
#             clusters[cluster_idx].append(point)
#             cluster_counts[cluster_idx] += 1
        
#         # Step 2.3: Update centroids incrementally using the mini-batch
#         new_centroids = centroids.copy()
#         for cluster_idx, cluster in enumerate(clusters):
#             if cluster:
#                 cluster_centroid = compute_centroid(cluster)
                
#                 # Incrementally update centroid using a weighted average
#                 if cluster_counts[cluster_idx] > 0:
#                     new_centroids[cluster_idx] = (
#                         (cluster_counts[cluster_idx] - len(cluster)) * centroids[cluster_idx] + cluster_centroid
#                     ) // cluster_counts[cluster_idx]
        
#         # Check for convergence (if centroids do not change)
#         if np.array_equal(centroids, new_centroids):
#             print("Convergence reached.")
#             break
        
#         centroids = new_centroids
        
#     # Save the cluster assignments and centroids
#     np.save("pre_cluster_assignments.npy", np.array(cluster_assignments))
#     np.save("pre_centroids.npy", np.array(centroids))
    
#     # Final assignment: Assign all points in the dataset to the nearest centroid
#     # ================== Multi-threaded version with tqdm ==================
#     with Pool(cpu_count()) as pool:
#         final_results = list(tqdm(pool.imap(compute_hamming_distances, [(point, centroids) for point in dataset]), total=n))
#     for i, (point, cluster_idx) in enumerate(final_results):
#         cluster_assignments[i] = cluster_idx  # Assign cluster index directly by position
#     # ================== Single-threaded version with tqdm ==================
#     # for i, point in enumerate(tqdm(dataset)):
#     #     final_results = compute_hamming_distances((point, centroids))
#     #     cluster_assignments[i] = final_results[1]
#     # =======================================================================
    
        
#     return cluster_assignments, centroids

def mini_batch_kmeans_hamming(dataset, k, batch_size=1000, max_iters=10, is_resume=False, 
                              pre_computed_file="pre_computed.npy", checkpoint_interval=1500000, checkpoint_file="checkpoint.npy",
                              final_data_file="final_data.npy"):
    """
    Perform Mini-Batch K-means clustering using Hamming distance for 64-bit binary codes.
    
    Args:
        dataset (list of int): List of 64-bit binary codes as integers.
        k (int): Number of clusters.
        batch_size (int): Size of each mini-batch.
        max_iters (int): Maximum number of iterations.
        checkpoint_interval (int): Number of points processed between checkpoints.
        checkpoint_file (str): File to save checkpoint data.
        
    Returns:
        list: Cluster assignments for each data point.
        list: Final centroids for each cluster.
    """
    n = len(dataset)
    
    if is_resume:
        if not os.path.exists(pre_computed_file):
            raise ValueError("Pre-computed file does not exist.")
        pre_computed_data = np.load(pre_computed_file, allow_pickle=True).item()
        cluster_assignments = pre_computed_data["cluster_assignments"]
        centroids = pre_computed_data["centroids"]
        print("Resuming from pre-computed data.")
    else:
        # Step 1: Initialize centroids (randomly select k points from the dataset)
        rng = np.random.default_rng()
        centroids = rng.choice(dataset, k, replace=False)
        
        # Initialize cluster assignments
        cluster_assignments = [None] * n
        cluster_counts = [0] * k  # Track number of points assigned to each cluster
        print("Initialization complete.")
        
        # Step 2: Iterate until convergence or max iterations
        for iteration in tqdm(range(max_iters)):
            # Step 2.1: Sample a mini-batch from the dataset
            mini_batch = dataset[rng.choice(n, batch_size, replace=False)]
            
            # Step 2.2: Assign points in the mini-batch to the nearest centroid
            clusters = [[] for _ in range(k)]
            
            # Use multiprocessing to compute distances in parallel
            with Pool(cpu_count()) as pool:
                results = list(pool.imap(compute_hamming_distances, [(point, centroids) for point in mini_batch]))
            
            # Update cluster assignments and counts based on mini-batch results
            for point, cluster_idx in results:
                clusters[cluster_idx].append(point)
                cluster_counts[cluster_idx] += 1
            
            # Step 2.3: Update centroids incrementally using the mini-batch
            new_centroids = centroids.copy()
            for cluster_idx, cluster in enumerate(clusters):
                if cluster:
                    cluster_centroid = compute_centroid(cluster)
                    
                    # Incrementally update centroid using a weighted average
                    if cluster_counts[cluster_idx] > 0:
                        new_centroids[cluster_idx] = (
                            (cluster_counts[cluster_idx] - len(cluster)) * centroids[cluster_idx] + cluster_centroid
                        ) // cluster_counts[cluster_idx]
            
            # Check for convergence (if centroids do not change)
            if np.array_equal(centroids, new_centroids):
                print("Convergence reached.")
                break
            
            centroids = new_centroids
        
        # Save the cluster assignments and centroids into a checkpoint file
        checkpoint_data = {
            "cluster_assignments": cluster_assignments,
            "centroids": centroids
        }
        np.save(pre_computed_file, checkpoint_data)
    
    # Final assignment: Assign all points in the dataset to the nearest centroid
    print("Performing final assignment...")
    
    # Load checkpoint if it exists
    if is_resume and os.path.exists(checkpoint_file):
        checkpoint_data = np.load(checkpoint_file, allow_pickle=True).item()
        cluster_assignments = checkpoint_data["cluster_assignments"]
        start_idx = checkpoint_data["start_idx"]
        print(f"Resuming from checkpoint at index {start_idx}.")
    else:
        start_idx = 0
    
    # Process the dataset in chunks
    with Pool(cpu_count()) as pool:
        for i in range(start_idx, n, checkpoint_interval):
            end_idx = min(i + checkpoint_interval, n)
            chunk = dataset[i:end_idx]
            
            # Compute assignments for this chunk in parallel
            results = list(tqdm(pool.imap(compute_hamming_distances, [(point, centroids) for point in chunk]), total=len(chunk)))
            
            # Update cluster assignments
            for j, (point, cluster_idx) in enumerate(results):
                cluster_assignments[i + j] = cluster_idx
            
            # Save a checkpoint
            checkpoint_data = {
                "cluster_assignments": cluster_assignments,
                "start_idx": end_idx
            }
            np.save(checkpoint_file, checkpoint_data)
    
    # Save the final cluster assignments and centroids
    final_data = {
        "cluster_assignments": cluster_assignments,
        "centroids": centroids
    }
    np.save(final_data_file, final_data)
    
    # Clean up the checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    return cluster_assignments, centroids


if __name__ == "__main__":
    # Get argument for whether to resume from checkpoint
    if len(sys.argv) > 1:
        is_resume = bool(int(sys.argv[1]))
    else:
        is_resume = False
    # How to use this script:
    # python mini_batch_k_mean.py 0 (to start from scratch)
    # python mini_batch_k_mean.py 1 (to resume from checkpoint)
    
    anno_files_list = get_all_anno_files(".")
    anno_bit_vectors = load_all_annotation_bit_vectors(anno_files_list).T
    print(anno_bit_vectors.shape) # [rows, cols]
    packed_anno_bit_vectors = np.packbits(anno_bit_vectors, axis=1)
    packed_anno_bit_vectors = np.ascontiguousarray(packed_anno_bit_vectors)
    print(packed_anno_bit_vectors.shape) # [rows, cols/8]
    
    # Perform Mini-Batch K-means clustering
    cluster_assignments, centroids = mini_batch_kmeans_hamming(packed_anno_bit_vectors, k=50000, batch_size=10000, max_iters=100, is_resume=is_resume)