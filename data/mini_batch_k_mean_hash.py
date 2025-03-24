import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random
from collections import Counter

from utils import load_binary_file, hamming_distance, hamming_distances_vectorized


def compute_centroid(cluster):
    """
    Compute the centroid (bitwise majority) of a cluster of binary codes.
    
    Args:
        cluster (list of int): List of 64-bit binary codes as integers.
        
    Returns:
        int: Centroid as a 64-bit binary code.
    """
    if not cluster:
        return None
    
    # Convert cluster to an array of bits
    cluster_bits = np.array([list(map(int, bin(code)[2:].zfill(64))) for code in cluster])
    
    # Compute bitwise majority
    majority_bits = (cluster_bits.sum(axis=0) >= (len(cluster) / 2)).astype(int)
    
    # Convert majority bits back to integer
    centroid = int("".join(map(str, majority_bits)), 2)
    return centroid


def compute_hamming_distances(args):
    """
    Compute the Hamming distances between a point and all centroids.
    
    Args:
        args (tuple): (point, centroids)
        
    Returns:
        tuple: (point, index of closest centroid, distance to closest centroid)
    """
    point, centroids = args
    # distances = [(point ^ centroid).bit_count() for centroid in centroids]
    distances = hamming_distances_vectorized(point, np.array(centroids))
    cluster_idx = np.argmin(distances)
    return point, cluster_idx


def mini_batch_kmeans_hamming(dataset, k, batch_size=1000, max_iters=10):
    """
    Perform Mini-Batch K-means clustering using Hamming distance for 64-bit binary codes.
    
    Args:
        dataset (list of int): List of 64-bit binary codes as integers.
        k (int): Number of clusters.
        batch_size (int): Size of each mini-batch.
        max_iters (int): Maximum number of iterations.
        
    Returns:
        list: Cluster assignments for each data point.
        list: Final centroids for each cluster.
    """
    n = len(dataset)
    
    # Step 1: Initialize centroids (randomly select k points from the dataset)
    centroids = np.random.choice(dataset, k, replace=False)
    
    # Initialize cluster assignments
    cluster_assignments = [None] * n
    cluster_counts = [0] * k  # Track number of points assigned to each cluster
    
    # Step 2: Iterate until convergence or max iterations
    for iteration in range(max_iters):
        print(f"Iteration {iteration + 1}")
        
        # Step 2.1: Sample a mini-batch from the dataset
        # mini_batch = random.sample(dataset, batch_size)
        # Step 2.1: Sample a mini-batch from the dataset
        mini_batch = random.sample(list(dataset), batch_size)
        
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
    
    # Final assignment: Assign all points in the dataset to the nearest centroid
    with Pool(cpu_count()) as pool:
        final_results = list(tqdm(pool.imap(compute_hamming_distances, [(point, centroids) for point in dataset]), total=n))
    
    for i, (point, cluster_idx) in enumerate(final_results):
        cluster_assignments[i] = cluster_idx  # Assign cluster index directly by position
        
    return cluster_assignments, centroids

if __name__ == '__main__':
    all_binary = load_binary_file('output_set.bin')
    
    # Perform Mini-Batch K-means clustering
    cluster_assignments, centroids = mini_batch_kmeans_hamming(all_binary, k=20000, batch_size=10000, max_iters=100)
    
    # Save the cluster assignments and centroids
    np.save("cluster_assignments.npy", cluster_assignments)
    np.save("centroids.npy", centroids)
    