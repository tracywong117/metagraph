import numpy as np
import faiss
from tqdm import tqdm
import os

def check_file_exists(file_path):
    """
    Check if a file exists
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Failed to open file: {file_path}")

def load_binary_file(file):
    """
    Load a numpy array from a binary file (for hash codes set)
    """
    check_file_exists(file)
    with open(file, 'rb') as f:
        packed_data = f.read()
        
    return np.frombuffer(packed_data, dtype=np.uint64)


def write_binary_file(file, data):
    """
    Write a numpy array to a binary file (for hash codes set)
    """
    with open(file, 'wb') as f:
        f.write(data.tobytes())


def load_binary_file_bit_vector(file):
    """
    Load a numpy array from a binary file (for annotation bit vectors)
    """
    check_file_exists(file)
    with open(file, 'rb') as f:
        packed_data = f.read()
        
    data = np.frombuffer(packed_data, dtype=np.uint8)
    return np.unpackbits(data, axis=-1, bitorder='little')


def write_binary_file_bit_vector(file, data):
    """
    Write a numpy array to a binary file (for annotation bit vectors)
    """
    packed_data = np.packbits(data, axis=-1, bitorder='little').view(np.uint8)
    with open(file, 'wb') as f:
        f.write(packed_data.tobytes())


def get_all_anno_files(anno_dir, start_with="anno_"):
    """
    Get all annotation files in the directory
    """
    files = os.listdir(anno_dir)
    anno_files = [f for f in files if f.startswith(start_with) and f.endswith(".bin")]
    return anno_files


def load_annotation_bit_vector_positions_little(file, positions):
    """
    Load specific positions from an annotation bit vector file
    """
    check_file_exists(file)
    with open(file, 'rb') as file:
        byte_data = file.read()

    bit_vector = []
    for position in positions:
        byte_index = position // 8
        bit_index = position % 8
        bit_vector.append((byte_data[byte_index] >> bit_index) & 1)

    return bit_vector

        
def hamming_distance(a, b):
    """Calculate the Hamming distance using bit_count() (Python 3.10+)."""
    return (a ^ b).bit_count()


def hamming_distances_vectorized(point, lst_binary):
    """
    Compute the Hamming distances between a point and a list of binary codes.
    """
    
    # XOR the point with the list
    xor_result = point ^ lst_binary  # XOR result is still uint64
    
    # Convert uint64 to uint8 view (8 bytes per uint64)
    xor_bytes = xor_result.view(np.uint8)  # Interpret each uint64 as 8 uint8 values
    
    # Unpack bits and count the number of 1s for each uint64
    unpacked_bits = np.unpackbits(xor_bytes, axis=0).reshape(len(lst_binary), 64)
    distances = unpacked_bits.sum(axis=1)
    
    return distances


def rowdiff_find_path(anchor, hash_codes_set, cluster_indices):
    """
    Find the path starting from anchor
    Within a cluster, starting from the anchor, find the nearest top 1 point as the successor using FAISS.
    Repeat this process until all points in the cluster are visited.
    """
    num_points = len(cluster_indices)
    
    # Create an index
    index = faiss.IndexBinaryFlat(64)
    b_uint8 = hash_codes_set[cluster_indices].view(np.uint8).reshape(-1, 8)
    index.add(b_uint8)
    
    anchor_uint8 = np.array([anchor]).view(np.uint8).reshape(1, -1)
    
    D, I = index.search(anchor_uint8, k=1)
    print(f"Nearest neighbors: {I} with distance {D}")
    
    current_point_index = I[0][0]
    print(f"Current points: {current_point_index}")
    path = [current_point_index]
    cost = [D[0][0]]
    
    for i in tqdm(range(num_points)):
        # Remove the current point from the index
        index.remove_ids(np.array([current_point_index], dtype=np.int64))
        
        # Perform a search
        D, I = index.search(np.array([hash_codes_set[current_point_index]]).view(np.uint8).reshape(1, -1), k=1)
        current_point_index = I[0][0]
        path.append(current_point_index)
        cost.append(D[0][0])
        
    return path, cost


# def get_row(anno_files_list, position):
#     """
#     Get a specific row from an annotation file
#     """
#     # Load annotation bit vectors
#     row_bit_vectors = [load_annotation_bit_vector_positions_little(file, [position])[0] for file in anno_files_list]
#     return np.array(row_bit_vectors)


def load_all_annotation_bit_vectors(anno_files_list):
    """
    Load all annotation bit vectors from a list of files into memory.
    Returns a list of full bit vectors (one per file).
    """
    all_bit_vectors = []
    for file in anno_files_list:
        check_file_exists(file)
        with open(file, 'rb') as f:
            byte_data = f.read()
        # Convert byte data into a full bit vector
        bit_vector = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8), axis=-1, bitorder='little')
        all_bit_vectors.append(bit_vector)
    return np.array(all_bit_vectors) # [column, row]


def get_row(preloaded_bit_vectors, position):
    """
    Get a specific row (bit values at a specific position) from preloaded bit vectors.
    """
    # Extract the bit at the specified position from each preloaded bit vector
    row_bit_vectors = [bit_vector[position] for bit_vector in preloaded_bit_vectors]
    return np.array(row_bit_vectors)


def count_set_bits(row):
    """
    Count the number of set bits
    """
    return np.count_nonzero(row)


def compare_rows(anno_files_list, positions):
    """
    Compare annotation row bit vectors from all column files and compute the Hamming distance between two rows.
    """
    
    # Check len(positions) == 2
    if len(positions) != 2:
        print("Error: positions must contain exactly two values.")
        return None
    
    # Load annotation bit vectors
    try:
        annotation_bit_vectors = [
            load_annotation_bit_vector_positions_little(file, positions)
            for file in anno_files_list
        ]
    except Exception as e:
        print(f"Error while loading annotation bit vectors: {e}")
        return None

    annotation_bit_vectors = np.array(annotation_bit_vectors, dtype=np.bool_)
    annotation_bit_vectors = annotation_bit_vectors.T
    hamming_distance = np.count_nonzero(annotation_bit_vectors[0] != annotation_bit_vectors[1])

    return hamming_distance


if __name__ == "__main__":
    hash_codes_set = load_binary_file('output_set.bin')
    anno_bit_vector = load_binary_file_bit_vector('anno_SRR2125928.fastq_embedding_little.bin')
    
    print(hash_codes_set.shape)
    print(anno_bit_vector.shape)
    print(hash_codes_set[0:10])
    print(anno_bit_vector[0:10])