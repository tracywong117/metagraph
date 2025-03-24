#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <sdsl/bit_vectors.hpp>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <future>
#include <fstream>
#include <vector>
#include <memory>
#include <filesystem>
#include "ProgressBar.h"

// Alias for readability
using bit_vector = sdsl::bit_vector;

// Custom `std::make_unique` implementation for compatibility with C++11
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Function to generate random column data
std::vector<std::unique_ptr<bit_vector>> generate_random_columns(size_t num_columns,
                                                                 size_t num_rows) {
    std::vector<std::unique_ptr<bit_vector>> columns;

    // Random number generator for bits (0 or 1)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (size_t i = 0; i < num_columns; ++i) {
        auto column = make_unique<bit_vector>(num_rows, 0);
        for (size_t j = 0; j < num_rows; ++j) {
            (*column)[j] = dis(gen); // Randomly assign 0 or 1
        }
        columns.push_back(std::move(column));
    }

    return columns;
}

// Function to load a single bit vector from a binary file
std::vector<bool> load_annotation_bit_vector_little(const std::string& file_path) {
    // Open the file in binary mode
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    // Read the file contents into a vector of uint8_t
    std::vector<uint8_t> byte_data((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());

    // Close the file
    file.close();

    // Unpack the bits in little-endian order
    std::vector<bool> bit_vector;
    for (uint8_t byte : byte_data) {
        // Process each byte and extract its bits in little-endian order
        for (int i = 0; i < 8; ++i) {
            bit_vector.push_back((byte >> i) & 1); // Extract the i-th bit
        }
    }

    return bit_vector;
}

// Function to load all columns from files in a folder
std::vector<std::unique_ptr<bit_vector>>
load_columns_from_folder(const std::string& folder_path) {
    std::vector<std::unique_ptr<bit_vector>> columns;

    // Traverse the folder to find files starting with "anno" and ending with ".bin"
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        const std::string file_name = entry.path().filename().string();

        // Check if the file name matches the pattern
        if (file_name.rfind("sorted_anno", 0) == 0 && file_name.size() > 4
            && file_name.substr(file_name.size() - 4) == ".bin") {
            std::cout << "Loading file: " << entry.path() << std::endl;

            // Load the bit vector from the file
            std::vector<bool> bit_vector_data
                    = load_annotation_bit_vector_little(entry.path().string());

            // Convert std::vector<bool> to sdsl::bit_vector
            auto bv = make_unique<bit_vector>(bit_vector_data.size(), 0);
            for (size_t i = 0; i < bit_vector_data.size(); ++i) {
                (*bv)[i] = bit_vector_data[i];
            }

            // Add to the columns
            columns.push_back(std::move(bv));
        }
    }

    if (columns.empty()) {
        throw std::runtime_error("No valid files found in the folder.");
    }

    return columns;
}

// Function to count set bits in each column
std::vector<size_t> count_set_bits(const std::vector<std::unique_ptr<bit_vector>>& columns) {
    std::vector<size_t> set_bits_count(columns.size(), 0);

    for (size_t col = 0; col < columns.size(); ++col) {
        for (size_t row = 0; row < columns[col]->size(); ++row) {
            if ((*columns[col])[row]) {
                ++set_bits_count[col];
            }
        }
    }

    return set_bits_count;
}

// Function to print set bit counts
void print_set_bits_count(const std::vector<size_t>& counts, const std::string& label) {
    std::cout << label << ":" << std::endl;
    for (size_t i = 0; i < counts.size(); ++i) {
        std::cout << "Column " << i << ": " << counts[i] << " set bits" << std::endl;
    }
}

// Function to transpose a 2D bit vector
std::vector<std::unique_ptr<bit_vector>>
transpose_columns(const std::vector<std::unique_ptr<bit_vector>>& columns) {
    if (columns.empty()) {
        throw std::runtime_error("No columns to transpose.");
    }

    size_t num_columns = columns.size();
    size_t num_rows = columns[0]->size();

    std::vector<std::unique_ptr<bit_vector>> transposed(num_rows);
    for (size_t i = 0; i < num_rows; ++i) {
        transposed[i] = make_unique<bit_vector>(num_columns, 0);
    }

    for (size_t col = 0; col < num_columns; ++col) {
        for (size_t row = 0; row < num_rows; ++row) {
            (*transposed[row])[col] = (*columns[col])[row];
        }
    }

    return transposed;
}

// Function to compute Hamming distance between two bit vectors
size_t hamming_distance(const bit_vector& a, const bit_vector& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Bit vectors must have the same size.");
    }

    size_t distance = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            ++distance;
        }
    }
    return distance;
}

// Function to compute the centroid (bitwise majority) of a cluster
std::unique_ptr<bit_vector>
compute_centroid(const std::vector<std::unique_ptr<bit_vector>>& cluster) {
    if (cluster.empty()) {
        return nullptr;
    }

    size_t num_rows = cluster[0]->size();
    auto centroid = std::make_unique<bit_vector>(num_rows, 0);

    // For each bit position, compute the majority
    for (size_t i = 0; i < num_rows; ++i) {
        size_t count_ones = 0;
        for (const auto& vec : cluster) {
            if ((*vec)[i]) {
                ++count_ones;
            }
        }
        (*centroid)[i] = (count_ones >= cluster.size() / 2); // Majority vote
    }

    return centroid;
}

// Function to assign a point to the nearest centroid
std::pair<size_t, size_t>
assign_to_centroid(const bit_vector& point,
                   const std::vector<std::unique_ptr<bit_vector>>& centroids) {
    size_t min_distance = std::numeric_limits<size_t>::max();
    size_t best_cluster_idx = 0;

    for (size_t i = 0; i < centroids.size(); ++i) {
        size_t distance = hamming_distance(point, *centroids[i]);
        if (distance < min_distance) {
            min_distance = distance;
            best_cluster_idx = i;
        }
    }

    return { best_cluster_idx, min_distance };
}

// Mini-Batch K-means clustering
std::pair<std::vector<size_t>, std::vector<std::unique_ptr<bit_vector>>>
mini_batch_kmeans(const std::vector<std::unique_ptr<bit_vector>>& dataset,
                  size_t k,
                  size_t batch_size,
                  size_t max_iters) {
    size_t n = dataset.size();
    if (n == 0) {
        throw std::runtime_error("Dataset is empty.");
    }

    size_t dim = dataset[0]->size();

    // Step 1: Initialize centroids (randomly select k points from the dataset)
    std::vector<std::unique_ptr<bit_vector>> centroids;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, n - 1);

    for (size_t i = 0; i < k; ++i) {
        centroids.push_back(std::make_unique<bit_vector>(*dataset[dis(gen)]));
    }

    // Cluster assignments
    std::vector<size_t> cluster_assignments(n, 0); // Initialize all points to cluster 0

    // Step 2: Iterate until convergence or max iterations
    for (size_t iter = 0; iter < max_iters; ++iter) {
        std::cout << "Iteration " << iter + 1 << "/" << max_iters << std::endl;

        // Step 2.1: Sample a mini-batch from the dataset
        std::vector<size_t> mini_batch_indices;
        for (size_t i = 0; i < batch_size; ++i) {
            mini_batch_indices.push_back(dis(gen));
        }

        // Step 2.2: Assign points in the mini-batch to the nearest centroid
        std::vector<std::vector<size_t>> clusters(k);
        for (size_t idx : mini_batch_indices) {
            const auto& point = *dataset[idx];
            auto [cluster_idx, _] = assign_to_centroid(point, centroids);
            clusters[cluster_idx].push_back(idx);
        }

        // Step 2.3: Update centroids
        for (size_t i = 0; i < k; ++i) {
            if (!clusters[i].empty()) {
                std::vector<std::unique_ptr<bit_vector>> cluster_data;
                for (size_t idx : clusters[i]) {
                    cluster_data.push_back(std::make_unique<bit_vector>(*dataset[idx]));
                }
                centroids[i] = compute_centroid(cluster_data);
            }
        }
    }

    // Final assignment: Assign all points in the dataset to the nearest centroid
    for (size_t i = 0; i < n; ++i) {
        auto [cluster_idx, _] = assign_to_centroid(*dataset[i], centroids);
        cluster_assignments[i] = cluster_idx;
    }

    return { cluster_assignments, std::move(centroids) };
}

// Perform Rowdiff based on anchors and clusters_points
std::vector<std::unique_ptr<bit_vector>>
perform_rowdiff(const std::vector<std::unique_ptr<bit_vector>>& data,
                const std::vector<std::unique_ptr<bit_vector>>& centroids,
                const std::vector<size_t>& cluster_assignments) {
    size_t num_rows = data.size();
    std::vector<std::unique_ptr<bit_vector>> result;

    // Step 1: Make a copy of the data
    for (const auto& row : data) {
        result.push_back(make_unique<bit_vector>(*row));
    }

    // Step 2: Perform rowdiff
    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        size_t cluster_idx = cluster_assignments[row_idx]; // Get the cluster ID for the row
        const bit_vector& centroid
                = *centroids[cluster_idx]; // Get the corresponding centroid

        bit_vector& target_row = *result[row_idx]; // Get the row to update
        for (size_t col = 0; col < target_row.size(); ++col) {
            target_row[col] = centroid[col] ^ target_row[col]; // XOR with the centroid
        }
    }

    return result;
}

// Save bit vectors to files (one file per column)
void save_columns_to_files(const std::vector<std::unique_ptr<bit_vector>>& columns,
                           const std::string& prefix) {
    for (size_t col = 0; col < columns.size(); ++col) {
        // Open the file in binary mode
        std::ofstream file(prefix + "_col" + std::to_string(col) + ".bin", std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + prefix + "_col"
                                     + std::to_string(col) + ".bin");
        }

        const bit_vector& bv = *columns[col];
        size_t num_bits = bv.size();
        size_t num_bytes = (num_bits + 7)
                / 8; // Calculate the number of bytes needed to store the bits
        std::vector<uint8_t> byte_data(num_bytes, 0); // Initialize a byte array with zeros

        // Pack the bits into bytes in little-endian order
        for (size_t i = 0; i < num_bits; ++i) {
            if (bv[i]) {
                byte_data[i / 8] |= (1 << (i % 8)); // Set the appropriate bit in the byte
            }
        }

        // Write the bytes to the file
        file.write(reinterpret_cast<const char*>(byte_data.data()), byte_data.size());

        // Close the file
        file.close();
    }
}

int main() {
    try {
        // Step 1: Load dataset (replace with your loading logic)
        // std::string folder_path
        //         = "/media/data/tracy/metagraph/data"; // Replace with your folder path

        if (argc != 5) {
            std::cerr << "Usage: " << argv[0] << " <folderPath> <k> <batch_size> <max_iters>" << std::endl;
            return 1;
        }
        std::string folder_path = argv[1];
        size_t k = std::stoul(argv[2]);
        size_t batch_size = std::stoul(argv[3]);
        size_t max_iters = std::stoul(argv[4]);

        auto columns = load_columns_from_folder(folder_path);

        size_t num_columns = columns.size();
        size_t num_rows = columns[0]->size();
        std::cout << "Loaded " << num_columns << " columns, each with " << num_rows
                  << " rows." << std::endl;

        // Step 2: Transpose the data for clustering
        auto transposed = transpose_columns(columns);

        // Step 3: Perform Mini-Batch K-means clustering
        auto [cluster_assignments, centroids]
                = mini_batch_kmeans(transposed, k, batch_size, max_iters);

        // Print the number of clusters
        std::cout << "Mini-Batch K-means clustering completed with " << centroids.size()
                  << " clusters." << std::endl;

        // Step 4: Perform Rowdiff
        auto rowdiff_result = perform_rowdiff(transposed, centroids, cluster_assignments);

        // Step 5: Transpose back the result of Rowdiff
        auto final_result = transpose_columns(rowdiff_result);

        // Step 6: Save the final result to files
        save_columns_to_files(final_result, "column_row_diff");

        std::cout << "Bit vectors written to files." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}