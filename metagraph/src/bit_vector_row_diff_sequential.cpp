#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <random>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include "ProgressBar.h"

// Class to precompute and store the indices of set bits in a bit vector
class BitIndexStore {
private:
    std::vector<size_t> set_bit_indices; // Vector to store indices of set bits

public:
    // Constructor that initializes the data structure with a bit vector
    BitIndexStore(const std::vector<bool> &bit_vector) {
        for (size_t i = 0; i < bit_vector.size(); ++i) {
            if (bit_vector[i]) {
                set_bit_indices.push_back(i); // Store the index of each set bit
            }
        }
    }

    // Public method to retrieve all indices of set bits
    const std::vector<size_t>& get_set_bit_indices() const {
        return set_bit_indices;
    }

    // Method to return the number of set bits
    size_t count_set_bits() const {
        return set_bit_indices.size();
    }

    // Method to count memory usage in bytes
    size_t memory_usage_bytes() const {
        return set_bit_indices.size() * sizeof(size_t);
    }
};

// Function to compute the Hamming distance using precomputed set bit indices
size_t compute_hamming_distance_index(const BitIndexStore &bv1, const BitIndexStore &bv2) {
    const std::vector<size_t>& indices1 = bv1.get_set_bit_indices();
    const std::vector<size_t>& indices2 = bv2.get_set_bit_indices();

    size_t i = 0, j = 0;
    size_t hamming = 0;

    // Compare the positions of set bits in both vectors
    while (i < indices1.size() && j < indices2.size()) {
        if (indices1[i] == indices2[j]) {
            // Both bit positions are the same; move forward
            ++i;
            ++j;
        } else if (indices1[i] < indices2[j]) {
            // Bit position in bv1 is smaller; it contributes to Hamming distance
            ++hamming;
            ++i;
        } else { // indices1[i] > indices2[j]
            // Bit position in bv2 is smaller; it contributes to Hamming distance
            ++hamming;
            ++j;
        }
    }

    // Add remaining bits in bv1 (if any)
    hamming += (indices1.size() - i);

    // Add remaining bits in bv2 (if any)
    hamming += (indices2.size() - j);

    return hamming;
}

// Function to generate random columns as bit vectors with a specified density
std::vector<std::vector<bool>> generate_random_columns(size_t num_columns, size_t num_rows, double density) {
    // Validate density value
    if (density < 0.0 || density > 1.0) {
        throw std::invalid_argument("Density must be between 0.0 and 1.0.");
    }

    std::vector<std::vector<bool>> columns(num_columns, std::vector<bool>(num_rows, false));

    // Random number generator for bits with density
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); // Generate real values between 0 and 1

    for (size_t i = 0; i < num_columns; ++i) {
        for (size_t j = 0; j < num_rows; ++j) {
            columns[i][j] = (dis(gen) < density); // Assign 1 based on the density probability
        }
    }

    return columns;
}

// Function to count the number of set bits in each column
std::vector<size_t> count_set_bits(const std::vector<std::vector<bool>>& columns) {
    std::vector<size_t> set_bits_count;
    for (const auto& column : columns) {
        BitIndexStore store(column);
        set_bits_count.push_back(store.count_set_bits());
    }
    return set_bits_count;
}

// Function to transpose a 2D bit vector (columns to rows)
std::vector<std::vector<bool>> transpose_columns(const std::vector<std::vector<bool>>& columns) {
    if (columns.empty() || columns[0].empty()) {
        throw std::runtime_error("Cannot transpose empty columns.");
    }

    size_t num_columns = columns.size();
    size_t num_rows = columns[0].size();

    // Create a new 2D vector for the transposed matrix
    std::vector<std::vector<bool>> transposed(num_rows, std::vector<bool>(num_columns, false));

    // Perform the transpose
    for (size_t col = 0; col < num_columns; ++col) {
        for (size_t row = 0; row < num_rows; ++row) {
            transposed[row][col] = columns[col][row];
        }
    }

    return transposed;
}

// Function to perform Greedy Sequential Clustering
std::pair<std::vector<size_t>, std::vector<std::vector<size_t>>>
greedySequentialClusteringHamming(const std::vector<std::vector<bool>>& data, size_t threshold) {

    size_t num_points = data.size();

    // Initialize the progress bar
    ProgressBar progressBar(50, "Clustering Progress", num_points);

    std::vector<size_t> anchors; // Anchor indices
    std::vector<std::vector<size_t>> clusters; // Clusters of indices

    for (size_t i = 0; i < num_points; ++i) {
        BitIndexStore current(data[i]);
        bool assigned_to_cluster = false;

        // Check if the point can be assigned to an existing cluster
        for (size_t j = 0; j < anchors.size(); ++j) {
            size_t anchor_idx = anchors[j];
            BitIndexStore anchor(data[anchor_idx]);

            if (compute_hamming_distance_index(current, anchor) < threshold) {
                clusters[j].push_back(i);
                assigned_to_cluster = true;
                break;
            }
        }

        // If not assigned to any cluster, create a new one
        if (!assigned_to_cluster) {
            anchors.push_back(i);
            clusters.push_back({ i });
        }

        // Update the progress bar
        progressBar.update();
    }

    // Finish the progress bar
    progressBar.finish();

    return { anchors, clusters };
}

// Main function
int main(int argc, char* argv[]) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " <folderPath> <threshold>" << std::endl;
            return 1;
        }

        std::string folder_path = argv[1];
        size_t threshold = std::stoul(argv[2]); // Maximum Hamming distance for clustering

        // Generate random columns instead of loading from files
        auto columns = generate_random_columns(32, 600'000, 0.1);

        // Validate generated columns
        if (columns.empty() || columns[0].empty()) {
            throw std::runtime_error("Generated columns are invalid.");
        }

        size_t num_columns = columns.size();
        size_t num_rows = columns[0].size();
        std::cout << "Loaded " << num_columns << " columns, each with " << num_rows
                  << " rows." << std::endl;

        // Count total set bits
        auto set_bits_count = count_set_bits(columns);
        size_t total_set_bits = std::accumulate(set_bits_count.begin(), set_bits_count.end(), 0);
        std::cout << "Total set bits: " << total_set_bits << std::endl;

        // Transpose the data before clustering
        auto transposed = transpose_columns(columns);
        std::cout << "Data transposed: " << transposed.size() << " rows, " << transposed[0].size()
                  << " columns." << std::endl;

        // Perform clustering
        auto [anchors, clusters] = greedySequentialClusteringHamming(transposed, threshold);

        // Output results
        std::cout << "Found " << anchors.size() << " clusters." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}