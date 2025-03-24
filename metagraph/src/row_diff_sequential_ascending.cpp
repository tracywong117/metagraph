#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <random>
#include <sdsl/bit_vectors.hpp>
#include <filesystem>
#include <algorithm>
#include <limits>
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
        if (file_name.rfind("anno", 0) == 0 && file_name.size() > 4
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

// Function to calculate Hamming distance between two bit vectors
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

// Function to perform Greedy Sequential Clustering
std::pair<std::vector<size_t>, std::vector<std::vector<size_t>>>
greedySequentialClusteringHamming(const std::vector<std::unique_ptr<bit_vector>>& data,
                             size_t threshold) {

    size_t num_points = data.size();

    // Initialize the progress bar
    ProgressBar progressBar(50, "Clustering Progress", num_points);

    std::vector<size_t> anchors; // Anchor indices
    std::vector<std::vector<size_t>> clusters; // Clusters of indices
    // int avgNumComparisons = 0; // For debugging

    for (size_t i = 0; i < num_points; ++i) {
        const bit_vector& point = *data[i];
        bool assigned_to_cluster = false;

        // Check if the point can be assigned to an existing cluster
        for (size_t j = 0; j < anchors.size(); ++j) {
            // avgNumComparisons++;
            size_t anchor_idx = anchors[j];
            if (hamming_distance(point, *data[anchor_idx]) < threshold) {
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

// Perform Rowdiff based on anchors and clusters_points
std::vector<std::unique_ptr<bit_vector>>
perform_rowdiff(const std::vector<std::unique_ptr<bit_vector>>& data,
                const std::vector<size_t>& anchors,
                const std::vector<std::vector<size_t>>& clusters_points) {
    size_t num_rows = data.size();
    std::vector<std::unique_ptr<bit_vector>> result;

    // Copy the data
    for (const auto& row : data) {
        result.push_back(make_unique<bit_vector>(*row));
    }

    // Perform rowdiff
    for (size_t cluster_idx = 0; cluster_idx < anchors.size(); ++cluster_idx) {
        size_t anchor_idx = anchors[cluster_idx];
        bit_vector& anchor_row = *result[anchor_idx];

        for (size_t row_idx : clusters_points[cluster_idx]) {
            if (row_idx == anchor_idx)
                continue; // Skip the anchor row

            bit_vector& target_row = *result[row_idx];
            for (size_t col = 0; col < target_row.size(); ++col) {
                target_row[col] = anchor_row[col] ^ target_row[col]; // XOR operation
            }
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
        // // Step 1: Generate random columns
        // size_t num_columns = 32; // Number of columns
        // size_t num_rows = 60000000;   // Number of rows
        // auto columns = generate_random_columns(num_columns, num_rows);
        // save_columns_to_files(columns, "column");

        // Alternatively, load columns from binary files
        // std::string folder_path
        //         = "/media/data/tracy/metagraph/data"; // Replace with your folder path
        // auto columns = load_columns_from_folder(folder_path);

        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " <folderPath> <threshold>" << std::endl;
            return 1;
        }

        std::string folder_path = argv[1];
        size_t threshold = std::stoul(argv[2]); // Maximum Hamming distance for clustering

        // Print the number of columns and rows
        size_t num_columns = columns.size();
        size_t num_rows = columns[0]->size();
        std::cout << "Loaded " << num_columns << " columns, each with " << num_rows
                  << " rows." << std::endl;

        // Count set bits before transpose
        auto initial_set_bits_count = count_set_bits(columns);
        // print_set_bits_count(initial_set_bits_count, "Set bits in each column BEFORE transpose");

        // print the total number of set bits
        size_t total_set_bits = std::accumulate(initial_set_bits_count.begin(),
                                                initial_set_bits_count.end(), 0);
        std::cout << "Total set bits BEFORE transpose: " << total_set_bits << std::endl;

        // Step 2: Transpose the data ([num_columns, num_rows] -> [num_rows, num_columns])
        auto transposed = transpose_columns(columns);

        // Step 2.5: Sort rows by the number of set bits (ascending order)
        std::sort(transposed.begin(), transposed.end(), [](const std::unique_ptr<bit_vector>& a,
                                                             const std::unique_ptr<bit_vector>& b) {
            // Count the set bits in each row using std::count
            size_t count_a = std::count(a->begin(), a->end(), true);
            size_t count_b = std::count(b->begin(), b->end(), true);
            return count_a < count_b;
        });

        // Optionally, you can print the set bits count of a few rows after sorting
        std::cout << "After sorting rows by set bits:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(10), transposed.size()); ++i) {
            size_t count = std::count(transposed[i]->begin(), transposed[i]->end(), true);
            std::cout << "Row " << i << ": " << count << " set bits" << std::endl;
        }

        // Step 3: Perform Greedy Sequential Clustering
        auto [anchors, clusters_points]
                = greedySequentialClusteringHamming(transposed, threshold);

        // Print the number of clusters
        std::cout << "Found " << anchors.size() << " clusters." << std::endl;

        // Step 4: Perform Rowdiff
        auto rowdiff_result = perform_rowdiff(transposed, anchors, clusters_points);

        // Step 5: Transpose back the result of Rowdiff
        auto final_result = transpose_columns(rowdiff_result);

        // Count set bits after transpose back
        auto final_set_bits_count = count_set_bits(final_result);
        // print_set_bits_count(final_set_bits_count, "Set bits in each column AFTER rowdiff and transpose");

        // print the total number of set bits
        size_t total_set_bits_after = std::accumulate(final_set_bits_count.begin(),
                                                      final_set_bits_count.end(), 0);
        std::cout << "Total set bits AFTER rowdiff and transpose: " << total_set_bits_after
                  << std::endl;

        // Step 6: Save the final result to files (one file per column)
        save_columns_to_files(final_result, "column_row_diff");

        std::cout << "Bit vectors written to files." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}