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
#include <omp.h>

// Class to precompute and store the indices of set bits in a bit vector
class BitIndexStore {
private:
    std::vector<size_t> set_bit_indices; // Vector to store indices of set bits
    size_t bit_vector_size = 0;         // Total size of the original bit vector

public:
    BitIndexStore(const std::vector<bool>& bit_vector) {
        bit_vector_size = bit_vector.size();
        for (size_t i = 0; i < bit_vector.size(); ++i) {
            if (bit_vector[i]) {
                set_bit_indices.push_back(i);
            }
        }
    }

    BitIndexStore(size_t size) : bit_vector_size(size) {}

    void set_bit(size_t index) {
        if (index >= bit_vector_size) {
            throw std::out_of_range("Index out of range in BitIndexStore::set_bit");
        }
        if (std::find(set_bit_indices.begin(), set_bit_indices.end(), index) == set_bit_indices.end()) {
            set_bit_indices.push_back(index);
        }
    }

    const std::vector<size_t>& get_set_bit_indices() const {
        return set_bit_indices;
    }

    size_t count_set_bits() const {
        return set_bit_indices.size();
    }

    size_t size() const {
        return bit_vector_size;
    }

    bool is_bit_set(size_t index) const {
        return std::find(set_bit_indices.begin(), set_bit_indices.end(), index) != set_bit_indices.end();
    }
};

// Calculate total set bits in a vector of BitIndexStore
size_t total_set_bits(const std::vector<BitIndexStore>& columns) {
    size_t total = 0;
    for (const auto& column : columns) {
        total += column.count_set_bits();
    }
    return total;
}

// Compute Hamming distance
size_t compute_hamming_distance_index(const BitIndexStore& bv1, const BitIndexStore& bv2) {
    const auto& indices1 = bv1.get_set_bit_indices();
    const auto& indices2 = bv2.get_set_bit_indices();

    size_t i = 0, j = 0;
    size_t hamming = 0;

    while (i < indices1.size() && j < indices2.size()) {
        if (indices1[i] == indices2[j]) {
            ++i;
            ++j;
        } else if (indices1[i] < indices2[j]) {
            ++hamming;
            ++i;
        } else {
            ++hamming;
            ++j;
        }
    }

    hamming += (indices1.size() - i);
    hamming += (indices2.size() - j);

    return hamming;
}

// Generate random columns
std::vector<BitIndexStore> generate_random_columns(size_t num_columns, size_t num_rows, double density) {
    if (density < 0.0 || density > 1.0) {
        throw std::invalid_argument("Density must be between 0.0 and 1.0.");
    }

    std::vector<BitIndexStore> columns;
    columns.reserve(num_columns);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < num_columns; ++i) {
        std::vector<bool> bit_vector(num_rows, false);
        for (size_t j = 0; j < num_rows; ++j) {
            bit_vector[j] = (dis(gen) < density);
        }
        columns.emplace_back(bit_vector);
    }

    return columns;
}

// Compute XOR between two BitIndexStore objects
BitIndexStore compute_xor(const BitIndexStore& bv1, const BitIndexStore& bv2) {
    const auto& indices1 = bv1.get_set_bit_indices();
    const auto& indices2 = bv2.get_set_bit_indices();
    size_t size = std::max(bv1.size(), bv2.size());

    std::vector<size_t> xor_indices;

    size_t i = 0, j = 0;
    while (i < indices1.size() && j < indices2.size()) {
        if (indices1[i] == indices2[j]) {
            ++i;
            ++j;
        } else if (indices1[i] < indices2[j]) {
            xor_indices.push_back(indices1[i]);
            ++i;
        } else {
            xor_indices.push_back(indices2[j]);
            ++j;
        }
    }

    while (i < indices1.size()) {
        xor_indices.push_back(indices1[i]);
        ++i;
    }

    while (j < indices2.size()) {
        xor_indices.push_back(indices2[j]);
        ++j;
    }

    BitIndexStore result(size);
    for (size_t index : xor_indices) {
        result.set_bit(index);
    }

    return result;
}

// Load columns from folder
std::vector<BitIndexStore> load_columns_from_folder(const std::string& folder_path) {
    std::vector<BitIndexStore> columns;

    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        const std::string file_name = entry.path().filename().string();
        if (file_name.rfind("anno", 0) == 0 && file_name.size() > 4 && file_name.substr(file_name.size() - 4) == ".bin") {
            std::ifstream file(entry.path(), std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file: " + file_name);
            }

            std::vector<bool> bit_vector;
            uint8_t byte;

            while (file.read(reinterpret_cast<char*>(&byte), 1)) {
                for (int i = 0; i < 8; ++i) {
                    bit_vector.push_back(byte & (1 << i));
                }
            }

            columns.emplace_back(bit_vector);

            std::cout << "Loaded column: " << file_name << std::endl;
        }
    }

    if (columns.empty()) {
        throw std::runtime_error("No valid files found in the folder.");
    }

    return columns;
}

// Transpose columns
std::vector<BitIndexStore> transpose_columns(const std::vector<BitIndexStore>& columns) {
    if (columns.empty() || columns[0].size() == 0) {
        throw std::runtime_error("Cannot transpose empty columns.");
    }

    size_t num_columns = columns.size();
    size_t num_rows = columns[0].size();

    std::cout << "Transposing " << num_columns << " columns with " << num_rows << " rows." << std::endl;

    std::vector<BitIndexStore> transposed(num_rows, BitIndexStore(num_columns));

    for (size_t col = 0; col < num_columns; ++col) {
        for (size_t row : columns[col].get_set_bit_indices()) {
            transposed[row].set_bit(col);
        }
    }

    return transposed;
}

// Greedy Sequential Clustering
std::pair<std::vector<size_t>, std::vector<std::vector<size_t>>>
greedySequentialClusteringHamming(const std::vector<BitIndexStore>& data, size_t threshold) {
    size_t num_points = data.size();
    ProgressBar progressBar(50, "Clustering Progress", num_points);

    std::vector<size_t> anchors;
    std::vector<std::vector<size_t>> clusters;

    for (size_t i = 0; i < num_points; ++i) {
        const BitIndexStore& current = data[i];
        bool assigned_to_cluster = false;

        for (size_t j = 0; j < anchors.size(); ++j) {
            size_t anchor_idx = anchors[j];
            const BitIndexStore& anchor = data[anchor_idx];

            if (compute_hamming_distance_index(current, anchor) <= threshold) {
                clusters[j].push_back(i);
                assigned_to_cluster = true;
                break;
            }
        }

        if (!assigned_to_cluster) {
            anchors.push_back(i);
            clusters.push_back({ i });
        }

        progressBar.update();
    }

    progressBar.finish();
    return { anchors, clusters };
}

// std::vector<BitIndexStore> perform_rowdiff(
//     const std::vector<BitIndexStore>& data,
//     const std::vector<size_t>& anchors,
//     const std::vector<std::vector<size_t>>& clusters_points) {
//     // Make a copy of the data to store the result
//     std::vector<BitIndexStore> result = data;

//     // Iterate over each cluster
//     for (size_t cluster_idx = 0; cluster_idx < anchors.size(); ++cluster_idx) {
//         size_t anchor_idx = anchors[cluster_idx];
//         const auto& anchor = result[anchor_idx]; // Anchor row

//         // Iterate over each row in the current cluster
//         for (size_t row_idx : clusters_points[cluster_idx]) {
//             if (row_idx == anchor_idx) {
//                 continue; // Skip the anchor row itself
//             }

//             // Apply XOR operation between the anchor and the target row
//             result[row_idx] = compute_xor(result[row_idx], anchor);
//         }
//     }

//     return result;
// }

std::vector<BitIndexStore> perform_rowdiff(
    const std::vector<BitIndexStore>& data,
    const std::vector<size_t>& anchors,
    const std::vector<std::vector<size_t>>& clusters_points) {
    // Make a copy of the data to store the result
    std::vector<BitIndexStore> result = data;

    ProgressBar progressBar(50, "Rowdiff Progress", anchors.size(), 1);
    // Parallelize the outer loop over clusters
    // #pragma omp parallel for
    for (size_t cluster_idx = 0; cluster_idx < anchors.size(); ++cluster_idx) {
        size_t anchor_idx = anchors[cluster_idx];
        const auto& anchor = result[anchor_idx]; // Anchor row

        // Iterate over each row in the current cluster
        for (size_t row_idx : clusters_points[cluster_idx]) {
            if (row_idx == anchor_idx) {
                continue; // Skip the anchor row itself
            }

            // Apply XOR operation between the anchor and the target row
            result[row_idx] = compute_xor(result[row_idx], anchor);
        }
        progressBar.update();
    }

    return result;
}

// Save columns to files
void save_columns_to_files(const std::vector<BitIndexStore>& columns, const std::string& prefix) {
    for (size_t col = 0; col < columns.size(); ++col) {
        std::ofstream file(prefix + "_col" + std::to_string(col) + ".bin", std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + prefix + "_col" + std::to_string(col) + ".bin");
        }

        const auto& indices = columns[col].get_set_bit_indices();
        std::vector<bool> bits(columns[col].size(), false);

        for (size_t idx : indices) {
            bits[idx] = true;
        }

        uint8_t byte = 0;
        for (size_t i = 0; i < bits.size(); ++i) {
            if (bits[i]) {
                byte |= (1 << (i % 8));
            }
            if (i % 8 == 7 || i == bits.size() - 1) {
                file.write(reinterpret_cast<char*>(&byte), 1);
                byte = 0;
            }
        }
    }
}

// Main function
int main(int argc, char* argv[]) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: " << argv[0] << " <folderPath> <threshold> <random|load>" << std::endl;
            return 1;
        }

        std::string folder_path = argv[1];
        size_t threshold = std::stoul(argv[2]);
        std::string mode = argv[3];

        std::vector<BitIndexStore> columns;

        if (mode == "random") {
            columns = generate_random_columns(32, 6'000'000, 0.1);
        } else if (mode == "load") {
            columns = load_columns_from_folder(folder_path);
        } else {
            throw std::invalid_argument("Invalid mode. Use 'random' or 'load'.");
        }

        size_t num_columns = columns.size();
        size_t num_rows = columns[0].size();
        if (mode == "random") {
            std::cout << "Generated " << num_columns << " columns, each with " << num_rows << " rows." << std::endl;
        } else {
            std::cout << "Loaded " << num_columns << " columns, each with " << num_rows << " rows." << std::endl;
        }

        std::cout << "Total set bits before clustering: " << total_set_bits(columns) << std::endl;

        auto transposed = transpose_columns(columns);
        auto [anchors, clusters] = greedySequentialClusteringHamming(transposed, threshold);

        // Print the number of clusters
        std::cout << "Found " << anchors.size() << " clusters." << std::endl;

        auto rowdiff_result = perform_rowdiff(transposed, anchors, clusters);
        std::cout << std::endl << "Finished rowdiff operation." << std::endl;
        auto final_result = transpose_columns(rowdiff_result);
        std::cout << "Finished transposing back." << std::endl;

        size_t final_set_bits = total_set_bits(final_result);
        std::cout << "Total set bits after rowdiff and transpose back: " << final_set_bits << std::endl;

        save_columns_to_files(final_result, "output");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}