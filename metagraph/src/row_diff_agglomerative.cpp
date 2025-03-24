#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <random>
#include <sdsl/bit_vectors.hpp>
#include <filesystem>
#include <algorithm>
#include <limits>
#include <climits> // For INT_MAX and other integer limits
#include <unordered_map>
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

// Function to compute the Hamming distance between two sdsl::bit_vector objects
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

// Class for efficient sampling, insertion, and deletion
class LargeDataSampler
{
private:
    std::vector<size_t> data;               // Stores the elements
    std::unordered_map<int, size_t> indexMap; // Maps element to its index in the vector
    std::mt19937 rng;                    // Random number generator

public:
    LargeDataSampler() : rng(std::random_device{}()) {}

    int get(int i)
    {
        return data[i];
    }

    // Insert an element (O(1))
    void insert(int val)
    {
        if (indexMap.find(val) != indexMap.end())
            return; // Element already exists
        indexMap[val] = data.size();
        data.push_back(val);
    }

    // Delete an element (O(1))
    void remove(int val)
    {
        auto it = indexMap.find(val);
        if (it == indexMap.end())
            return; // Element not found

        // Get the index of the element to be removed
        size_t idx = it->second;

        // Swap the element with the last element in the vector
        int lastElement = data.back();
        data[idx] = lastElement;
        indexMap[lastElement] = idx;

        // Remove the last element
        data.pop_back();
        indexMap.erase(val);
    }

    // Sample a random element (O(1))
    int sample()
    {
        if (data.empty())
            throw std::runtime_error("Set is empty!");
        std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
        return data[dist(rng)];
    }

    // Get the current size of the dataset
    size_t size() const
    {
        return data.size();
    }

    // Check if the dataset is empty
    bool empty() const
    {
        return data.empty();
    }
};

std::pair<std::vector<size_t>, std::vector<std::vector<size_t>>>
greedyAgglomerativeClusteringHamming(const std::vector<std::unique_ptr<bit_vector>> &data,
                                     int numClusters,
                                     int samplingSize)
{
    size_t nPoints = data.size();
    size_t numberLoop = nPoints - numClusters;

    // Cluster assignment and cluster representatives
    std::vector<size_t> pointsAssignCluster(nPoints);
    std::iota(pointsAssignCluster.begin(), pointsAssignCluster.end(), 0); // Initially, each point is its own cluster
    std::cout << "Initialized pointsAssignCluster" << std::endl;

    std::vector<std::unique_ptr<bit_vector>> representatives; // Each cluster initially represented by its own point
    for (const auto &v : data)
    {
        representatives.push_back(make_unique<bit_vector>(*v));
    }
    std::cout << "Initialized representatives" << std::endl;

    std::unordered_map<int, std::vector<size_t>> clustersPoints; // Cluster points map
    LargeDataSampler existingClusters;

    // Initialize clustersPoints and existingClusters
    for (size_t i = 0; i < nPoints; ++i)
    {
        clustersPoints[i] = {static_cast<size_t>(i)}; // Each cluster starts with its own point
        existingClusters.insert(i);               // Add each cluster ID to the sampler
    }
    std::cout << "Initialized clustersPoints and existingClusters" << std::endl;

    ProgressBar progressBar(50, "Clustering Progress", numberLoop); // Initialize progress bar

    for (size_t iter = 0; iter < numberLoop; ++iter)
    {
        // Step 1: Randomly sample cluster pairs
        std::vector<size_t> sampled1;
        std::vector<size_t> sampled2;
        for (int i = 0; i < samplingSize; ++i)
        {
            sampled1.push_back(existingClusters.sample());
            sampled2.push_back(existingClusters.sample());
        }

        // Step 2: Find the pair with the smallest Hamming distance
        int minDistance = INT_MAX;
        std::pair<int, int> bestPair = {-1, -1};
        for (size_t i = 0; i < sampled1.size(); ++i)
        {
            int c1 = sampled1[i];
            int c2 = sampled2[i];

            if (c1 == c2)
                continue; // Skip if clusters are the same

            int dist = hamming_distance(*representatives[c1], *representatives[c2]);
            if (dist < minDistance)
            {
                minDistance = dist;
                bestPair = {c1, c2};
            }
        }

        int c1 = bestPair.first;
        int c2 = bestPair.second;

        if (c1 == -1 || c2 == -1)
            continue; // Skip invalid pairs

        // Step 3: Merge the selected pair of clusters
        if (clustersPoints[c1].size() < clustersPoints[c2].size())
        {
            std::swap(c1, c2); // Ensure c1 is the larger cluster
        }

        // Merge c2 into c1
        for (int point : clustersPoints[c2])
        {
            pointsAssignCluster[point] = c1;
            clustersPoints[c1].push_back(point);
        }

        // Update the representative for c1 (e.g., recompute)
        for (size_t i = 0; i < representatives[c1]->size(); ++i)
        {
            (*representatives[c1])[i] = (*representatives[c1])[i] & (*representatives[c2])[i]; // Bitwise AND
        }

        // Remove c2 from active clusters
        existingClusters.remove(c2);

        // Update progress bar
        progressBar.update();
    }

    // Final calculations and reporting
    progressBar.finish();

    // Extract anchors and clusters
    std::vector<size_t> anchors;
    std::vector<std::vector<size_t>> clusters;

    for (size_t i = 0; i < existingClusters.size(); ++i)
    {
        anchors.push_back(existingClusters.get(i));
        clusters.push_back(clustersPoints[existingClusters.get(i)]);

    }

    return {anchors, clusters};
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
        std::string folder_path
                = "/media/data/tracy/metagraph/data"; // Replace with your folder path
        auto columns = load_columns_from_folder(folder_path);

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

        // Step 2: Transpose the data
        auto transposed = transpose_columns(columns);

        // Step 3: Perform Greedy Agglomerative Clustering
        int numClusters = 50'000; // Desired number of clusters
        int samplingSize = 100; // Number of random cluster pairs to sample
        auto [anchors, clusters_points] = greedyAgglomerativeClusteringHamming(transposed, numClusters, samplingSize);

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