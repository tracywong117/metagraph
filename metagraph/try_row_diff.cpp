#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <sdsl/bit_vectors.hpp> // For sdsl::bit_vector

// Alias for readability
using bit_vector = sdsl::bit_vector;

// Custom implementation of `std::make_unique` for C++11
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Function to generate random column data for testing
std::vector<std::unique_ptr<bit_vector>> generate_random_columns(size_t num_columns, size_t num_rows) {
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

// A function to transpose a 2D bit vector
std::vector<std::unique_ptr<bit_vector>> transpose_columns(const std::vector<std::unique_ptr<bit_vector>>& columns) {
    if (columns.empty()) {
        throw std::runtime_error("No columns to transpose.");
    }

    // Determine the number of rows and columns
    size_t num_columns = columns.size();
    size_t num_rows = columns[0]->size();

    // Verify that all columns have the same length
    for (const auto& column : columns) {
        if (column->size() != num_rows) {
            throw std::runtime_error("All columns must have the same length for transposition.");
        }
    }

    // Create transposed bit vectors
    std::vector<std::unique_ptr<bit_vector>> transposed(num_rows);
    for (size_t i = 0; i < num_rows; ++i) {
        transposed[i] = make_unique<bit_vector>(num_columns, 0); // Each row will become a column
    }

    // Perform the transposition
    for (size_t col = 0; col < num_columns; ++col) {
        for (size_t row = 0; row < num_rows; ++row) {
            (*transposed[row])[col] = (*columns[col])[row];
        }
    }

    return transposed;
}

// A function to perform rowdiff operation
std::vector<std::unique_ptr<bit_vector>> perform_rowdiff(
    const std::vector<std::unique_ptr<bit_vector>>& transposed_data,
    const std::vector<size_t>& anchors,
    const std::vector<std::vector<size_t>>& cluster_points) {

    size_t num_rows = transposed_data.size();

    // Validate inputs
    if (anchors.size() != cluster_points.size()) {
        throw std::runtime_error("Anchors size must match the number of clusters.");
    }

    for (const auto& cluster : cluster_points) {
        for (size_t idx : cluster) {
            if (idx >= num_rows) {
                throw std::runtime_error("Cluster point index out of bounds.");
            }
        }
    }

    // Create a copy of the transposed data
    std::vector<std::unique_ptr<bit_vector>> result;
    for (const auto& row : transposed_data) {
        result.push_back(make_unique<bit_vector>(*row));
    }

    // Perform the XOR operation for each cluster
    for (size_t cluster_idx = 0; cluster_idx < anchors.size(); ++cluster_idx) {
        size_t anchor_idx = anchors[cluster_idx];

        if (anchor_idx >= num_rows) {
            throw std::runtime_error("Anchor index out of bounds.");
        }

        bit_vector& anchor_row = *result[anchor_idx];

        for (size_t row_idx : cluster_points[cluster_idx]) {
            if (row_idx == anchor_idx) {
                continue; // Skip anchor row
            }

            bit_vector& target_row = *result[row_idx];

            // Perform XOR operation using temporary variables for `sdsl::int_vector<1>::reference`
            for (size_t col = 0; col < target_row.size(); ++col) {
                bool anchor_bit = anchor_row[col];  // Extract anchor bit
                bool target_bit = target_row[col];  // Extract target bit
                target_row[col] = anchor_bit ^ target_bit; // Perform XOR and assign
            }
        }
    }

    return result;
}

int main() {
    try {
        // Step 1: Generate random data
        size_t num_columns = 5; // Number of columns
        size_t num_rows = 10;    // Number of rows
        std::vector<std::unique_ptr<bit_vector>> columns = generate_random_columns(num_columns, num_rows);

        // Print the generated random data (columns)
        std::cout << "Generated Random Columns:" << std::endl;
        for (size_t col = 0; col < columns.size(); ++col) {
            std::cout << "Column " << col << ": ";
            for (size_t row = 0; row < columns[col]->size(); ++row) {
                std::cout << (*columns[col])[row];
            }
            std::cout << std::endl;
        }

        // Step 2: Transpose the data
        std::vector<std::unique_ptr<bit_vector>> transposed = transpose_columns(columns);

        // Print the transposed data
        std::cout << "\nTransposed Data:" << std::endl;
        for (size_t row = 0; row < transposed.size(); ++row) {
            std::cout << "Row " << row << ": ";
            for (size_t col = 0; col < transposed[row]->size(); ++col) {
                std::cout << (*transposed[row])[col];
            }
            std::cout << std::endl;
        }

        // Step 3: Define anchors and cluster points
        // Example: 2 clusters
        std::vector<size_t> anchors = {0, 2}; // Row indices of anchor points
        std::vector<std::vector<size_t>> cluster_points = {
            {0, 1, 3}, // Cluster 1: rows 0, 1, 3
            {2, 4, 5}  // Cluster 2: rows 2, 4, 5
        };

        // Step 4: Perform rowdiff operation
        std::vector<std::unique_ptr<bit_vector>> result = perform_rowdiff(transposed, anchors, cluster_points);

        // Print the result after rowdiff
        std::cout << "\nResult After Rowdiff:" << std::endl;
        for (size_t row = 0; row < result.size(); ++row) {
            std::cout << "Row " << row << ": ";
            for (size_t col = 0; col < result[row]->size(); ++col) {
                std::cout << (*result[row])[col];
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}