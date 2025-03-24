#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "method_constructors.hpp"
#include "annotation/annotation_converters.hpp"
#include "annotation/representation/column_compressed/annotate_column_compressed.hpp"
#include "graph/annotated_dbg.hpp"
#include "common/vectors/vector_algorithm.hpp"
#include "annotation/binary_matrix/multi_brwt/clustering.hpp"
#include "cli/transform_annotation.hpp"
#include "annotation/binary_matrix/multi_brwt/brwt_builders.hpp"

namespace {

using namespace mtg;
using namespace mtg::annot::matrix;

static const Eigen::IOFormat
        CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n");

// Function to get densities for columns
std::vector<double> get_densities(uint64_t num_cols, std::vector<double> densities) {
    if (densities.size() == 1) {
        densities.assign(num_cols, densities[0]);
    } else if (densities.size() != num_cols) {
        std::cout << "ERROR: wrong number of column counts" << std::endl;
        exit(1);
    }
    return densities;
}

// Build BRWT matrix from random columns
template <size_t density_numerator,
          size_t density_denominator,
          size_t rows_arg = 8,
          size_t cols_arg = 8,
          size_t unique_arg = 2, // number of unique columns
          size_t arity_arg = 2, // arity of the BRWT
          bool greedy_arg = true, // greedy matching
          size_t relax_arg = 2>
void BM_BRWTCompressSparse() {
    DataGenerator generator;
    generator.set_seed(42);

    // Set density argument
    auto density_arg = std::vector<double>(unique_arg,
                                           static_cast<double>(density_numerator)
                                                   / density_denominator);

    // Generate random columns
    std::vector<std::unique_ptr<bit_vector>> generated_columns
            = generator.generate_random_columns(
                    rows_arg, unique_arg, get_densities(unique_arg, density_arg),
                    std::vector<uint32_t>(unique_arg, cols_arg / unique_arg));

    std::cout << "Generated " << generated_columns.size() << " columns with " << rows_arg
              << " rows.\n";

    // Print the first column of the generated columns for validation
    std::cout << "Sample of generated columns:\n";
    for (size_t c = 0; c < cols_arg; c++) {
        for (size_t i = 0; i < generated_columns[c]->size(); ++i) {
            std::cout << (*generated_columns[c])[i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\n";

    // Generate BRWT matrix
    std::unique_ptr<annot::matrix::BinaryMatrix> matrix;

    matrix = experiments::generate_brwt_from_rows(std::move(generated_columns), arity_arg,
                                                  greedy_arg, relax_arg);

    // Print the matrix
    // std::cout << "BRWT matrix:\n";
    // matrix->print_tree_structure(std::cout);
    if (const auto* brwt_matrix
        = dynamic_cast<const mtg::annot::matrix::BRWT*>(matrix.get())) {
        brwt_matrix->print_tree_structure(std::cout);
    } else {
        std::cerr << "Error: The provided matrix is not a BRWT instance." << std::endl;
    }
    std::cout << "BRWT matrix generated with arity = " << arity_arg
              << ", greedy = " << greedy_arg << ", relax = " << relax_arg << ".\n";
}

// Query columns and rows from BRWT matrix
template <size_t rows_arg = 8,
          size_t cols_arg = 8,
          size_t unique_arg = 2,
          size_t arity_arg = 2,
          bool greedy_arg = true,
          size_t relax_arg = 2>
void BM_BRWTQueryRows(int range) {
    DataGenerator generator;
    generator.set_seed(42);

    // Set density argument
    auto density_arg = std::vector<double>(unique_arg, range / 100.);
    std::vector<std::unique_ptr<bit_vector>> generated_columns
            = generator.generate_random_columns(
                    rows_arg, unique_arg, get_densities(unique_arg, density_arg),
                    std::vector<uint32_t>(unique_arg, cols_arg / unique_arg));

    std::cout << "Generated " << generated_columns.size()
              << " columns for query with density = " << (range / 100.0) << ".\n";

    // Print the first column of the generated columns for validation
    std::cout << "Sample of generated columns:\n";
    for (size_t c = 0; c < cols_arg; c++) {
        for (size_t i = 0; i < generated_columns[c]->size(); ++i) {
            std::cout << (*generated_columns[c])[i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\n";

    // Generate BRWT matrix
    std::unique_ptr<annot::matrix::BinaryMatrix> matrix
            = experiments::generate_brwt_from_rows(std::move(generated_columns),
                                                   arity_arg, greedy_arg, relax_arg);

    // Query columns
    for (size_t i = 0; i < cols_arg; i++) {
        std::vector queried_col
                = matrix->get_column(i); // It returns the set bits positions in the column

        // Print the queried column
        std::cout << "Queried column " << i << " :\n";
        for (size_t j = 0; j < queried_col.size(); ++j) {
            std::cout << queried_col[j] << " ";
        }
        std::cout << "\n";
    }

    // Query rows
    std::vector<annot::matrix::BinaryMatrix::Row> rows_to_query = { 0, 1, 5 };
    std::vector<annot::matrix::BinaryMatrix::SetBitPositions> row_results
            = matrix->get_rows(rows_to_query);

    for (size_t i = 0; i < rows_to_query.size(); ++i) {
        std::cout << "Row " << rows_to_query[i] << ": ";
        const auto& set_bits = row_results[i];
        for (const auto& bit_pos : set_bits) {
            std::cout << bit_pos << " ";
        }
        std::cout << std::endl;
    }
}

// Function to load and unpack bits from a binary file
std::vector<bool> load_annotation_bit_vector_little(const std::string& file_path) {
    // Open the file in binary mode
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    // Read the file contents into a vector of uint8_t
    std::vector<uint8_t> byte_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

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

// // Function to load a single column from a binary annotation file
// std::vector<std::unique_ptr<bit_vector>> load_column_from_file(
//     const std::string &file_path, uint64_t column_length) {

//     // Load the bit vector from the binary file
//     std::vector<bool> bit_vector_bool = load_annotation_bit_vector_little(file_path);

//     // Verify that the bit vector has at least `column_length` bits
//     if (bit_vector_bool.size() < column_length) {
//         throw std::runtime_error("File does not contain enough bits for the requested column length.");
//     }

//     // Create an `sdsl::bit_vector`
//     sdsl::bit_vector column(column_length, 0);
//     for (size_t i = 0; i < column_length; ++i) {
//         if (bit_vector_bool[i]) {
//             column[i] = 1;
//         }
//     }

//     // Wrap the `sdsl::bit_vector` in a `std::unique_ptr` of `bit_vector` and return it
//     std::vector<std::unique_ptr<bit_vector>> columns(1); // Create a vector of `unique_ptr` with one element
//     columns[0] = std::make_unique<bit_vector_stat>(std::move(column));

//     return columns;
// }

// Function to load columns from binary annotation files in a folder
std::vector<std::unique_ptr<bit_vector>> load_columns_from_folder(const std::string& folder_path) {
    namespace fs = std::filesystem; // Alias for readability

    std::vector<std::unique_ptr<bit_vector>> columns; // To store the columns
    std::vector<std::string> file_paths;              // To store matching file paths

    // Iterate through the folder and find files with prefix "anno" and extension ".bin"
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            const std::string file_name = entry.path().filename().string();
            if (file_name.rfind("column", 0) == 0 && file_name.substr(file_name.size() - 4) == ".bin") {
                file_paths.push_back(entry.path().string());
            }
        }
    }

    // Check if there are no matching files
    if (file_paths.empty()) {
        throw std::runtime_error("No files with prefix 'column' and extension '.bin' found in the folder.");
    }

    // Determine the column length from the size of the first file
    const std::string first_file = file_paths[0];
    std::ifstream first_file_stream(first_file, std::ios::binary | std::ios::ate); // Open in binary mode, seek to end
    if (!first_file_stream.is_open()) {
        throw std::runtime_error("Failed to open file: " + first_file);
    }
    std::streamsize file_size = first_file_stream.tellg(); // Get file size in bytes
    first_file_stream.close();

    // Each byte contains 8 bits, so column_length = file_size * 8
    uint64_t column_length = static_cast<uint64_t>(file_size) * 8;

    // Load all files into columns
    for (const std::string& file_path : file_paths) {
        // Load the bit vector from the binary file
        std::vector<bool> bit_vector_bool = load_annotation_bit_vector_little(file_path);

        // Verify that the bit vector has at least `column_length` bits
        if (bit_vector_bool.size() < column_length) {
            throw std::runtime_error("File does not contain enough bits for the expected column length.");
        }

        // Create an `sdsl::bit_vector`
        sdsl::bit_vector column(column_length, 0);
        for (size_t i = 0; i < column_length; ++i) {
            if (bit_vector_bool[i]) {
                column[i] = 1;
            }
        }

        // Wrap the `sdsl::bit_vector` in a `std::unique_ptr` of `bit_vector`
        columns.push_back(std::make_unique<bit_vector_stat>(std::move(column)));
    }

    return columns;
}

// template <size_t rows_arg = 7, size_t cols_arg = 16, size_t unique_arg = 16, size_t sample_rows = 4>
// void BM_BRWTLinkageMatrix() {
//     DataGenerator generator;
//     generator.set_seed(42);

//     // Set density argument
//     auto density_arg = std::vector<double>(unique_arg, 0.5);

//     // Generate random columns
//     std::vector<std::unique_ptr<bit_vector>> generated_columns
//             = generator.generate_random_columns(
//                     rows_arg, unique_arg, get_densities(unique_arg, density_arg),
//                     std::vector<uint32_t>(unique_arg, cols_arg / unique_arg));

//     std::cout << "Generated " << generated_columns.size() << " columns for linkage matrix.\n";

//     // Print the first column of the generated columns for validation
//     std::cout << "Sample of generated columns:\n";
//     for (size_t c = 0; c < cols_arg; ++c) {
//         for (size_t i = 0; i < generated_columns[c]->size(); ++i) {
//             std::cout << (*generated_columns[c])[i] << " ";
//         }
//         std::cout << "\n";
//     }
//     std::cout << "\n\n";

//     // Subsample rows and construct subcolumns
//     std::vector<uint64_t> row_indexes; // To store the sampled row indexes
//     std::vector<std::unique_ptr<sdsl::bit_vector>> subcolumn_ptrs; // To store the subcolumns
//     std::vector<uint64_t> column_ids; // To rearrange the columns in their original order
//     uint64_t num_rows = 0;

//     // Mutex for thread-safe operations
//     std::mutex mu;

//     // ThreadPool for sampling (if you want to parallelize)
//     ThreadPool subsampling_pool(get_num_threads(), 1);

//     for (size_t i = 0; i < generated_columns.size(); ++i) {
//         subsampling_pool.enqueue([&, i]() {
//             sdsl::bit_vector* subvector;

//             {
//                 std::lock_guard<std::mutex> lock(mu);

//                 // Initialize row indexes if empty
//                 if (row_indexes.empty()) {
//                     num_rows = generated_columns[i]->size();
//                     row_indexes = annot::matrix::sample_row_indexes(num_rows, sample_rows);
//                 } else if (generated_columns[i]->size() != num_rows) {
//                     std::cerr << "Error: Column size mismatch for column " << i << std::endl;
//                     exit(1);
//                 }

//                 subcolumn_ptrs.emplace_back(new sdsl::bit_vector());
//                 subvector = subcolumn_ptrs.back().get();
//                 column_ids.push_back(i);
//             }

//             // Subsample the column
//             *subvector = sdsl::bit_vector(row_indexes.size(), false);
//             for (size_t j = 0; j < row_indexes.size(); ++j) {
//                 if ((*generated_columns[i])[row_indexes[j]]) {
//                     (*subvector)[j] = true;
//                 }
//             }
//         });
//     }

//     subsampling_pool.join();

//     // Rearrange the columns in their original order
//     std::vector<sdsl::bit_vector> subcolumns(subcolumn_ptrs.size());
//     for (size_t i = 0; i < column_ids.size(); ++i) {
//         subcolumns.at(column_ids[i]) = std::move(*subcolumn_ptrs[i]);
//     }

//     // Print the subcolumns
//     std::cout << "Subcolumns for linkage matrix:\n";
//     for (size_t i = 0; i < subcolumns.size(); ++i) {
//         std::cout << "Subcolumn " << i << ": ";
//         for (size_t j = 0; j < subcolumns[i].size(); ++j) {
//             std::cout << subcolumns[i][j];
//         }
//         std::cout << std::endl;
//     }

//     // Compute the linkage matrix
//     auto linkage_matrix = annot::matrix::agglomerative_greedy_linkage(std::move(subcolumns),
//                                                                       get_num_threads());

//     // Print the resulting linkage matrix
//     for (int i = 0; i < linkage_matrix.rows(); ++i) {
//         std::cout << "Linkage " << i << ": ";
//         for (int j = 0; j < linkage_matrix.cols(); ++j) {
//             std::cout << linkage_matrix(i, j) << " "; // Access element (i, j)
//         }
//         std::cout << std::endl;
//     }

//     // Serialize the linkage matrix
//     auto linkage_file = "clustering.linkage";
//     std::ofstream out(linkage_file);
//     out << linkage_matrix.format(CSVFormat) << std::endl;
// }

std::vector<std::string> split_string(const std::string& string,
                                      const std::string& delimiter,
                                      bool skip_empty_parts) {
    if (!string.size())
        return {};

    if (!delimiter.size())
        return {
            string,
        };

    std::vector<std::string> result;

    size_t current_pos = 0;
    size_t delimiter_pos;

    while ((delimiter_pos = string.find(delimiter, current_pos)) != std::string::npos) {
        if (delimiter_pos > current_pos || !skip_empty_parts)
            result.push_back(string.substr(current_pos, delimiter_pos - current_pos));
        current_pos = delimiter_pos + delimiter.size();
    }
    if (current_pos < string.size()) {
        result.push_back(string.substr(current_pos));
    }

    assert(result.size());
    return result;
}

std::vector<std::vector<uint64_t>> parse_linkage_matrix(const std::string& filename) {
    std::ifstream in(filename);

    std::vector<std::vector<uint64_t>> linkage;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<std::string> parts = split_string(line, " ", true);
        if (parts.empty())
            continue;

        try {
            if (parts.size() != 4)
                throw std::runtime_error("Invalid format");

            uint64_t first = std::stoi(parts.at(0));
            uint64_t second = std::stoi(parts.at(1));
            uint64_t merged = std::stoi(parts.at(3));

            if (first == second || first >= merged || second >= merged) {
                exit(1);
            }

            while (linkage.size() <= merged) {
                linkage.push_back({});
            }

            linkage[merged].push_back(first);
            linkage[merged].push_back(second);

        } catch (const std::exception& e) {
            exit(1);
        }
    }

    return linkage;
}

// void BM_SerializeLoadLinkage_BRWT() {
//     // Load the linkage matrix
//     auto linkage_file = "clustering.linkage";
//     std::cout << "Loading linkage matrix from " << linkage_file << ":\n";
//     // auto linkage_matrix = cli::parse_linkage_matrix(linkage_file);
//     auto linkage_matrix = parse_linkage_matrix(
//             linkage_file); // write the function here instead of using cli::parse_linkage_matrix to avoid "missing linkage to the zlib" error

//     // linkage_matrix is a std::vector<std::vector<uint64_t>>
//     std::cout << "Linkage matrix loaded from " << linkage_file << ":\n";
//     for (size_t i = 0; i < linkage_matrix.size(); ++i) {
//         std::cout << "Linkage " << i << ": ";
//         for (size_t j = 0; j < linkage_matrix[i].size(); ++j) {
//             std::cout << linkage_matrix[i][j] << " ";
//         }
//         std::cout << std::endl;
//     }
// }

// Function to build BRWT from loaded columns
void BM_BRWTLinkageMatrix(std::vector<std::unique_ptr<bit_vector>>& columns, string linkage_file) {
    // Check if columns are non-empty
    if (columns.empty()) {
        std::cerr << "Error: Input columns are empty.\n";
        return;
    }

    // Subsample rows and construct subcolumns
    std::vector<uint64_t> row_indexes; // To store the sampled row indexes
    std::vector<std::unique_ptr<sdsl::bit_vector>> subcolumn_ptrs; // To store the subcolumns
    std::vector<uint64_t> column_ids; // To rearrange the columns in their original order
    uint64_t num_rows = 0;

    // Mutex for thread-safe operations
    std::mutex mu;

    // ThreadPool for sampling (if you want to parallelize)
    ThreadPool subsampling_pool(get_num_threads(), 1);

    for (size_t i = 0; i < columns.size(); ++i) {
        subsampling_pool.enqueue([&, i]() {
            sdsl::bit_vector* subvector;

            {
                std::lock_guard<std::mutex> lock(mu);

                // Initialize row indexes if empty
                if (row_indexes.empty()) {
                    num_rows = columns[i]->size();
                    row_indexes = annot::matrix::sample_row_indexes(num_rows, 1'000'000 /* sample_rows */);
                } else if (columns[i]->size() != num_rows) {
                    std::cerr << "Error: Column size mismatch for column " << i << std::endl;
                    exit(1);
                }

                subcolumn_ptrs.emplace_back(new sdsl::bit_vector());
                subvector = subcolumn_ptrs.back().get();
                column_ids.push_back(i);
            }

            // Subsample the column
            *subvector = sdsl::bit_vector(row_indexes.size(), false);
            for (size_t j = 0; j < row_indexes.size(); ++j) {
                if ((*columns[i])[row_indexes[j]]) {
                    (*subvector)[j] = true;
                }
            }
        });
    }

    subsampling_pool.join();

    // Rearrange the columns in their original order
    std::vector<sdsl::bit_vector> subcolumns(subcolumn_ptrs.size());
    for (size_t i = 0; i < column_ids.size(); ++i) {
        subcolumns.at(column_ids[i]) = std::move(*subcolumn_ptrs[i]);
    }

    // Compute the linkage matrix
    auto linkage_matrix = annot::matrix::agglomerative_greedy_linkage(std::move(subcolumns),
                                                                      get_num_threads());

    // Print the resulting linkage matrix
    for (int i = 0; i < linkage_matrix.rows(); ++i) {
        std::cout << "Linkage " << i << ": ";
        for (int j = 0; j < linkage_matrix.cols(); ++j) {
            std::cout << linkage_matrix(i, j) << " "; // Access element (i, j)
        }
        std::cout << std::endl;
    }

    // Serialize the linkage matrix
    std::ofstream out(linkage_file);
    out << linkage_matrix.format(CSVFormat) << std::endl;
}

bool load_columns_from_folder_callback(const std::string& folder_path, 
                              const std::function<void(uint64_t, const std::string&, std::unique_ptr<bit_vector>&&)>& callback,
                              size_t num_threads) {
    namespace fs = std::filesystem;

    // Find all files in the folder with the prefix "anno" and extension ".bin"
    std::vector<std::string> file_paths;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            const std::string file_name = entry.path().filename().string();
            if (file_name.rfind("column", 0) == 0 && file_name.substr(file_name.size() - 4) == ".bin") {
                file_paths.push_back(entry.path().string());
            }
        }
    }

    // Check if no matching files were found
    if (file_paths.empty()) {
        throw std::runtime_error("No files with prefix 'anno' and extension '.bin' found in the folder.");
        return false;
    }

    // Determine the column length from the size of the first file
    const std::string first_file = file_paths[0];
    std::ifstream first_file_stream(first_file, std::ios::binary | std::ios::ate); // Open in binary mode
    if (!first_file_stream.is_open()) {
        throw std::runtime_error("Failed to open file: " + first_file);
        return false;
    }
    std::streamsize file_size = first_file_stream.tellg(); // Get file size in bytes
    first_file_stream.close();

    uint64_t column_length = static_cast<uint64_t>(file_size) * 8; // Each byte is 8 bits

    // Process files in parallel using OpenMP (if enabled)
    bool success = true; // To track overall success
    std::mutex callback_mutex; // Mutex for thread-safe callback calls

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (size_t i = 0; i < file_paths.size(); ++i) {
        const auto& file_path = file_paths[i];

        try {
            // Open the file in binary mode
            std::ifstream file_stream(file_path, std::ios::binary);
            if (!file_stream.is_open()) {
                #pragma omp critical
                {
                    throw std::runtime_error("Failed to open file: " + file_path);
                    success = false;
                }
                continue;
            }

            // Read the binary data
            std::vector<uint8_t> buffer(file_size);
            file_stream.read(reinterpret_cast<char*>(buffer.data()), file_size);
            file_stream.close();

            // Create an `sdsl::bit_vector` for the column
            sdsl::bit_vector column(column_length, 0);
            for (size_t byte_idx = 0; byte_idx < buffer.size(); ++byte_idx) {
                for (int bit = 0; bit < 8; ++bit) {
                    if (buffer[byte_idx] & (1 << bit)) {
                        column[(byte_idx * 8) + bit] = 1;
                    }
                }
            }

            // Wrap the `sdsl::bit_vector` in a `std::unique_ptr` of `bit_vector_stat`
            auto column_ptr = std::make_unique<bit_vector_stat>(std::move(column));

            // Generate a label for the column based on its file name
            std::string label = fs::path(file_path).filename().string();

            // Pass the column to the callback
            {
                std::lock_guard<std::mutex> lock(callback_mutex);
                callback(i, label, std::move(column_ptr));
            }
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                throw std::runtime_error("Error processing file: " + file_path);
                success = false;
            }
        }
    }

    return success;
}

// Function to serialize std::vector<std::pair<uint64_t, std::string>> column_names
void serialize_column_names(const std::vector<std::pair<uint64_t, std::string>>& column_names, const std::string& filename) {
    std::ofstream out_file(filename, std::ios::binary); // Open file in binary mode
    if (!out_file) {
        throw std::runtime_error("Failed to open file for writing");
    }

    // Write the number of pairs
    size_t size = column_names.size();
    out_file.write(reinterpret_cast<const char*>(&size), sizeof(size)); // Write size of vector

    // Write each pair's first and second elements
    for (const auto& pair : column_names) {
        out_file.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first)); // Write first element
        size_t length = pair.second.size(); // Get the length of the string
        out_file.write(reinterpret_cast<const char*>(&length), sizeof(length)); // Write length
        out_file.write(pair.second.data(), length); // Write string content
    }

    out_file.close();
}

// Build BRWT
void build_BRWT(const std::string columns_folder_path, const std::vector<std::vector<uint64_t>> &linkage, const std::string &tmp_path, size_t num_nodes_parallel, size_t num_threads,
                const std::string &output_path) {
    std::unique_ptr<BRWT> binary_matrix;

    // auto get_columns = [&](const BRWTBottomUpBuilder::CallColumn &call_column) {
    //     bool success = ColumnCompressed<>::merge_load(
    //         annotation_files,
    //         [&](uint64_t j,
    //                 const std::string &label,
    //                 std::unique_ptr<bit_vector>&& column) {
    //             call_column(j, std::move(column));
    //             std::lock_guard<std::mutex> lock(mu);
    //             column_names.emplace_back(j, label);
    //         },
    //         num_threads
    //     );
    //     if (!success) {
    //         throw std::runtime_error("Failed to load columns from folder: " + columns_folder_path);
    //         exit(1);
    //     }
    // };

    std::mutex mu; // Mutex for thread-safe access
    std::vector<std::pair<uint64_t, std::string>> column_names; // Store column index and label
    auto get_columns = [&](const BRWTBottomUpBuilder::CallColumn& call_column) {

        // Use load_columns_from_folder to process columns
        bool success = load_columns_from_folder_callback(columns_folder_path, 
            [&](uint64_t j, const std::string& label, std::unique_ptr<bit_vector>&& column) {
                // j: column index, label: column label, column: column data
                // Pass the column to the BRWT builder
                call_column(j, std::move(column));

                // Store column names (thread-safe)
                std::lock_guard<std::mutex> lock(mu);
                column_names.emplace_back(j, label);
            }, 
            num_threads);

        if (!success) {
            throw std::runtime_error("Failed to load columns from folder: " + columns_folder_path);
            exit(1); // Exit if there's an error
        }
    };

    binary_matrix = std::make_unique<BRWT>(BRWTBottomUpBuilder::build(get_columns, linkage, tmp_path, num_nodes_parallel, num_threads));

    // Serialize the BRWT
    std::ofstream out(output_path + ".brwt", std::ios::binary);
    binary_matrix->serialize(out);

    // Serialize the column names
    serialize_column_names(column_names, output_path + ".columns");

}

// // Function to load a binary file into a vector of uint64_t
// std::vector<uint64_t> load_binary_2_vec_uint64_t(const std::string& filename) {
//     std::ifstream file(filename, std::ios::binary | std::ios::ate); // Open the file in binary mode, seek to end
//     if (!file) {
//         throw std::runtime_error("Failed to open file: " + filename);
//     }

//     // Get the size of the file
//     std::streamsize size = file.tellg();
//     file.seekg(0, std::ios::beg);

//     // Check that the file size is a multiple of uint64_t
//     if (size % sizeof(uint64_t) != 0) {
//         throw std::runtime_error("File size is not a multiple of uint64_t");
//     }

//     // Read the file contents into a vector
//     std::vector<uint64_t> data(size / sizeof(uint64_t));
//     if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
//         throw std::runtime_error("Failed to read file content");
//     }

//     return data;
// }

// // Function to find the index of a uint64_t value in a vector of uint64_t
// // Used to find the index of hash code in the annotation matrix row
// int find_uint64_t_value(const std::vector<uint64_t>& content, uint64_t value) {
//     // Find the value in the content vector
//     auto it = std::find(content.begin(), content.end(), value);

//     // If the value is found, return the index
//     if (it != content.end()) {
//         return std::distance(content.begin(), it);
//     }

//     // If not found, return -1
//     return -1;
// }



// // Function to load a binary file into a vector of std::pair<uint64_t, std::string>
// std::vector<std::pair<uint64_t, std::string>> load_binary_2_vec_pair_uint64_t_string(const std::string& filename) {
//     std::ifstream in_file(filename, std::ios::binary); // Open file in binary mode
//     if (!in_file) {
//         throw std::runtime_error("Failed to open file for reading");
//     }

//     // Read the number of pairs
//     size_t size;
//     in_file.read(reinterpret_cast<char*>(&size), sizeof(size));

//     std::vector<std::pair<uint64_t, std::string>> content(size);

//     // Read each pair's first and second elements
//     for (size_t i = 0; i < size; ++i) {
//         in_file.read(reinterpret_cast<char*>(&content[i].first), sizeof(content[i].first)); // Read first element

//         size_t length;
//         in_file.read(reinterpret_cast<char*>(&length), sizeof(length)); // Read length of string

//         std::string str(length, '\0'); // Allocate memory for the string
//         in_file.read(&str[0], length); // Read string content
//         content[i].second = std::move(str); // Store the pair in the vector
//     }

//     in_file.close();
//     return content;
// }

// // // Function to serialize a vector of strings into a binary file
// // void serialize_vec_string(const std::vector<std::string>& content, const std::string& filename) {
// //     std::ofstream out_file(filename, std::ios::binary); // Open file in binary mode
// //     if (!out_file) {
// //         throw std::runtime_error("Failed to open file for writing");
// //     }

// //     // Write the number of strings
// //     size_t size = content.size();
// //     out_file.write(reinterpret_cast<const char*>(&size), sizeof(size)); // Write size of vector

// //     // Write each string's length and content
// //     for (const auto& str : content) {
// //         size_t length = str.size(); // Get the length of the string
// //         out_file.write(reinterpret_cast<const char*>(&length), sizeof(length)); // Write length
// //         out_file.write(str.data(), length); // Write string content
// //     }

// //     out_file.close();
// // }

// // // Function to load a binary file into a vector of strings
// // std::vector<std::string> load_binary_2_vec_string(const std::string& filename) {
// //     std::ifstream in_file(filename, std::ios::binary); // Open file in binary mode
// //     if (!in_file) {
// //         throw std::runtime_error("Failed to open file for reading");
// //     }

// //     // Read the number of strings
// //     size_t size;
// //     in_file.read(reinterpret_cast<char*>(&size), sizeof(size));

// //     std::vector<std::string> content(size);

// //     // Read each string's length and content
// //     for (size_t i = 0; i < size; ++i) {
// //         size_t length;
// //         in_file.read(reinterpret_cast<char*>(&length), sizeof(length)); // Read length of string

// //         std::string str(length, '\0'); // Allocate memory for the string
// //         in_file.read(&str[0], length); // Read string content
// //         content[i] = std::move(str);  // Store the string in the vector
// //     }

// //     in_file.close();
// //     return content;
// // }

// // Function to access string item in a vector of string given the index
// // Used to find accession ID in the annotation matrix column given the index
// std::string get_string_item(const std::vector<std::string>& content, size_t index) {
//     // Check if the index is within bounds
//     if (index < content.size()) {
//         return content[index];
//     }

//     // Return an empty string if the index is out of bounds
//     return "";
// }


} // namespace

int main() {
    
    // // Read bit vectors from a binary file (/opt/metagraph/data/anno_SRR2125928.fastq_embedding_little.bin)
    // std::string file_path = "/opt/metagraph/data/anno_SRR2125928.fastq_embedding_little.bin";
    // std::vector<bool> one_column = load_annotation_bit_vector_little(file_path);
    // std::vector<std::unique_ptr<bit_vector>> columns = load_column_from_file(file_path, 1000);   

    // // Print std::vector<bool> one_column
    // for (size_t i = 0; i < one_column.size(); ++i) {
    //     std::cout << one_column[i];
    //     if ((i + 1) % 16 == 0) {
    //         std::cout << " "; // Add a space after every byte for readability
    //     }
    // }
    // std::cout << std::endl;
    
    // // Print std::vector<std::unique_ptr<bit_vector>> columns
    // for (size_t i = 0; i < columns.size(); ++i) {
    //     for (size_t j = 0; j < columns[i]->size(); ++j) {
    //         std::cout << (*columns[i])[j];
    //     }
    //     std::cout << std::endl;
    // }

    /* ============================================ */

    std::string folder_path = "/opt/metagraph/metagraph/src";

    // Load columns from the folder
    std::vector<std::unique_ptr<bit_vector>> columns = load_columns_from_folder(folder_path);

    // Output the number of loaded columns
    std::cout << "Loaded " << columns.size() << " columns from the folder.\n";

    // Print some sample data from the first column (if it exists)
    if (!columns.empty()) {
        std::cout << "Sample data from the first column:\n";
        for (size_t i = 0; i < std::min<size_t>(columns[0]->size(), 64); ++i) { // Limit to first 64 bits for display
            std::cout << (*columns[0])[i];
            if ((i + 1) % 8 == 0) {
                std::cout << " "; // Add spacing for readability
            }
        }
        std::cout << std::endl;
    }

    // Output the size of columns
    std::cout << "Size of columns: " << columns.size() << std::endl;
    // Output the size of the first column
    std::cout << "Size of the first column: " << columns[0]->size() << std::endl;

    // Precompute the linkage matrix
    std::string linkage_file = "clustering.linkage";
    BM_BRWTLinkageMatrix(columns, linkage_file);

    // Load the linkage matrix
    build_BRWT(folder_path, parse_linkage_matrix(linkage_file), "/opt/tmp", 1, 1, "row_diff");

    /* ============================================ */

    // // Run sparse compression benchmarks
    // BM_BRWTCompressSparse<1, 10>();
    // BM_BRWTCompressSparse<1, 100>();
    // BM_BRWTCompressSparse<1, 1000>();

    /* ============================================ */

    // // Run query benchmarks for different ranges
    // for (int i = 0; i <= 10; ++i) {
    //     BM_BRWTQueryRows<7, 16, 16, 2, true, 0>(i);
    // }

    /* ============================================ */

    // Run linkage matrix computation
    // BM_BRWTLinkageMatrix<>(); // <> is used to call the default template arguments
    // BM_SerializeLoadLinkage_BRWT<>();

    /* ============================================ */

    return 0;
}