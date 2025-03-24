#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>  // For uint64_t
#include <algorithm>  // For std::find

// Function to load a binary file into a vector of uint64_t
std::vector<uint64_t> load_binary_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate); // Open the file in binary mode, seek to end
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Get the size of the file
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check that the file size is a multiple of uint64_t
    if (size % sizeof(uint64_t) != 0) {
        throw std::runtime_error("File size is not a multiple of uint64_t");
    }

    // Read the file contents into a vector
    std::vector<uint64_t> data(size / sizeof(uint64_t));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Failed to read file content");
    }

    return data;
}

// Function to query the binary data for a specific uint64_t value
int query_binary(const std::vector<uint64_t>& content, uint64_t value) {
    // Find the value in the content vector
    auto it = std::find(content.begin(), content.end(), value);

    // If the value is found, return the index
    if (it != content.end()) {
        return std::distance(content.begin(), it);
    }

    // If not found, return -1
    return -1;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <filename> <query_value>" << std::endl;
        return 1;
    }

    try {
        // Load the binary file
        std::string filename = argv[1];
        std::vector<uint64_t> row_index = load_binary_file(filename);

        // Query the binary data for a specific value
        uint64_t query_value = std::stoull(argv[2]);
        int index = query_binary(row_index, query_value);

        // Print the result
        if (index != -1) {
            std::cout << "Value found at index: " << index << std::endl;
        } else {
            std::cout << "Value not found" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}