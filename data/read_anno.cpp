#include <iostream>
#include <fstream>
#include <vector>
#include <bitset>

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

int main() {
    try {
        // Specify the path to the binary file
        std::string file_path = "anno_SRR2125928.fastq_embedding_little.bin";

        // Load and unpack the bit vector
        std::vector<bool> bit_vector = load_annotation_bit_vector_little(file_path);

        // Print the unpacked bits
        std::cout << "Unpacked bits in little-endian order:\n";
        for (size_t i = 0; i < bit_vector.size(); ++i) {
            std::cout << bit_vector[i];
            if ((i + 1) % 16 == 0) {
                std::cout << " "; // Add a space after every byte for readability
                break; // Break after printing the first byte
            }
        }
        // Print the number of bits in the bit vector
        std::cout << "\nNumber of bits: " << bit_vector.size() << std::endl;
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}