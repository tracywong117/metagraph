#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

// Serialization function
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

// Deserialization function
std::vector<std::pair<uint64_t, std::string>> deserialize_column_names(const std::string& filename) {
    std::ifstream in_file(filename, std::ios::binary); // Open file in binary mode
    if (!in_file) {
        throw std::runtime_error("Failed to open file for reading");
    }

    // Read the number of pairs
    size_t size;
    in_file.read(reinterpret_cast<char*>(&size), sizeof(size));

    std::vector<std::pair<uint64_t, std::string>> column_names;
    column_names.reserve(size); // Reserve space for performance

    // Read each pair
    for (size_t i = 0; i < size; ++i) {
        uint64_t first;
        in_file.read(reinterpret_cast<char*>(&first), sizeof(first)); // Read first element

        size_t length;
        in_file.read(reinterpret_cast<char*>(&length), sizeof(length)); // Read string length

        std::string second(length, '\0'); // Allocate memory for the string
        in_file.read(&second[0], length); // Read string content

        column_names.emplace_back(first, std::move(second)); // Add pair to vector
    }

    in_file.close();
    return column_names;
}

int main() {
    try {
        // // Example vector of pairs
        // std::vector<std::pair<uint64_t, std::string>> column_names = {
        //     {1, "ColumnA"},
        //     {2, "ColumnB"},
        //     {3, "ColumnC"}
        // };

        // // Serialize the vector to a binary file
        std::string filename = "output.columns";
        // serialize_column_names(column_names, filename);



        // Deserialize the binary file back into a vector of pairs
        std::vector<std::pair<uint64_t, std::string>> loaded_column_names = deserialize_column_names(filename);

        // Print the deserialized content
        std::cout << "Deserialized content:" << std::endl;
        for (const auto& pair : loaded_column_names) {
            std::cout << "ID: " << pair.first << ", Name: " << pair.second << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}