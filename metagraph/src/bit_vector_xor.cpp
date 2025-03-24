#include <iostream>
#include <vector>
#include <algorithm> // For std::find
#include <chrono>    // For benchmarking
#include <random>    // For generating random test data

// Class to store indices of set bits
class BitIndexStore {
private:
    std::vector<size_t> set_bit_indices; // Indices of set bits
    size_t bit_vector_size = 0;          // Total size of the bit vector

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

    size_t size() const {
        return bit_vector_size;
    }

    bool is_bit_set(size_t index) const {
        return std::find(set_bit_indices.begin(), set_bit_indices.end(), index) != set_bit_indices.end();
    }
};

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

// Perform XOR operation directly on two std::vector<bool>
std::vector<bool> simple_vector_xor(const std::vector<bool>& v1, const std::vector<bool>& v2) {
    size_t size = std::max(v1.size(), v2.size());
    std::vector<bool> result(size, false);

    for (size_t i = 0; i < size; ++i) {
        bool bit1 = i < v1.size() ? v1[i] : false;
        bool bit2 = i < v2.size() ? v2[i] : false;
        result[i] = bit1 ^ bit2;
    }

    return result;
}

// Generate random bit vector with density control
std::vector<bool> generate_random_bit_vector(size_t num_bits, double density) {
    std::vector<bool> bit_vector(num_bits, false);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < num_bits; ++i) {
        if (dis(gen) < density) {
            bit_vector[i] = true;
        }
    }

    return bit_vector;
}

int main() {
    // User-defined parameters
    size_t num_bits = 100000; // Number of bits
    double density = 0.1;     // Density of 1s (e.g., 0.1 = 10% bits set to 1)

    // Ask user for input
    std::cout << "Enter the number of bits: ";
    std::cin >> num_bits;

    std::cout << "Enter the density of 1s (e.g., 0.1 = 10%): ";
    std::cin >> density;

    // Generate random bit vectors
    std::vector<bool> bit_vector1 = generate_random_bit_vector(num_bits, density);
    std::vector<bool> bit_vector2 = generate_random_bit_vector(num_bits, density);

    // Benchmark BitIndexStore XOR
    BitIndexStore bv1(bit_vector1);
    BitIndexStore bv2(bit_vector2);
    auto start1 = std::chrono::high_resolution_clock::now();
    BitIndexStore xor_result = compute_xor(bv1, bv2);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    std::cout << "BitIndexStore XOR time: " << elapsed1.count() * 1000 << " ms\n";

    // Benchmark std::vector<bool> XOR
    auto start2 = std::chrono::high_resolution_clock::now();
    std::vector<bool> vector_xor_result = simple_vector_xor(bit_vector1, bit_vector2);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end2 - start2;
    std::cout << "std::vector<bool> XOR time: " << elapsed2.count() * 1000 << " ms\n";

    // Storage usage by <vector<bool>>
    size_t vector_size = bit_vector1.size() * sizeof(bool); // Unit is byte
    std::cout << "Memory usage by std::vector<bool>: " << vector_size << " bytes\n";

    // Verify results are the same
    bool results_match = true;
    const auto& xor_indices = xor_result.get_set_bit_indices();
    for (size_t i = 0; i < num_bits; ++i) {
        bool bit1 = std::find(xor_indices.begin(), xor_indices.end(), i) != xor_indices.end();
        if (bit1 != vector_xor_result[i]) {
            results_match = false;
            break;
        }
    }

    if (results_match) {
        std::cout << "Results match between BitIndexStore and std::vector<bool> XOR.\n";
    } else {
        std::cout << "Results do NOT match between BitIndexStore and std::vector<bool> XOR.\n";
    }

    return 0;
}