#include <iostream>
#include <vector>
#include <chrono>

int main() {
    // Create a vector with 600 million elements
    size_t n = 600000000;
    std::vector<int> vec(n);
    for (size_t i = 0; i < n; ++i) {
        vec[i] = i;
    }

    // Delete an element at a known index
    size_t index = n / 2;  // Middle index
    auto start = std::chrono::high_resolution_clock::now();
    vec.erase(vec.begin() + index);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken to delete element in C++: " << elapsed.count() << " seconds\n";

    return 0;
}