#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <chrono> // For timing

class LargeDataSampler {
private:
    std::vector<int> data; // Stores the elements
    std::unordered_map<int, size_t> indexMap; // Maps element to its index in the vector
    std::mt19937 rng; // Random number generator

public:
    LargeDataSampler() : rng(std::random_device{}()) {}

    // Insert an element (O(1))
    void insert(int val) {
        if (indexMap.find(val) != indexMap.end()) return; // Element already exists
        indexMap[val] = data.size();
        data.push_back(val);
    }

    // Delete an element (O(1))
    void remove(int val) {
        auto it = indexMap.find(val);
        if (it == indexMap.end()) return; // Element not found

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
    int sample() {
        if (data.empty()) throw std::runtime_error("Set is empty!");
        std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
        return data[dist(rng)];
    }

    // Get the current size of the dataset
    size_t size() const {
        return data.size();
    }
};

int main() {
    const int numEntries = 60'000'000; // 60 million entries
    const int numSamples = 1000; // Number of random samples
    const int numDeletions = 1000; // Number of deletions

    // Step 1: Initialize the data structure
    LargeDataSampler sampler;

    // Step 2: Insert 60 million entries
    std::cout << "Inserting " << numEntries << " entries..." << std::endl;
    for (int i = 0; i < numEntries; ++i) {
        sampler.insert(i);
    }
    std::cout << "Insertion complete. Current size: " << sampler.size() << "." << std::endl;

    // Step 3: Measure time for random sampling
    std::cout << "Sampling " << numSamples << " random elements..." << std::endl;
    auto startSampling = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numSamples; ++i) {
        sampler.sample();
    }
    auto endSampling = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> samplingTime = endSampling - startSampling;
    std::cout << "Time taken for " << numSamples << " random samples: " << samplingTime.count() << " seconds." << std::endl;

    // Step 4: Measure time for deletions
    std::cout << "Deleting " << numDeletions << " elements..." << std::endl;
    auto startDeletion = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numDeletions; ++i) {
        sampler.remove(i); // Remove elements from 0 to numDeletions-1
    }
    auto endDeletion = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deletionTime = endDeletion - startDeletion;
    std::cout << "Time taken for " << numDeletions << " deletions: " << deletionTime.count() << " seconds." << std::endl;

    // Step 5: Measure time for random sampling after deletion
    std::cout << "Sampling " << numSamples << " random elements after deletion..." << std::endl;
    startSampling = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numSamples; ++i) {
        sampler.sample();
    }
    endSampling = std::chrono::high_resolution_clock::now();
    samplingTime = endSampling - startSampling;
    std::cout << "Time taken for " << numSamples << " random samples after deletion: " << samplingTime.count() << " seconds." << std::endl;

    return 0;
}