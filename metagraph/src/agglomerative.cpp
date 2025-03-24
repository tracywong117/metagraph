#include <climits> // For INT_MAX and other integer limits
#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <numeric>
#include <chrono>
#include "ProgressBar.h"

using namespace std;

// Function to compute the Hamming distance between two uint8_t arrays
inline int hammingDistance(const vector<uint8_t> &a, const vector<uint8_t> &b)
{
    int distance = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        distance += __builtin_popcount(a[i] ^ b[i]); // Built-in popcount for fast bit counting
    }
    return distance;
}

// Class for efficient sampling, insertion, and deletion
class LargeDataSampler
{
private:
    std::vector<int> data;               // Stores the elements
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

// Greedy Agglomerative Clustering using Hamming Distance
void greedyAgglomerativeClusteringHamming(
    const std::vector<std::vector<uint8_t>> &data,
    int numClusters,
    int samplingSize)
{
    size_t nPoints = data.size();
    size_t numberLoop = nPoints - numClusters;
    std::vector<int> pointsAssignCluster(nPoints);                        // Cluster assignments
    std::iota(pointsAssignCluster.begin(), pointsAssignCluster.end(), 0); // Initially, each point is its own cluster

    std::vector<std::vector<uint8_t>> representatives = data;       // Each cluster initially represented by its own point
    std::unordered_map<int, std::vector<int>> clustersPoints;       // Cluster points map
    LargeDataSampler existingClusters;                              // Replace the vector with LargeDataSampler

    // Initialize clustersPoints and existingClusters
    for (size_t i = 0; i < nPoints; ++i)
    {
        clustersPoints[i] = {static_cast<int>(i)}; // Each cluster starts with its own point
        existingClusters.insert(i);               // Add each cluster ID to the sampler
    }

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 rng(rd());

    cout << "Starting Greedy Agglomerative Clustering..." << endl;

    ProgressBar bar(50, "Clustering Progress", numberLoop); // Initialize progress bar

    for (size_t iter = 0; iter < numberLoop; ++iter)
    {
        bar.update(); // Update progress bar

        // Step 1: Randomly sample cluster pairs
        std::vector<int> sampled1;
        std::vector<int> sampled2;
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

            int dist = hammingDistance(representatives[c1], representatives[c2]);
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

        // Update the representative for c1 (e.g., choose new representative or recompute)
        representatives[c1] = representatives[c2];

        // Remove c2 from active clusters
        existingClusters.remove(c2);
    }

    // Final calculations and reporting
    bar.finish();
    std::cout << "\nFinal number of clusters: " << existingClusters.size() << std::endl;

    // use existingClusters to get the final clusters
    // use pointsAssignCluster to get the final cluster assignments
    // print the final clusters and their points
    double avgClusterSize = 0;
    for (size_t i = 0; i < existingClusters.size(); ++i)
    {
        avgClusterSize += clustersPoints[existingClusters.get(i)].size();
        std::cout << "Cluster " << i  << " (" << existingClusters.get(i) << ") " << " size: " << clustersPoints[existingClusters.get(i)].size() << std::endl;
    }
    avgClusterSize /= numClusters;
    std::cout << "Average cluster size: " << avgClusterSize << std::endl;
    
}

// Main function
int main()
{
    // Example Usage
    size_t nPoints = 60'000'000;    // Number of points
    size_t bitDim = 32;           // Number of bits per vector
    size_t packedDim = bitDim / 8; // Packed dimension (4 bytes for 32 bits)
    int numClusters = 50'000;  // Desired number of clusters
    int samplingSize = 1000; // Number of random cluster pairs to sample

    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<uint8_t> dist(0, 255);

    // Generate random binary data
    vector<vector<uint8_t>> data(nPoints, vector<uint8_t>(packedDim));
    for (size_t i = 0; i < nPoints; ++i)
    {
        for (size_t j = 0; j < packedDim; ++j)
        {
            data[i][j] = dist(rng);
        }
    }
    cout << "Generated random binary data with " << nPoints << " points and " << bitDim << " bits each." << endl;

    // Perform clustering
    greedyAgglomerativeClusteringHamming(data, numClusters, samplingSize);

    return 0;
}