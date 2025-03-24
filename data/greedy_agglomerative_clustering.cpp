#include <climits> // For INT_MAX and other integer limits
#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <bitset>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>
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

// Efficient random sampling without shuffling the entire vector
std::vector<int> sampleWithoutReplacement(const std::vector<int> &vec, int sampleSize, std::mt19937 &rng)
{
    std::unordered_set<int> sampledIndices; // To ensure uniqueness
    std::vector<int> samples;
    samples.reserve(sampleSize);

    std::uniform_int_distribution<int> dist(0, vec.size() - 1);

    while (samples.size() < sampleSize)
    {
        int index = dist(rng);
        if (sampledIndices.find(index) == sampledIndices.end())
        {
            sampledIndices.insert(index);
            samples.push_back(vec[index]);
        }
    }

    return samples;
}

// Greedy Agglomerative Clustering using Hamming Distance
void greedyAgglomerativeClusteringHamming(
    const std::vector<std::vector<uint8_t>> &data,
    int numClusters,
    int samplingSize)
{
    size_t nPoints = data.size();
    size_t dim = data[0].size();
    std::vector<int> pointsAssignCluster(nPoints);                        // Cluster assignments
    std::iota(pointsAssignCluster.begin(), pointsAssignCluster.end(), 0); // Initially, each point is its own cluster

    std::vector<std::vector<uint8_t>> representatives = data;       // Each cluster initially represented by its own point
    std::unordered_map<int, std::vector<int>> clustersPoints;       // Cluster points map
    std::vector<int> existingClusters(nPoints);                     // Array of existing clusters
    std::iota(existingClusters.begin(), existingClusters.end(), 0); // Initialize with cluster IDs [0, 1, ..., nPoints-1]

    size_t numberLoop = nPoints - numClusters;

    std::random_device rd;
    std::mt19937 rng(rd());

    ProgressBar bar(50, "Clustering Progress"); // Initialize progress bar

    // Timing variables
    // auto startTime = chrono::steady_clock::now();  // Start time for the entire algorithm
    // chrono::steady_clock::time_point stepStartTime, stepEndTime;

    // chrono::duration<double> timeSampling(0);
    // chrono::duration<double> timeHamming(0);
    // chrono::duration<double> timeMerging(0);

    // Initialize clustersPoints
    for (size_t i = 0; i < nPoints; ++i)
    {
        clustersPoints[i] = {static_cast<int>(i)}; // Each cluster starts with its own point
    }

    for (size_t iter = 0; iter < numberLoop; ++iter)
    {
        bar.update(iter, numberLoop); // Update progress bar

        // Step 1: Randomly sample cluster pairs
        std::vector<int> sampled1 = sampleWithoutReplacement(existingClusters, samplingSize, rng);
        std::vector<int> sampled2 = sampleWithoutReplacement(existingClusters, samplingSize, rng);

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
        auto it = std::find(existingClusters.begin(), existingClusters.end(), c2);
        if (it != existingClusters.end())
        {
            existingClusters.erase(it);
        }
    }

    // Final calculations and reporting
    bar.finish();
    std::cout << "\nFinal number of clusters: " << existingClusters.size() << std::endl;

    double avgClusterSize = 0;
    for (int cluster : existingClusters)
    {
        avgClusterSize += clustersPoints[cluster].size();
        std::cout << "Cluster " << cluster << " size: " << clustersPoints[cluster].size() << std::endl;
    }
    avgClusterSize /= existingClusters.size();
    std::cout << "Average cluster size: " << avgClusterSize << std::endl;
    // Total time for the algorithm
    // auto endTime = chrono::steady_clock::now();
    // auto totalTime = chrono::duration_cast<chrono::seconds>(endTime - startTime).count();
}

// Main function
int main()
{
    // Example Usage
    size_t nPoints = 60'000'000;    // Number of points
    size_t bitDim = 32;            // Number of bits per vector
    size_t packedDim = bitDim / 8; // Packed dimension (4 bytes for 32 bits)

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

    int numClusters = 50;  // Desired number of clusters
    int samplingSize = 10; // Number of random cluster pairs to sample

    // Perform clustering
    greedyAgglomerativeClusteringHamming(data, numClusters, samplingSize);

    return 0;
}