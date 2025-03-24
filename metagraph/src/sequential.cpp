#include <climits> // For INT_MAX and other integer limits
#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <numeric>
#include <chrono>
#include <algorithm>
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

// Greedy Sequential Clustering using Hamming Distance
void greedySequentialClusteringHamming(
    const std::vector<std::vector<uint8_t>> &data,
    int threshold)
{
    size_t num_points = data.size();

    // Initialize the progress bar
    ProgressBar progressBar(50, "Clustering Progress", num_points);

    std::vector<size_t> anchors; // Anchor indices
    std::vector<std::vector<size_t>> clusters; // Clusters of indices

    for (size_t i = 0; i < num_points; ++i) {
        bool assigned_to_cluster = false;

        // Check if the point can be assigned to an existing cluster
        for (size_t j = 0; j < anchors.size(); ++j) {
            size_t anchor_idx = anchors[j];
            if (hammingDistance(data[i], data[anchor_idx]) < threshold) {
                clusters[j].push_back(i);
                assigned_to_cluster = true;
                break;
            }
        }

        // If not assigned to any cluster, create a new one
        if (!assigned_to_cluster) {
            anchors.push_back(i);
            clusters.push_back({ i });
        }

        // Update the progress bar
        progressBar.update();
    }

    // Finish the progress bar
    progressBar.finish();

    // Print the number of clusters
    std::cout << "Found " << anchors.size() << " clusters." << std::endl;

}

// Main function
int main()
{
    // Example Usage
    size_t nPoints = 60'000'000;    // Number of points
    size_t bitDim = 32;           // Number of bits per vector
    size_t packedDim = bitDim / 8; // Packed dimension (4 bytes for 32 bits)
    int threshold = 12; // Maximum Hamming distance for clustering

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

    // Sort the data for better clustering performance
    sort(data.begin(), data.end()); // ascending order
    cout << "Sorted the data for better clustering performance." << endl;

    // Perform clustering
    greedySequentialClusteringHamming(data, threshold);

    return 0;
}