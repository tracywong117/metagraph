#include <vector>
#include <iostream>

using namespace std;

int hammingDistance(const vector<uint8_t> &a, const vector<uint8_t> &b)
{
    int distance = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        distance += __builtin_popcount(a[i] ^ b[i]); // Built-in popcount for fast bit counting
    }
    return distance;
}

int main()
{
    // Test the hammingDistance function
    vector<uint8_t> a = {0b10101010, 0b01010101};
    vector<uint8_t> b = {0b11110000, 0b00001111};
    int distance = hammingDistance(a, b);
    std::cout << "Hamming distance: " << distance << std::endl;

    return 0;
}