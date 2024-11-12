#include <cmath>
#include <array>

constexpr double Pi = 3.14159265358979323846;

struct Block {
    double[8][8] pixels;
};

// Helper function to compute the sum for DCT
static double sum(const Block& block, int u, int v) {
    double res = 0;
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            double a1 = ((2 * x + 1) * u * Pi) / 16;
            double a2 = ((2 * y + 1) * v * Pi) / 16;
            res += block.pixels[x][y] * cos(a1) * cos(a2);
        }
    }
    return res;
}

// Perform the 8x8 DCT on the input block
Block DCT(const Block& block) {
    Block result;

    // Calculate the DCT for each frequency coefficient (k, m)
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            double res = sum(block, u, v) / 4.0;

            // Apply the normalization factor
            if (u == 0) res /= sqrt(2);
            if (v == 0) res /= sqrt(2);

            result.pixels[u][v] = res;
        }
    }

    return result;
}

Block IDCT(const Block& block) {
    
}