#include <iostream>

constexpr double Pi = 3.14159265358979323846;

struct Block {
    double pixels[8][8];
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
            if (u == 0) res /= std::sqrt(2);
            if (v == 0) res /= std::sqrt(2);

            result.pixels[u][v] = res;
        }
    }

    return result;
}

// Helper function for the inverse DCT to apply the cosine transform
double inverseSum(const Block& block, int x, int y) {
    double res = 0;
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            double a1 = ((2 * x + 1) * u * Pi) / 16;
            double a2 = ((2 * y + 1) * v * Pi) / 16;
            double coef = block.pixels[u][v] * cos(a1) * cos(a2);

            // Apply normalization factor
            if (u == 0) coef /= sqrt(2);
            if (v == 0) coef /= sqrt(2);

            res += coef;
        }
    }
    return res;
}

// IDCT function
Block IDCT(const Block& block) {
    Block result;

    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            result.pixels[x][y] = inverseSum(block, x, y) / 4.0;
        }
    }

    return result;
}

// Test function
void printBlock(const Block &block) {
    for (int i = 0; i < 8; i++) {
        std::cout << "[";
        for (int j = 0; j < 8; j++) {
            std::cout << block.pixels[i][j] << ", ";
        }
        std::cout << "],";
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    // Define a sample 8x8 signal block
  Block signal = {{{52, 55, 61, 66, 70, 61, 64, 73},
        {63, 59, 55, 90, 109, 85, 69, 72},
        {62, 59, 68, 113, 144, 104, 66, 73},
        {63, 58, 71, 122, 154, 106, 70, 69},
        {67, 61, 68, 104, 126, 88, 68, 70},
        {79, 65, 60, 70, 77, 68, 58, 75},
        {85, 71, 64, 59, 55, 61, 65, 83},
                   {87, 79, 69, 68, 65, 76, 78, 94}}};

    std::cout << "Original Signal Block:\n";
    printBlock(signal);

    // Perform DCT on the signal block
    Block dctBlock = DCT(signal);
    std::cout << "DCT Coefficients Block:\n";
    printBlock(dctBlock);

    // Perform IDCT on the DCT coefficients block
    Block reconstructedBlock = IDCT(dctBlock);
    std::cout << "Reconstructed Signal Block:\n";
    printBlock(reconstructedBlock);

    return 0;
}