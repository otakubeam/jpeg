#include <iostream>
#include <array>
#include <vector>
#include <utility>


constexpr double Pi = 3.14159265358979323846;

struct Block {
    double pixels[8][8];

  class ZigZagIterator {
  public:
    ZigZagIterator(Block& blk__, std::size_t index__ = 0): blk{blk__}, index(index__) {}

    typedef double      value_type;
    typedef value_type& reference;
    typedef value_type* pointer;

    // Dereference Operators
    pointer operator->() const { 
        auto [row, col] = zigzagOrder[index];
        return &blk.pixels[row][col]; 
    }
    reference operator*() const { return *operator->(); }

    // Pre-increment
    ZigZagIterator& operator++() { ++index; return *this; }

    bool operator==(ZigZagIterator right) const { return &right.blk == &blk && right.index == index; }
    bool operator!=(ZigZagIterator right) const { return !operator==(right); }

    reference operator[](std::size_t idx) {
        auto [row, col] = zigzagOrder[idx];
        return blk.pixels[row][col];
    }

  private:
    Block& blk;
    std::size_t index;

    static constexpr std::array<std::pair<int, int>, 64> zigzagOrder = {
        {{0, 0}, {0, 1}, {1, 0}, {2, 0}, {1, 1}, {0, 2}, {0, 3}, {1, 2},
         {2, 1}, {3, 0}, {4, 0}, {3, 1}, {2, 2}, {1, 3}, {0, 4}, {0, 5},
         {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0}, {6, 0}, {5, 1}, {4, 2},
         {3, 3}, {2, 4}, {1, 5}, {0, 6}, {0, 7}, {1, 6}, {2, 5}, {3, 4},
         {4, 3}, {5, 2}, {6, 1}, {7, 0}, {7, 1}, {6, 2}, {5, 3}, {4, 4},
         {3, 5}, {2, 6}, {1, 7}, {2, 7}, {3, 6}, {4, 5}, {5, 4}, {6, 3},
         {7, 2}, {7, 3}, {6, 4}, {5, 5}, {4, 6}, {3, 7}, {4, 7}, {5, 6},
         {6, 5}, {7, 4}, {7, 5}, {6, 6}, {5, 7}, {6, 7}, {7, 6}, {7, 7}}};
  };

  
    ZigZagIterator begin() { return ZigZagIterator(*this); }
    ZigZagIterator end() { return ZigZagIterator(*this, 64); }

};


using DC = int;
using AC = int;
using RunLengthResultAC = std::vector<std::pair<int, AC>>;

auto performRle(Block blk) -> RunLengthResultAC {
    std::size_t zero_count = 0;
    RunLengthResultAC result;

    for (auto elem_double : blk) {
        if (auto elem = elem_double; elem == 0) {
            zero_count++; 
        } else {
            result.emplace_back(zero_count, elem);
        }
    }

    if (zero_count != 0) {
        result.emplace_back(0, 0);
    }

    return result;
}

class BitWriter {
public:
    using StorageType = std::size_t;

    BitWriter() : store(0), bits_written(0), total_bits_written(0) {}

    void write(int val, size_t bitlen) {
        while (bitlen > 0) {
            bitlen--;
            store <<= 1;
            store |= (val >> bitlen) & 1;
            bits_written++;
            
            if (bits_written == sizeof(StorageType) * 8) {  // Use bits instead of bytes
                buffer.push_back(store);
                store = 0;
                bits_written = 0;
                total_bits_written += sizeof(StorageType) * 8;
            }
        }
    }

    void flush() {
        if (bits_written == 0) {
            return;
        }
        auto bits_left = (sizeof(StorageType) * 8) - bits_written;
        store <<= bits_left;
        buffer.push_back(store);
        total_bits_written += bits_written;
        store = 0;
        bits_written = 0;
    }

    const std::vector<StorageType>& getBuffer() const {
        return buffer;
    }

    size_t getTotalBitsWritten() const {
        return total_bits_written;
    }

private:

    std::vector<StorageType> buffer;
    StorageType store;
    size_t bits_written;
    size_t total_bits_written;
};


constexpr Block luminanceQuantTable = {{
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
}};

constexpr Block chrominanceQuantTable = {{
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}
}};

static void quantize(Block& block, const Block& quantTable) {
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            auto& elem = block.pixels[x][y];
            const auto& quant = quantTable.pixels[x][y];
            elem = int(elem / quant);
        }
    }
}

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


    quantize(dctBlock, luminanceQuantTable);
    // quantize(dctBlock, chrominanceQuantTable);
    // quantize(dctBlock, chrominanceQuantTable);

    // Perform IDCT on the DCT coefficients block
    Block reconstructedBlock = IDCT(dctBlock);
    std::cout << "Reconstructed Signal Block:\n";
    printBlock(reconstructedBlock);

    return 0;
}