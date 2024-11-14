#include <cstdint>
#include <map>
#include <array>
#include <numeric>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <utility>
#include <fstream>
#include <algorithm>

constexpr double Pi = 3.14159265358979323846;

struct Block {
    double pixels[8][8];

    // Forward declaration of both iterator types
    template<bool IsConst>
    class ZigZagIteratorImpl;

    // Type aliases for convenience
    using iterator = ZigZagIteratorImpl<false>;
    using const_iterator = ZigZagIteratorImpl<true>;

    template<bool IsConst>
    class ZigZagIteratorImpl {
    public:
        using BlockType = typename std::conditional<IsConst, const Block, Block>::type;
        using value_type = double;
        using reference = typename std::conditional<IsConst, const value_type&, value_type&>::type;
        using pointer = typename std::conditional<IsConst, const value_type*, value_type*>::type;

        ZigZagIteratorImpl(BlockType& blk__, std::size_t index__ = 0)
            : blk{blk__}, index(index__) {}

        // Allow conversion from non-const to const iterator
        template<bool WasConst>
        ZigZagIteratorImpl(const ZigZagIteratorImpl<WasConst>& other,
            typename std::enable_if<IsConst || !WasConst>::type* = nullptr)
            : blk(other.blk), index(other.index) {}

        // Dereference Operators
        pointer operator->() const {
            auto [row, col] = zigzagOrder[index];
            return &blk.pixels[row][col];
        }
        reference operator*() const { return *operator->(); }

        // Pre-increment
        ZigZagIteratorImpl& operator++() { ++index; return *this; }

        // Post-increment
        ZigZagIteratorImpl operator++(int) {
            ZigZagIteratorImpl tmp(*this);
            ++index;
            return tmp;
        }

        template<bool OtherConst>
        bool operator==(const ZigZagIteratorImpl<OtherConst>& right) const {
            return &right.blk == &blk && right.index == index;
        }

        template<bool OtherConst>
        bool operator!=(const ZigZagIteratorImpl<OtherConst>& right) const {
            return !operator==(right);
        }

        reference operator[](std::size_t idx) const {
            auto [row, col] = zigzagOrder[idx];
            return blk.pixels[row][col];
        }

    private:
        BlockType& blk;
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

        // Make other instantiations friends
        template<bool> friend class ZigZagIteratorImpl;
    };

    // Non-const iteration
    iterator begin() { return iterator(*this); }
    iterator end() { return iterator(*this, 64); }

    // Const iteration
    const_iterator begin() const { return const_iterator(*this); }
    const_iterator end() const { return const_iterator(*this, 64); }

    // Const iteration through cbegin/cend
    const_iterator cbegin() const { return const_iterator(*this); }
    const_iterator cend() const { return const_iterator(*this, 64); }
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

class BitReader {
public:
    using StorageType = std::size_t;

    // Constructor takes a buffer to read from
    BitReader(const std::vector<StorageType>& buffer)
        : buffer(buffer), store(0), bits_read(0), total_bits_read(0), buffer_index(0) {
        if (!buffer.empty()) {
            store = buffer[0];
        }
    }

    // Read 'bitlen' bits and return as an integer
    int read(size_t bitlen) {
        int result = 0;
        while (bitlen > 0) {
            if (bits_read == sizeof(StorageType) * 8) {
                // Move to the next storage element in the buffer
                buffer_index++;
                if (buffer_index >= buffer.size()) {
                    throw std::out_of_range("No more bits to read");
                }
                store = buffer[buffer_index];
                bits_read = 0;
            }

            // Calculate the number of bits we can read in this iteration
            size_t bits_to_read = std::min(bitlen, sizeof(StorageType) * 8 - bits_read);

            // Shift result and append the new bits
            result <<= bits_to_read;
            result |= (store >> (sizeof(StorageType) * 8 - bits_read - bits_to_read)) & ((1 << bits_to_read) - 1);

            // Update counters
            bits_read += bits_to_read;
            total_bits_read += bits_to_read;
            bitlen -= bits_to_read;
        }

        return result;
    }

    // Get total number of bits read
    size_t getTotalBitsRead() const {
        return total_bits_read;
    }

private:
    const std::vector<StorageType>& buffer;
    StorageType store;
    size_t bits_read;
    size_t total_bits_read;
    size_t buffer_index;
};

using HuffmanCode = std::pair<size_t, size_t>;
using HuffmanTable = std::map<std::pair<int, int>, HuffmanCode>;

// Initialize the unordered_map with HuffmanCode structs
HuffmanTable acLuminanceHuffmanTable = {
    // Format: {{run, size}, {code, bit_length}}
    // {0, 0} is End of Block (EOB)
    {{0, 0}, {0b1010, 4}},
    
    // Run = 0
    {{0, 1}, {0b00, 2}},          {{0, 2}, {0b01, 2}},          
    {{0, 3}, {0b100, 3}},         {{0, 4}, {0b1011, 4}},
    {{0, 5}, {0b11010, 5}},       {{0, 6}, {0b1111000, 7}},     
    {{0, 7}, {0b11111000, 8}},    {{0, 8}, {0b1111110110, 10}},
    {{0, 9}, {0b1111111110000010, 16}}, {{0, 10}, {0b1111111110000011, 16}},
    
    // Run = 1
    {{1, 1}, {0b1100, 4}},        {{1, 2}, {0b11011, 5}},       
    {{1, 3}, {0b1111001, 7}},     {{1, 4}, {0b111110110, 9}},
    {{1, 5}, {0b1111110111, 10}},  {{1, 6}, {0b111111110100, 12}},
    {{1, 7}, {0b1111111110000100, 16}}, 
    {{1, 8}, {0b1111111110000101, 16}}, 
    {{1, 9}, {0b1111111110000110, 16}}, 
    {{1, 10}, {0b1111111110000111, 16}},
    
    // Run = 2
    {{2, 1}, {0b11100, 5}},       {{2, 2}, {0b11111001, 8}},    
    {{2, 3}, {0b1111111000, 10}},  {{2, 4}, {0b111111110101, 12}},
    {{2, 5}, {0b1111111110001000, 16}}, 
    {{2, 6}, {0b1111111110001001, 16}}, 
    {{2, 7}, {0b1111111110001010, 16}},
    {{2, 8}, {0b1111111110001011, 16}}, 
    {{2, 9}, {0b1111111110001100, 16}}, 
    {{2, 10}, {0b1111111110001101, 16}},
    
    // Run = 3
    {{3, 1}, {0b111010, 6}},      {{3, 2}, {0b111110111, 9}},   
    {{3, 3}, {0b111111110110, 12}},
    {{3, 4}, {0b1111111110001110, 16}},
    {{3, 5}, {0b1111111110001111, 16}}, 
    {{3, 6}, {0b1111111110010000, 16}}, 
    {{3, 7}, {0b1111111110010001, 16}},
    {{3, 8}, {0b1111111110010010, 16}}, 
    {{3, 9}, {0b1111111110010011, 16}}, 
    {{3, 10}, {0b1111111110010100, 16}},
    
    // Run = 4
    {{4, 1}, {0b111011, 6}},      {{4, 2}, {0b1111111001, 10}},  
    {{4, 3}, {0b111111110111, 12}},
    {{4, 4}, {0b1111111110010101, 16}},
    {{4, 5}, {0b1111111110010110, 16}}, 
    {{4, 6}, {0b1111111110010111, 16}}, 
    {{4, 7}, {0b1111111110011000, 16}},
    {{4, 8}, {0b1111111110011001, 16}}, 
    {{4, 9}, {0b1111111110011010, 16}}, 
    {{4, 10}, {0b1111111110011011, 16}},
    
    // Run = 5
    {{5, 1}, {0b1111010, 7}},     {{5, 2}, {0b1111111010, 10}},  
    {{5, 3}, {0b1111111110011100, 16}},
    {{5, 4}, {0b1111111110011101, 16}},
    {{5, 5}, {0b1111111110011110, 16}}, 
    {{5, 6}, {0b1111111110011111, 16}}, 
    {{5, 7}, {0b1111111110100000, 16}},
    {{5, 8}, {0b1111111110100001, 16}}, 
    {{5, 9}, {0b1111111110100010, 16}}, 
    {{5, 10}, {0b1111111110100011, 16}},
    
    // Run = 6
    {{6, 1}, {0b1111011, 7}},     {{6, 2}, {0b1111111011, 10}},  
    {{6, 3}, {0b1111111110100100, 16}},
    {{6, 4}, {0b1111111110100101, 16}},
    {{6, 5}, {0b1111111110100110, 16}}, 
    {{6, 6}, {0b1111111110100111, 16}}, 
    {{6, 7}, {0b1111111110101000, 16}},
    {{6, 8}, {0b1111111110101001, 16}}, 
    {{6, 9}, {0b1111111110101010, 16}}, 
    {{6, 10}, {0b1111111110101011, 16}},
    
    // Run = 7
    {{7, 1}, {0b11111010, 8}},    {{7, 2}, {0b11111111000, 11}}, 
    {{7, 3}, {0b1111111110101100, 16}},
    {{7, 4}, {0b1111111110101101, 16}},
    {{7, 5}, {0b1111111110101110, 16}}, 
    {{7, 6}, {0b1111111110101111, 16}}, 
    {{7, 7}, {0b1111111110110000, 16}},
    {{7, 8}, {0b1111111110110001, 16}}, 
    {{7, 9}, {0b1111111110110010, 16}}, 
    {{7, 10}, {0b1111111110110011, 16}},
    
    // Run = 8
    {{8, 1}, {0b11111011, 8}},    {{8, 2}, {0b1111111110110100, 16}}, 
    {{8, 3}, {0b1111111110110101, 16}},
    {{8, 4}, {0b1111111110110110, 16}},
    {{8, 5}, {0b1111111110110111, 16}}, 
    {{8, 6}, {0b1111111110111000, 16}}, 
    {{8, 7}, {0b1111111110111001, 16}},
    {{8, 8}, {0b1111111110111010, 16}}, 
    {{8, 9}, {0b1111111110111011, 16}}, 
    {{8, 10}, {0b1111111110111100, 16}},
    
    // Run = 9
    {{9, 1}, {0b111111000, 9}},   {{9, 2}, {0b1111111110111101, 16}}, 
    {{9, 3}, {0b1111111110111110, 16}},
    {{9, 4}, {0b1111111110111111, 16}},
    {{9, 5}, {0b1111111111000000, 16}}, 
    {{9, 6}, {0b1111111111000001, 16}}, 
    {{9, 7}, {0b1111111111000010, 16}},
    {{9, 8}, {0b1111111111000011, 16}}, 
    {{9, 9}, {0b1111111111000100, 16}}, 
    {{9, 10}, {0b1111111111000101, 16}},
    
    // Run = 10
    {{10, 1}, {0b111111001, 9}},  {{10, 2}, {0b1111111111000110, 16}}, 
    {{10, 3}, {0b1111111111000111, 16}},
    {{10, 4}, {0b1111111111001000, 16}},
    {{10, 5}, {0b1111111111001001, 16}}, 
    {{10, 6}, {0b1111111111001010, 16}}, 
    {{10, 7}, {0b1111111111001011, 16}},
    {{10, 8}, {0b1111111111001100, 16}}, 
    {{10, 9}, {0b1111111111001101, 16}}, 
    {{10, 10}, {0b1111111111001110, 16}},
    
    // Run = 11
    {{11, 1}, {0b111111010, 9}},  {{11, 2}, {0b1111111111001111, 16}}, 
    {{11, 3}, {0b1111111111010000, 16}},
    {{11, 4}, {0b1111111111010001, 16}},
    {{11, 5}, {0b1111111111010010, 16}}, 
    {{11, 6}, {0b1111111111010011, 16}}, 
    {{11, 7}, {0b1111111111010100, 16}},
    {{11, 8}, {0b1111111111010101, 16}}, 
    {{11, 9}, {0b1111111111010110, 16}}, 
    {{11, 10}, {0b1111111111010111, 16}},
    
    // Run = 12
    {{12, 1}, {0b11111111001, 11}}, 
    {{12, 2}, {0b1111111111011000, 16}}, 
    {{12, 3}, {0b1111111111011001, 16}},
    {{12, 4}, {0b1111111111011010, 16}},
    {{12, 5}, {0b1111111111011011, 16}}, 
    {{12, 6}, {0b1111111111011100, 16}}, 
    {{12, 7}, {0b1111111111011101, 16}},
    {{12, 8}, {0b1111111111011110, 16}}, 
    {{12, 9}, {0b1111111111011111, 16}}, 
    {{12, 10}, {0b1111111111100000, 16}},
    
    // Run = 13
    {{13, 1}, {0b11111111100, 11}}, 
    {{13, 2}, {0b1111111111100001, 16}}, 
    {{13, 3}, {0b1111111111100010, 16}},
    {{13, 4}, {0b1111111111100011, 16}},
    {{13, 5}, {0b1111111111100100, 16}}, 
    {{13, 6}, {0b1111111111100101, 16}}, 
    {{13, 7}, {0b1111111111100110, 16}},
    {{13, 8}, {0b1111111111100111, 16}}, 
    {{13, 9}, {0b1111111111101000, 16}}, 
    {{13, 10}, {0b1111111111101001, 16}},
    
    // Run = 14
    {{14, 1}, {0b1111111111101010, 16}}, 
    {{14, 2}, {0b1111111111101011, 16}}, 
    {{14, 3}, {0b1111111111101100, 16}},
    {{14, 4}, {0b1111111111101101, 16}},
    {{14, 5}, {0b1111111111101110, 16}}, 
    {{14, 6}, {0b1111111111101111, 16}}, 
    {{14, 7}, {0b1111111111110000, 16}},
    {{14, 8}, {0b1111111111110001, 16}}, 
    {{14, 9}, {0b1111111111110010, 16}}, 
    {{14, 10}, {0b1111111111110011, 16}},
    
    // Run = 15 (Special Cases)
    // {15, 0} is ZRL (Zero Run Length)
    {{15, 0}, {0b11111111011, 11}}, 
    {{15, 1}, {0b1111111111110100, 16}}, 
    {{15, 2}, {0b1111111111110101, 16}},
    {{15, 3}, {0b1111111111110110, 16}},
    {{15, 4}, {0b1111111111110111, 16}}, 
    {{15, 5}, {0b1111111111111000, 16}}, 
    {{15, 6}, {0b1111111111111001, 16}},
    {{15, 7}, {0b1111111111111010, 16}}, 
    {{15, 8}, {0b1111111111111011, 16}}, 
    {{15, 9}, {0b1111111111111100, 16}},
    {{15, 10}, {0b1111111111111101, 16}}
};

// DC Luminance Huffman Table initialization
HuffmanTable dcLuminanceHuffmanTable = {
    // Format: {{category, symbol}, {code, bit_length}}
    // Category = number of bits needed to represent the difference
    {{0, 0}, {0b00, 2}},        // 0: 00
    {{1, 0}, {0b010, 3}},       // 1: 010
    {{2, 0}, {0b011, 3}},       // 2: 011
    {{3, 0}, {0b100, 3}},       // 3: 100
    {{4, 0}, {0b101, 3}},       // 4: 101
    {{5, 0}, {0b110, 3}},       // 5: 110
    {{6, 0}, {0b1110, 4}},      // 6: 1110
    {{7, 0}, {0b11110, 5}},     // 7: 11110
    {{8, 0}, {0b111110, 6}},    // 8: 111110
    {{9, 0}, {0b1111110, 7}},   // 9: 1111110
    {{10, 0}, {0b11111110, 8}}, // 10: 11111110
    {{11, 0}, {0b111111110, 9}} // 11: 111111110
};

// AC Chrominance Huffman Table initialization
HuffmanTable acChrominanceHuffmanTable = {
    // Format: {{run, size}, {code, bit_length}}
    // {0, 0} is End of Block (EOB)
    {{0, 0}, {0b00, 2}},
    
    // Run = 0
    {{0, 1}, {0b01, 2}},          {{0, 2}, {0b100, 3}},          
    {{0, 3}, {0b1010, 4}},         {{0, 4}, {0b11000, 5}},
    {{0, 5}, {0b11001, 5}},        {{0, 6}, {0b111000, 6}},     
    {{0, 7}, {0b1111000, 7}},      {{0, 8}, {0b111110100, 9}},
    {{0, 9}, {0b1111110110, 10}},  {{0, 10}, {0b111111110100, 12}},
    
    // Run = 1
    {{1, 1}, {0b1011, 4}},         {{1, 2}, {0b111001, 6}},       
    {{1, 3}, {0b11110110, 8}},     {{1, 4}, {0b111110101, 9}},
    {{1, 5}, {0b1111111000, 10}},  {{1, 6}, {0b111111110101, 12}},
    {{1, 7}, {0b111111111000010, 15}}, 
    {{1, 8}, {0b111111111000011, 15}}, 
    {{1, 9}, {0b111111111000100, 15}}, 
    {{1, 10}, {0b111111111000101, 15}},
    
    // Run = 2
    {{2, 1}, {0b11010, 5}},        {{2, 2}, {0b11110111, 8}},    
    {{2, 3}, {0b1111110111, 10}},  {{2, 4}, {0b111111110110, 12}},
    {{2, 5}, {0b111111111000110, 15}}, 
    {{2, 6}, {0b111111111000111, 15}}, 
    {{2, 7}, {0b111111111001000, 15}},
    {{2, 8}, {0b111111111001001, 15}}, 
    {{2, 9}, {0b111111111001010, 15}}, 
    {{2, 10}, {0b111111111001011, 15}},
    
    // Run = 3
    {{3, 1}, {0b111010, 6}},       {{3, 2}, {0b11111000, 7}},   
    {{3, 3}, {0b1111111001, 10}},  {{3, 4}, {0b111111111001100, 15}},
    {{3, 5}, {0b111111111001101, 15}}, 
    {{3, 6}, {0b111111111001110, 15}}, 
    {{3, 7}, {0b111111111001111, 15}},
    {{3, 8}, {0b111111111010000, 15}}, 
    {{3, 9}, {0b111111111010001, 15}}, 
    {{3, 10}, {0b111111111010010, 15}},
    
    // Run = 4
    {{4, 1}, {0b111011, 6}},       {{4, 2}, {0b111110110, 9}},  
    {{4, 3}, {0b111111111010011, 15}},
    {{4, 4}, {0b111111111010100, 15}},
    {{4, 5}, {0b111111111010101, 15}}, 
    {{4, 6}, {0b111111111010110, 15}}, 
    {{4, 7}, {0b111111111010111, 15}},
    {{4, 8}, {0b111111111011000, 15}}, 
    {{4, 9}, {0b111111111011001, 15}}, 
    {{4, 10}, {0b111111111011010, 15}},
    
    // Run = 5
    {{5, 1}, {0b1111001, 7}},      {{5, 2}, {0b111111010, 9}},  
    {{5, 3}, {0b111111111011011, 15}},
    {{5, 4}, {0b111111111011100, 15}},
    {{5, 5}, {0b111111111011101, 15}}, 
    {{5, 6}, {0b111111111011110, 15}}, 
    {{5, 7}, {0b111111111011111, 15}},
    {{5, 8}, {0b111111111100000, 15}}, 
    {{5, 9}, {0b111111111100001, 15}}, 
    {{5, 10}, {0b111111111100010, 15}},
    
    // Run = 6
    {{6, 1}, {0b1111010, 7}},      {{6, 2}, {0b1111111001, 10}},  
    {{6, 3}, {0b111111111100011, 15}},
    {{6, 4}, {0b111111111100100, 15}},
    {{6, 5}, {0b111111111100101, 15}}, 
    {{6, 6}, {0b111111111100110, 15}}, 
    {{6, 7}, {0b111111111100111, 15}},
    {{6, 8}, {0b111111111101000, 15}}, 
    {{6, 9}, {0b111111111101001, 15}}, 
    {{6, 10}, {0b111111111101010, 15}},
    
    // Run = 7
    {{7, 1}, {0b11111001, 8}},     {{7, 2}, {0b1111111010, 10}}, 
    {{7, 3}, {0b111111111101011, 15}},
    {{7, 4}, {0b111111111101100, 15}},
    {{7, 5}, {0b111111111101101, 15}}, 
    {{7, 6}, {0b111111111101110, 15}}, 
    {{7, 7}, {0b111111111101111, 15}},
    {{7, 8}, {0b111111111110000, 15}}, 
    {{7, 9}, {0b111111111110001, 15}}, 
    {{7, 10}, {0b111111111110010, 15}},
    
    // Run = 8
    {{8, 1}, {0b11111010, 8}},     {{8, 2}, {0b111111111110011, 15}}, 
    {{8, 3}, {0b111111111110100, 15}},
    {{8, 4}, {0b111111111110101, 15}},
    {{8, 5}, {0b111111111110110, 15}}, 
    {{8, 6}, {0b111111111110111, 15}}, 
    {{8, 7}, {0b111111111111000, 15}},
    {{8, 8}, {0b111111111111001, 15}}, 
    {{8, 9}, {0b111111111111010, 15}}, 
    {{8, 10}, {0b111111111111011, 15}},
    
    // Run = 9
    {{9, 1}, {0b111111000, 9}},    {{9, 2}, {0b111111111111100, 15}}, 
    {{9, 3}, {0b111111111111101, 15}},
    {{9, 4}, {0b111111111111110, 15}},
    {{9, 5}, {0b111111111111111, 15}}, 
    {{9, 6}, {0b111111111100000, 15}}, 
    {{9, 7}, {0b111111111100001, 15}},
    {{9, 8}, {0b111111111100010, 15}}, 
    {{9, 9}, {0b111111111100011, 15}}, 
    {{9, 10}, {0b111111111100100, 15}},
    
    // Run = 10
    {{10, 1}, {0b111111001, 9}},   {{10, 2}, {0b111111111100101, 15}}, 
    {{10, 3}, {0b111111111100110, 15}},
    {{10, 4}, {0b111111111100111, 15}},
    {{10, 5}, {0b111111111101000, 15}}, 
    {{10, 6}, {0b111111111101001, 15}}, 
    {{10, 7}, {0b111111111101010, 15}},
    {{10, 8}, {0b111111111101011, 15}}, 
    {{10, 9}, {0b111111111101100, 15}}, 
    {{10, 10}, {0b111111111101101, 15}},
    
    // Run = 11 (Special Cases)
    {{15, 0}, {0b1111111011, 10}}, // ZRL (Zero Run Length)
    {{11, 1}, {0b111111010, 9}},   
    {{11, 2}, {0b111111111100111, 15}}, 
    {{11, 3}, {0b111111111101000, 15}}, 
    {{11, 4}, {0b111111111101001, 15}},
    {{11, 5}, {0b111111111101010, 15}}, 
    {{11, 6}, {0b111111111101011, 15}}, 
    {{11, 7}, {0b111111111101100, 15}},
    {{11, 8}, {0b111111111101101, 15}}, 
    {{11, 9}, {0b111111111101110, 15}}, 
    {{11, 10}, {0b111111111101111, 15}}
};

// DC Chrominance Huffman Table initialization
HuffmanTable dcChrominanceHuffmanTable = {
    // Format: {{category, symbol}, {code, bit_length}}
    // Category = number of bits needed to represent the difference
    {{0, 0}, {0b00, 2}},         // 0: 00
    {{1, 0}, {0b01, 2}},         // 1: 01
    {{2, 0}, {0b10, 2}},         // 2: 10
    {{3, 0}, {0b110, 3}},        // 3: 110
    {{4, 0}, {0b1110, 4}},       // 4: 1110
    {{5, 0}, {0b11110, 5}},      // 5: 11110
    {{6, 0}, {0b111110, 6}},     // 6: 111110
    {{7, 0}, {0b1111110, 7}},    // 7: 1111110
    {{8, 0}, {0b11111110, 8}},   // 8: 11111110
    {{9, 0}, {0b111111110, 9}},  // 9: 111111110
    {{10, 0}, {0b1111111110, 10}} // 10: 1111111110
};

void encodeAC(RunLengthResultAC rledAC) {
    BitWriter writer;

    for (auto ac : rledAC) {
        auto [code, bitlen] = acLuminanceHuffmanTable[ac];
        writer.write(code, bitlen);
    }

    writer.flush();
}

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

void writeHuffmanTable(BitWriter& w, const HuffmanTable& table, int id) {
    w.write(0xFFC4, 16);  // DHT marker, 16 bits

    // Count the bitlen frequencies 
    std::array<int, 32> bitlen_freq = {0};;

    // HuffmanSymbol is {run_length, size} pair
    using HuffmanSymbol = std::pair<int,int>;                  
    // Within the same bitlen sort the symbols num value
    using HuffmanSymbolsByCode = std::map<int, HuffmanSymbol>;
    // Vector of Bitlen -> SortedSymbols
    std::vector<HuffmanSymbolsByCode> symbols(17);  // 0 unused!

    for (auto [k,v]: table) {
        auto [code, bitlen] = v;
        symbols[bitlen].emplace(code, k);
        bitlen_freq[bitlen]++;
    }
    auto sector_length = 3 + // Length (2 bytes) + Table Class 
                        16 + // Frequencies of symbols of varying bitlengths
            std::accumulate(bitlen_freq.begin(), bitlen_freq.end(), 0); // No. of symbols
    w.write(sector_length, 16);     // Length of DHT segment, 16 bits (example length)
    w.write(id, 8);                 // Table Class (DC=0) and Table ID, 8 bits

    // Write frequencies of bitlengths in header
    for (int i = 1; i <= 16; i++) {
        auto freq = bitlen_freq[i];
        w.write(freq, 8);
    }

    // Write corresponding codes in order of incresing bitlengths
    for (auto huff_symbols_by_code: symbols) {
        for (auto [code, symbol] : huff_symbols_by_code) {
            auto [runlength, size] = symbol;
            uint8_t packed_symbol = (runlength << 4) | size; 
            w.write(packed_symbol, 8);
        }
    }
}

    for (int i = 1; i <= 16; i++) {

enum class QuantizeMode { Quantize, Dequantize };

void quantize(Block& block, const Block& quantTable, QuantizeMode mode) {
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            auto& elem = block.pixels[x][y];
            const auto& quant = quantTable.pixels[x][y];
            
            if (mode == QuantizeMode::Quantize) {
                elem = std::round(elem / quant);
            } else {
                elem = elem * quant;
            }
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
            }
        }
    }
}

void writeJpegHeader() {
    std::ofstream file("test.jpeg", std::ofstream::binary);
    
    // The structure of JPEG header:
    // +-----------------+
    // | SOI Marker      | Start of Image - Marks the beginning of the JPEG file
    // | (0xFFD8)        |
    // +-----------------+
    // | APP0 Marker     | Application Marker - Usually contains JFIF or Exif metadata
    // | (0xFFE0)        |
    // |  - Identifier   |   (e.g., "JFIF\0")
    // |  - Version      |   (e.g., 1.01)
    // |  - Density Info |   (pixel density and aspect ratio)
    // |  - Thumbnail    |   (optional thumbnail image)
    // +-----------------+
    // | DQT Marker      | Define Quantization Table - Stores quantization tables
    // | (0xFFDB)        |   (Used for compressing image data)
    // |  - Table Info   |
    // |  - Quant. Data  |
    // +-----------------+
    // | SOF Marker      | Start of Frame - Indicates the start of image data
    // | (0xFFC0)        |   (Baseline DCT in most JPEGs)
    // |  - Data Format  |
    // |  - Image Size   |   (height and width in pixels)
    // |  - Components   |   (e.g., Y, Cb, Cr for color channels)
    // +-----------------+
    // | DHT Marker      | Define Huffman Table - Stores Huffman encoding tables
    // | (0xFFC4)        |   (Used for encoding the compressed image data)
    // |  - Table Info   |
    // |  - Huffman Data |
    // +-----------------+
    // | SOS Marker      | Start of Scan - Marks the start of encoded image data
    // | (0xFFDA)        |   (contains compressed image data)
    // |  - Component IDs|
    // |  - Compression  |
    // |  - Data Stream  |
    // +-----------------+
    // | Image Data      | Encoded Image Data - Compressed image data
    // | (Compressed     |   (Uses Huffman coding and quantization)
    // |  Bytes)         |
    // +-----------------+
    // | EOI Marker      | End of Image - Marks the end of the JPEG file
    // | (0xFFD9)        |
    // +-----------------+

    BitWriter jpegHeaderWriter;

    // SOI Marker (Start of Image)
    jpegHeaderWriter.write(0xFFD8, 16);  // SOI marker, 16 bits

    // APP0 Marker (Application Marker for metadata, e.g., JFIF or Exif)
    jpegHeaderWriter.write(0xFFE0, 16);      // APP0 marker, 16 bits
    jpegHeaderWriter.write(0x0010, 16);      // Length of APP0 segment, 16 bits (example length)
    jpegHeaderWriter.write(0x4A464946, 32);  // "JFIF" identifier, 32 bits (ASCII: 'JFIF')
    jpegHeaderWriter.write(0x00, 8);         // Null terminator for identifier, 8 bits
    jpegHeaderWriter.write(0x0101, 16);      // JFIF version (e.g., 1.1), 16 bits
    jpegHeaderWriter.write(0x01, 8);         // Density units, 8 bits (1 for DPI)
    jpegHeaderWriter.write(0x0048, 16);      // X density, 16 bits (example value)
    jpegHeaderWriter.write(0x0048, 16);      // Y density, 16 bits (example value)
    jpegHeaderWriter.write(0x00, 8);         // Thumbnail width, 8 bits (0 for none)
    jpegHeaderWriter.write(0x00, 8);         // Thumbnail height, 8 bits (0 for none)

    // Omit DQT Marker (do not define custom quantization table)

    // SOF Marker (Start of Frame, Baseline DCT)
    jpegHeaderWriter.write(0xFFC0, 16);  // SOF marker, 16 bits
    jpegHeaderWriter.write(0x0011, 16);  // Length of SOF segment, 16 bits (example length)
    jpegHeaderWriter.write(0x08, 8);     // Precision, 8 bits (usually 8 for baseline JPEG)
    jpegHeaderWriter.write(0x0400, 16);  // Height, 16 bits (example: 1024)
    jpegHeaderWriter.write(0x0400, 16);  // Width, 16 bits (example: 1024)
    jpegHeaderWriter.write(0x03, 8);     // Number of components, 8 bits (3 for Y, Cb, Cr)
    // Component data for Y, Cb, Cr (example values)
    jpegHeaderWriter.write(0x01, 8);     // Component ID for Y, 8 bits
    jpegHeaderWriter.write(0x22, 8);     // Sampling factors for Y (4 bits horizontal, 4 bits vertical)
    jpegHeaderWriter.write(0x00, 8);     // Quantization table ID for Y, 8 bits
    jpegHeaderWriter.write(0x02, 8);     // Component ID for Cb, 8 bits
    jpegHeaderWriter.write(0x11, 8);     // Sampling factors for Cb, 8 bits
    jpegHeaderWriter.write(0x01, 8);     // Quantization table ID for Cb, 8 bits
    jpegHeaderWriter.write(0x03, 8);     // Component ID for Cr, 8 bits
    jpegHeaderWriter.write(0x11, 8);     // Sampling factors for Cr, 8 bits
    jpegHeaderWriter.write(0x01, 8);     // Quantization table ID for Cr, 8 bits

    // DHT Marker (Define Huffman Table)
    writeHuffmanTable(jpegHeaderWriter, acLuminanceHuffmanTable,    0b10000);
    writeHuffmanTable(jpegHeaderWriter, dcChrominanceHuffmanTable,  0b00001);
    writeHuffmanTable(jpegHeaderWriter, dcLuminanceHuffmanTable,    0b00010);
    writeHuffmanTable(jpegHeaderWriter, acChrominanceHuffmanTable,  0b10011);

    // SOS Marker (Start of Scan)
    jpegHeaderWriter.write(0xFFDA, 16);  // SOS marker, 16 bits
    jpegHeaderWriter.write(0x000C, 16);  // Length of SOS segment, 16 bits
    jpegHeaderWriter.write(0x03, 8);     // Number of components, 8 bits
    // Component data (for each component in scan)
    jpegHeaderWriter.write(0x01, 8);     // Component ID (Y), 8 bits
    jpegHeaderWriter.write(0x00, 8);     // DC and AC Huffman table selector for Y, 8 bits
    jpegHeaderWriter.write(0x02, 8);     // Component ID (Cb), 8 bits
    jpegHeaderWriter.write(0x11, 8);     // DC and AC Huffman table selector for Cb, 8 bits
    jpegHeaderWriter.write(0x03, 8);     // Component ID (Cr), 8 bits
    jpegHeaderWriter.write(0x11, 8);     // DC and AC Huffman table selector for Cr, 8 bits
    jpegHeaderWriter.write(0x00, 8);     // Start of spectral selection, 8 bits
    jpegHeaderWriter.write(0x3F, 8);     // End of spectral selection, 8 bits
    jpegHeaderWriter.write(0x00, 8);     // Successive approximation, 8 bits


    // Image Data (compressed binary data for image content, Huffman coded)
    // In real JPEG files, this would contain the compressed data following the Huffman coding.
    // For this example, this part is omitted.

    // EOI Marker (End of Image)
    jpegHeaderWriter.write(0xFFD9, 16);  // EOI marker, 16 bits
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


    quantize(dctBlock, luminanceQuantTable, QuantizeMode::Quantize);
    // quantize(dctBlock, chrominanceQuantTable);
    // quantize(dctBlock, chrominanceQuantTable);

    // Perform IDCT on the DCT coefficients block
    Block reconstructedBlock = IDCT(dctBlock);
    std::cout << "Reconstructed Signal Block:\n";
    printBlock(reconstructedBlock);

    return 0;
}