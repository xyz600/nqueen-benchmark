#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

struct State
{
    std::uint64_t row;
    std::uint64_t column_bitmap;
    std::uint64_t upper_left_bitmap;
    std::uint64_t upper_right_bitmap;

    void clear()
    {
        row = 0;
        column_bitmap = 0;
        upper_left_bitmap = 0;
        upper_right_bitmap = 0;
    }
};

std::uint64_t solve(State& state, const std::size_t n)
{
    if (state.row == n)
    {
        return 1ull;
    }
    std::uint64_t counter = 0;

    auto bitmask = ~((state.column_bitmap | (state.upper_left_bitmap >> (n - 1 - state.row)) | (state.upper_right_bitmap >> state.row))) & ((1ull << n) - 1ull);
    while (bitmask > 0)
    {
        const auto least_mask = -bitmask & bitmask;

        const auto column_bit = least_mask;
        const auto upper_left_bit = least_mask << (n - 1 - state.row);
        const auto upper_right_bit = least_mask << state.row;

        state.column_bitmap ^= column_bit;
        state.upper_left_bitmap ^= upper_left_bit;
        state.upper_right_bitmap ^= upper_right_bit;
        state.row++;
        counter += solve(state, n);
        state.column_bitmap ^= column_bit;
        state.upper_left_bitmap ^= upper_left_bit;
        state.upper_right_bitmap ^= upper_right_bit;
        state.row--;

        bitmask -= least_mask;
    }
    return counter;
}

std::uint64_t cpu_solve(const std::size_t n)
{
    std::vector<std::tuple<int, int, int>> c1c2;
    const std::size_t col1_max = n / 2 + (n % 2 == 0 ? 0 : 1);
    for (std::size_t col1 = 0; col1 < col1_max; col1++)
    {
        for (std::size_t col2 = 0; col2 < n; col2++)
        {
            if (n % 2 == 0)
            {
                c1c2.emplace_back(col1, col2, 2);
            }
            else
            {
                c1c2.emplace_back(col1, col2, col1 == col1_max - 1 ? 1 : 2);
            }
        }
    }
    std::vector<std::size_t> counter(c1c2.size(), 0);
#pragma omp parallel for
    for (std::size_t i = 0; i < c1c2.size(); i++)
    {
        State state;
        state.clear();
        const auto [column1, column2, rate] = c1c2[i];

        state.column_bitmap |= 1u << column1;
        state.upper_left_bitmap |= 1ull << (n - 1 + column1);
        state.upper_right_bitmap |= 1ull << column1;
        state.row += 1;

        const std::uint32_t column_bit = 1u << column2;
        const std::uint64_t upper_left_bit = 1ull << (n - 1 - state.row + column2);
        const std::uint64_t upper_right_bit = 1ull << (state.row + column2);
        if (((state.column_bitmap & column_bit) == 0u) && ((state.upper_left_bitmap & upper_left_bit) == 0ull) && ((state.upper_right_bitmap & upper_right_bit) == 0ull))
        {
            state.column_bitmap |= column_bit;
            state.upper_left_bitmap |= upper_left_bit;
            state.upper_right_bitmap |= upper_right_bit;
            state.row++;
            counter[i] = solve(state, n) * rate;
        }
    }
    return std::accumulate(counter.begin(), counter.end(), 0);
}

int main()
{
    for (std::size_t n = 8; n <= 18; n++)
    {
        const auto start = std::chrono::system_clock::now();
        const auto count = cpu_solve(n);
        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "result count: " << count << ", elapsed = " << elapsed << "[ms]" << std::endl;
    }
}