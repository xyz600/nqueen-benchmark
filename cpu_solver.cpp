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
    std::uint32_t row;
    std::uint32_t column_bitmap;
    std::uint32_t upper_left_bitmap;
    std::uint32_t upper_right_bitmap;

    void clear()
    {
        row = 0;
        column_bitmap = 0;
        upper_left_bitmap = 0;
        upper_right_bitmap = 0;
    }
    void push(std::uint32_t column_bit, const State& prev)
    {
        row = prev.row + 1;
        column_bitmap = prev.column_bitmap | column_bit;
        upper_left_bitmap = (prev.upper_left_bitmap | column_bit) >> 1;
        upper_right_bitmap = (prev.upper_right_bitmap | column_bit) << 1;
    }

    void push_self(std::uint32_t column_bit)
    {
        row++;
        column_bitmap |= column_bit;
        upper_left_bitmap = (upper_left_bitmap | column_bit) >> 1;
        upper_right_bitmap = (upper_right_bitmap | column_bit) << 1;
    }

    bool can_push(std::uint32_t column_bit)
    {
        return ((column_bitmap | upper_left_bitmap | upper_right_bitmap) & column_bit) == 0;
    }
};

std::uint64_t solve(const State& state, const std::size_t n)
{
    if (state.row == n)
    {
        return 1ull;
    }
    std::uint64_t counter = 0;

    auto bitmask = ~(state.column_bitmap | state.upper_left_bitmap | state.upper_right_bitmap) & ((1ull << n) - 1ull);
    while (bitmask > 0)
    {
        const auto least_mask = -bitmask & bitmask;
        State next;
        next.push(least_mask, state);

        counter += solve(next, n);

        bitmask -= least_mask;
    }
    return counter;
}

std::uint64_t cpu_solve(const std::size_t n)
{
    std::vector<std::tuple<int, int, int>> c1c2;
    const std::size_t col1_max = n / 2 + (n % 2 == 0 ? 0 : 1);
    for (int col1 = 0; col1 < col1_max; col1++)
    {
        for (int col2 = 0; col2 < n; col2++)
        {
            if (abs(col1 - col2) > 1)
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
    }
    std::vector<std::size_t> counter(c1c2.size(), 0);
#pragma omp parallel for
    for (std::size_t i = 0; i < c1c2.size(); i++)
    {
        State state;
        state.clear();
        const auto [column1, column2, rate] = c1c2[i];

        state.push_self(1u << column1);
        state.push_self(1u << column2);

        counter[i] = solve(state, n) * rate;
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