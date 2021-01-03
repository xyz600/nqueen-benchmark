#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

struct State
{
    std::uint8_t row;
    std::uint32_t column_bitmap;
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
    for (std::size_t column = 0; column < n; column++)
    {
        const std::uint32_t column_bit = 1u << column;
        const std::uint64_t upper_left_bit = 1ull << (n - 1 - state.row + column);
        const std::uint64_t upper_right_bit = 1ull << (state.row + column);
        if (((state.column_bitmap & column_bit) == 0u) && ((state.upper_left_bitmap & upper_left_bit) == 0ull) && ((state.upper_right_bitmap & upper_right_bit) == 0ull))
        {
            state.column_bitmap |= column_bit;
            state.upper_left_bitmap |= upper_left_bit;
            state.upper_right_bitmap |= upper_right_bit;
            state.row++;
            counter += solve(state, n);
            state.column_bitmap &= ~column_bit;
            state.upper_left_bitmap &= ~upper_left_bit;
            state.upper_right_bitmap &= ~upper_right_bit;
            state.row--;
        }
    }
    return counter;
}

std::uint64_t cpu_solve(const std::size_t n)
{
    std::vector<std::size_t> counter(n, 0);
#pragma omp parallel for
    for (std::size_t column = 0; column < n; column++)
    {
        State state;
        state.clear();
        state.column_bitmap |= 1u << column;
        state.upper_left_bitmap |= 1ull << (n - 1 + column);
        state.upper_right_bitmap |= 1ull << column;
        state.row++;
        counter[column] = solve(state, n);
    }
    return std::accumulate(counter.begin(), counter.end(), 0);
}

int main()
{
    for (std::size_t n = 8; n <= 17; n++)
    {
        const auto start = std::chrono::system_clock::now();
        const auto count = cpu_solve(n);
        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "result count: " << count << ", elapsed = " << elapsed << "[ms]" << std::endl;
    }
}