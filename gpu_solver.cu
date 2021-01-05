#include "cuda_utility.cuh"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <vector>

struct State
{
    std::uint32_t row;
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

struct PartialState
{
    std::uint8_t row;
    std::uint8_t column;
};

template <std::uint32_t MaxSize>
struct TaskList
{
    State data[MaxSize];
    std::uint32_t index;
    std::uint32_t task_size;
};

constexpr std::uint32_t MaxTaskSize = 1024 * 1024;

__device__ std::uint32_t simulate(State& state, const std::uint32_t n)
{
    std::uint32_t counter = 0;

    __shared__ std::uint32_t sh_stack[1024][10];
    auto* ptr_column = &sh_stack[threadIdx.x][0];
    *ptr_column = 0;

    const int init_row = state.row;

    // その column が探索終了した時に stack を戻す
    auto rollback = [&]() {
        if (init_row != state.row)
        {
            state.row--;
            ptr_column--;

            const std::uint32_t least_mask = 1u << *ptr_column;
            const auto column_bit = least_mask;
            const auto upper_left_bit = static_cast<std::uint64_t>(least_mask) << (n - 1 - state.row);
            const auto upper_right_bit = static_cast<std::uint64_t>(least_mask) << state.row;

            state.column_bitmap ^= column_bit;
            state.upper_left_bitmap ^= upper_left_bit;
            state.upper_right_bitmap ^= upper_right_bit;
        }
        (*ptr_column)++;
    };

    auto advance = [&](const std::uint32_t least_mask) {
        const auto column_bit = least_mask;
        const auto upper_left_bit = static_cast<std::uint64_t>(least_mask) << (n - 1 - state.row);
        const auto upper_right_bit = static_cast<std::uint64_t>(least_mask) << state.row;
        *ptr_column = 31 - __clz(least_mask);

        state.column_bitmap ^= column_bit;
        state.upper_left_bitmap ^= upper_left_bit;
        state.upper_right_bitmap ^= upper_right_bit;

        state.row++;
        *(++ptr_column) = 0;
    };

    // スタックの最初で、column が終了に到達したら
    while (!(state.row == init_row && *ptr_column == n))
    {
        if (state.row == n)
        {
            counter++;
            rollback();
        }
        else
        {
            const std::uint32_t column_mask = ((1u << n) - 1u) - ((1u << *ptr_column) - 1);
            const std::uint32_t bitmask = ~static_cast<std::uint32_t>((state.column_bitmap | (state.upper_left_bitmap >> (n - 1 - state.row)) | (state.upper_right_bitmap >> state.row))) & column_mask;

            const auto least_mask = -bitmask & bitmask;

            if (least_mask != 0)
            {
                advance(least_mask);
            }
            else
            {
                rollback();
            }
        }
    }
    return counter;
}

__global__ void solve(TaskList<MaxTaskSize>* que, std::uint32_t* counter, const std::uint32_t n)
{
    std::uint32_t local_counter = 0;

    while (que->index < que->task_size)
    {
        const auto pos = atomicAdd(&que->index, 1);
        if (pos < que->task_size)
        {
            local_counter += simulate(que->data[pos], n);
        }
    }
    atomicAdd(counter, local_counter);
}

template <int MaxSize>
struct Stack
{
    State data[MaxSize];
};
std::uint64_t gpu_solve(const std::size_t n)
{
    constexpr std::size_t stream_size = 4;

    const auto dev_task_list = cuda::make_unique<TaskList<MaxTaskSize>[]>(stream_size);
    const auto dev_counter = cuda::make_unique<std::uint32_t>();
    {
        std::uint32_t tmp = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(dev_counter.get(), &tmp, sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    }

    std::array<cudaStream_t, stream_size> stream_array;
    for (std::size_t i = 0; i < stream_size; i++)
    {
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_array[i]));
    }

    int min_grid_size, block_size;
    CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, solve));

    // generate initial solution by cpu
    {
        std::size_t thrown_index = 0;
        auto host_task_list = std::make_unique<TaskList<MaxTaskSize>[]>(stream_size);
        for (std::size_t i = 0; i < stream_size; i++)
        {
            host_task_list[i].index = 0;
            host_task_list[i].task_size = 0;
        }

        // TODO: multi thread
        Stack<128> stack;
        stack.data[0].clear();
        int counter = 0;
        State* ptr = &stack.data[1];

        while (ptr != stack.data)
        {
            const auto state = *(--ptr);

            if (state.row == std::max<std::size_t>(0u, n - 8))
            {
                host_task_list[thrown_index].data[host_task_list[thrown_index].task_size++] = state;
                if (host_task_list[thrown_index].task_size == MaxTaskSize)
                {
                    counter++;
                    CHECK_CUDA_ERROR(cudaMemcpyAsync(&dev_task_list[thrown_index], &host_task_list[thrown_index], sizeof(TaskList<MaxTaskSize>), cudaMemcpyHostToDevice, stream_array[thrown_index]));
                    solve<<<min_grid_size, block_size, 0, stream_array[thrown_index]>>>(&dev_task_list[thrown_index], dev_counter.get(), n);

                    thrown_index = (thrown_index + 1) % stream_size;
                    host_task_list[thrown_index].task_size = 0;
                }
            }
            else
            {
                auto bitmask = ~((state.column_bitmap | (state.upper_left_bitmap >> (n - 1 - state.row)) | (state.upper_right_bitmap >> state.row))) & ((1ull << n) - 1ull);
                while (bitmask > 0)
                {
                    const auto least_mask = -bitmask & bitmask;

                    const auto column_bit = least_mask;
                    const auto upper_left_bit = least_mask << (n - 1 - state.row);
                    const auto upper_right_bit = least_mask << state.row;

                    // push next state to stack
                    ptr->column_bitmap = state.column_bitmap | column_bit;
                    ptr->upper_left_bitmap = state.upper_left_bitmap | upper_left_bit;
                    ptr->upper_right_bitmap = state.upper_right_bitmap | upper_right_bit;
                    ptr->row = state.row + 1;

                    ptr++;
                    bitmask -= least_mask;
                }
            }
        }

        if (host_task_list[thrown_index].task_size > 0)
        {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(&dev_task_list[thrown_index], &host_task_list[thrown_index], sizeof(TaskList<MaxTaskSize>), cudaMemcpyHostToDevice, stream_array[thrown_index]));
            solve<<<min_grid_size, block_size, 0, stream_array[thrown_index]>>>(&dev_task_list[thrown_index], dev_counter.get(), n);
            counter++;
        }
    }

    for (std::size_t i = 0; i < stream_size; i++)
    {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_array[i]));
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream_array[i]));
    }
    std::uint32_t tmp;
    CHECK_CUDA_ERROR(cudaMemcpy(&tmp, dev_counter.get(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    return tmp;
}

int main()
{
    for (std::size_t n = 8; n <= 18; n++)
    {
        const auto start = std::chrono::system_clock::now();
        const auto count = gpu_solve(n);
        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "result count: " << count << ", elapsed = " << elapsed << "[ms]" << std::endl;
    }
    return 0;
}