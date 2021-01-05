#include "cuda_utility.cuh"

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

template <std::uint32_t MaxSize>
struct TaskList
{
    State data[MaxSize];
    std::uint32_t index;
    std::uint32_t task_size;
};

constexpr std::uint32_t MaxTaskSize = 1024 * 1024;

struct Stack
{
    State data[1024];
    std::uint32_t top_index;
};

__device__ std::uint32_t simulate(State& init_state, const std::uint32_t n)
{
    std::uint32_t counter = 0;

    Stack stack;
    stack.top_index = 1;
    stack.data[0] = init_state;

    while (stack.top_index > 0)
    {
        const auto state = stack.data[--stack.top_index];

        if (state.row == n)
        {
            counter++;
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
                stack.data[stack.top_index].column_bitmap = state.column_bitmap | column_bit;
                stack.data[stack.top_index].upper_left_bitmap = state.upper_left_bitmap | upper_left_bit;
                stack.data[stack.top_index].upper_right_bitmap = state.upper_right_bitmap | upper_right_bit;
                stack.data[stack.top_index].row = state.row + 1;

                stack.top_index++;

                bitmask -= least_mask;
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

std::uint64_t gpu_solve(const std::size_t n)
{
    constexpr std::size_t stream_size = 1;

    const auto dev_task_list = cuda::make_unique<TaskList<MaxTaskSize>[]>(stream_size);
    const auto dev_counter = cuda::make_unique_managed<std::uint32_t>();
    *dev_counter = 0;

    std::array<cudaStream_t, stream_size> stream_array;
    for (std::size_t i = 0; i < stream_size; i++)
    {
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_array[i]));
    }

    int min_grid_size, block_size;
    CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, solve));
    min_grid_size = 14;

    // generate initial solution by cpu
    {
        // TODO: multi thread
        Stack stack;
        stack.top_index = 1;
        stack.data[0].clear();

        std::size_t thrown_index = 0;
        auto host_task_list = std::make_unique<TaskList<MaxTaskSize>>();
        host_task_list->index = 0;
        host_task_list->task_size = 0;

        while (stack.top_index > 0)
        {
            const auto state = stack.data[--stack.top_index];

            if (state.row == std::max<std::size_t>(0u, n - 8))
            {
                host_task_list->data[host_task_list->task_size++] = state;
                if (host_task_list->task_size == MaxTaskSize)
                {
                    CHECK_CUDA_ERROR(cudaMemcpyAsync(&dev_task_list[thrown_index], host_task_list.get(), sizeof(TaskList<MaxTaskSize>), cudaMemcpyHostToDevice, stream_array[thrown_index]));
                    solve<<<min_grid_size, block_size, 0, stream_array[thrown_index]>>>(dev_task_list.get(), dev_counter.get(), n);

                    thrown_index = (thrown_index + 1) % stream_size;
                    host_task_list->task_size = 0;
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
                    stack.data[stack.top_index].column_bitmap = state.column_bitmap | column_bit;
                    stack.data[stack.top_index].upper_left_bitmap = state.upper_left_bitmap | upper_left_bit;
                    stack.data[stack.top_index].upper_right_bitmap = state.upper_right_bitmap | upper_right_bit;
                    stack.data[stack.top_index].row = state.row + 1;

                    stack.top_index++;
                    bitmask -= least_mask;
                }
            }
        }

        if (host_task_list->task_size > 0)
        {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(&dev_task_list[thrown_index], host_task_list.get(), sizeof(TaskList<MaxTaskSize>), cudaMemcpyHostToDevice, stream_array[thrown_index]));
            solve<<<min_grid_size, block_size, 0, stream_array[thrown_index]>>>(dev_task_list.get(), dev_counter.get(), n);
        }
    }

    for (std::size_t i = 0; i < stream_size; i++)
    {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_array[i]));
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream_array[i]));
    }
    return *dev_counter;
}

int main()
{
    for (std::size_t n = 8; n <= 17; n++)
    {
        const auto start = std::chrono::system_clock::now();
        const auto count = gpu_solve(n);
        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "result count: " << count << ", elapsed = " << elapsed << "[ms]" << std::endl;
    }
    return 0;
}