#include "cuda_utility.cuh"

#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <omp.h>
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
    __device__ __host__ void push(std::uint32_t column_bit, const State& prev)
    {
        row = prev.row + 1;
        column_bitmap = prev.column_bitmap | column_bit;
        upper_left_bitmap = (prev.upper_left_bitmap | column_bit) >> 1;
        upper_right_bitmap = (prev.upper_right_bitmap | column_bit) << 1;
    }

    __device__ __host__ void push_self(std::uint32_t column_bit)
    {
        row++;
        column_bitmap |= column_bit;
        upper_left_bitmap = (upper_left_bitmap | column_bit) >> 1;
        upper_right_bitmap = (upper_right_bitmap | column_bit) << 1;
    }

    __device__ __host__ bool can_push(std::uint32_t column_bit)
    {
        return ((column_bitmap | upper_left_bitmap | upper_right_bitmap) & column_bit) == 0;
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

__device__ std::uint32_t simulate(const State& init, const std::uint32_t n)
{
    std::uint32_t counter = 0;

    __shared__ State sh_stack[96][32];

    const auto* ptr_end = &sh_stack[threadIdx.x][0];
    auto* ptr_top = &sh_stack[threadIdx.x][0];
    *ptr_top = init;
    ptr_top++;

    const std::uint32_t all = (1u << n) - 1;

    // スタックの最初で、column が終了に到達したら
    while (ptr_top != ptr_end)
    {
        const State state = *(--ptr_top);

        if (state.row == n)
        {
            counter++;
        }
        else
        {
            std::uint32_t bitmask = (~(state.column_bitmap | state.upper_left_bitmap | state.upper_right_bitmap)) & all;

            while (bitmask > 0)
            {
                const auto least_mask = -bitmask & bitmask;

                ptr_top->push(least_mask, state);
                ptr_top++;

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
    constexpr std::size_t stream_size = 4;

    const auto dev_task_list = cuda::make_unique<TaskList<MaxTaskSize>[]>(stream_size * 2);
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
    block_size = 96;

    std::vector<std::tuple<int, int, int>> c1c2;
    for (int col1 = 0; col1 < n; col1++)
    {
        for (int col2 = 0; col2 < n; col2++)
        {
            if (abs(col1 - col2) > 1)
            {
                c1c2.emplace_back(col1, col2, 1);
            }
        }
    }

    auto host_task_list = std::make_unique<TaskList<MaxTaskSize>[]>(stream_size * 2);

#pragma omp parallel for num_threads(stream_size)
    for (std::size_t i = 0; i < c1c2.size(); i++)
    {
        const int stream_idx = omp_get_thread_num();
        int buffer_index = stream_idx * 2;
        host_task_list[buffer_index].task_size = 0;
        host_task_list[buffer_index].index = 0;

        const auto& stream = stream_array[stream_idx];

        {
            State init;
            init.clear();
            int column1, column2, rate;
            std::tie(column1, column2, rate) = c1c2[i];

            init.push_self(1u << column1);
            init.push_self(1u << column2);

            // TODO: multi thread
            State stack[128];
            stack[0] = init;
            State* ptr_top = &stack[1];

            const std::uint32_t all = (1u << n) - 1;

            auto subsolve = [&]() {
                CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
                CHECK_CUDA_ERROR(cudaMemcpyAsync(&dev_task_list[buffer_index], &host_task_list[buffer_index], sizeof(TaskList<MaxTaskSize>), cudaMemcpyHostToDevice, stream));
                solve<<<min_grid_size, block_size, 0, stream>>>(&dev_task_list[buffer_index], dev_counter.get(), n);

                buffer_index = (buffer_index % 2 == 0 ? buffer_index + 1 : buffer_index - 1);
                host_task_list[buffer_index].task_size = 0;
            };

            while (ptr_top != stack)
            {
                const auto state = *(--ptr_top);

                if (state.row >= std::max<std::size_t>(0u, n - 11))
                {
                    host_task_list[buffer_index].data[host_task_list[buffer_index].task_size++] = state;
                    if (host_task_list[buffer_index].task_size == MaxTaskSize)
                    {
                        subsolve();
                    }
                }
                else
                {
                    std::uint32_t bitmask = (~(state.column_bitmap | state.upper_left_bitmap | state.upper_right_bitmap)) & all;
                    while (bitmask > 0)
                    {
                        const auto least_mask = -bitmask & bitmask;
                        ptr_top->push(least_mask, state);
                        ptr_top++;
                        bitmask -= least_mask;
                    }
                }
            }

            if (host_task_list[buffer_index].task_size > 0)
            {
                subsolve();
            }
        }
    }

    for (std::size_t i = 0; i < stream_size; i++)
    {
        cudaStreamSynchronize(stream_array[i]);
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