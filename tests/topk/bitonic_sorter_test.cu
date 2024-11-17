#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "bits/cuch.hpp"
#include "bits/cuda_array.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/generator/data_generator.hpp"
#include "bits/knn.hpp"

#include "bits/topk/bitonic_sort.cuh"
#include "bits/topk/bitonic_sorter.cuh"

template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, bool USE_OLD = false>
__global__ void bitonic_sorter_kernel(float* global_keys, std::int32_t* global_values,
                                      std::size_t value_count)
{
    constexpr std::size_t VALUE_COUNT = BLOCK_SIZE * ITEMS_PER_THREAD;

    if (USE_OLD)
    {
        block_sort<true, 1>(soa_layout{global_keys, global_values}, value_count);
    }
    else
    {
        __shared__ float shm_keys[VALUE_COUNT];
        __shared__ std::int32_t shm_values[VALUE_COUNT];

        float keys[ITEMS_PER_THREAD];
        std::int32_t values[ITEMS_PER_THREAD];

        bitonic_sorter<BLOCK_SIZE, ITEMS_PER_THREAD> sorter{shm_keys, shm_values};
        sorter.sort(keys, values, global_keys, global_values, value_count);
    }
}

std::vector<float> random_permutation(std::size_t value_count)
{
    std::default_random_engine eng{42};

    std::vector<float> values(value_count);
    float val = 0.0f;
    for (std::size_t i = 0; i < value_count; ++i)
    {
        values[i] = val;
        val += 0.1f;
    }

    std::shuffle(values.begin(), values.end(), eng);
    return values;
}

template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, bool USE_OLD = false>
double run_bitonic_sorter_test(std::size_t value_count)
{
    constexpr std::size_t BLOCK_COUNT = 1;

    std::vector<float> keys_cpu = random_permutation(value_count);
    std::vector<std::int32_t> values_cpu(keys_cpu.size());
    for (std::size_t i = 0; i < values_cpu.size(); ++i)
    {
        values_cpu[i] = static_cast<std::int32_t>(i);
    }

    cuda_array<float, 2> keys_gpu{{1, value_count}};
    cuda_array<std::int32_t, 2> values_gpu{{1, value_count}};

    cuda_stream::make_default()
        .copy_to_gpu_async(keys_gpu.view(), keys_cpu.data())
        .copy_to_gpu_async(values_gpu.view(), values_cpu.data())
        .sync();

    const auto begin = std::chrono::steady_clock::now();

    bitonic_sorter_kernel<BLOCK_SIZE, ITEMS_PER_THREAD, USE_OLD><<<BLOCK_COUNT, BLOCK_SIZE>>>(
        keys_gpu.view().data(), values_gpu.view().data(), value_count);
    CUCH(cudaGetLastError());

    cuda_stream::make_default().sync();

    const auto end = std::chrono::steady_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;

    std::vector<float> output_keys_cpu(keys_cpu.size());
    std::vector<std::int32_t> output_values_cpu(keys_cpu.size());
    cuda_stream::make_default()
        .copy_from_gpu_async(output_keys_cpu.data(), keys_gpu.view())
        .copy_from_gpu_async(output_values_cpu.data(), values_gpu.view())
        .sync();

    // the result is sorted
    for (std::size_t i = 0; i + 1 < value_count; ++i)
    {
        REQUIRE(output_keys_cpu[i] <= output_keys_cpu[i + 1]);
    }

    for (std::size_t i = 0; i < value_count; ++i)
    {
        const auto key = keys_cpu[i];
        const auto value = values_cpu[i];

        // all keys are present
        const auto it = std::lower_bound(output_keys_cpu.begin(), output_keys_cpu.end(), key);
        REQUIRE(it != output_keys_cpu.end());
        REQUIRE(*it == key);

        // keys are properly matched with values
        const auto idx = std::distance(output_keys_cpu.begin(), it);
        const auto other_value = output_values_cpu[idx];
        REQUIRE(value == other_value);
    }
    return duration;
}

TEST_CASE("Sort small block", "[bitonic_sorter]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 4;
    constexpr std::size_t BLOCK_SIZE = 32;

    run_bitonic_sorter_test<BLOCK_SIZE, ITEMS_PER_THREAD>(BLOCK_SIZE * ITEMS_PER_THREAD);
}

TEST_CASE("Top level merge in global memory", "[bitonic_sorter]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 4;
    constexpr std::size_t BLOCK_SIZE = 32;

    run_bitonic_sorter_test<BLOCK_SIZE, ITEMS_PER_THREAD>(2 * BLOCK_SIZE * ITEMS_PER_THREAD);
}

TEST_CASE("Sort in global memory", "[bitonic_sorter]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 4;
    constexpr std::size_t BLOCK_SIZE = 32;

    run_bitonic_sorter_test<BLOCK_SIZE, ITEMS_PER_THREAD>(16 * BLOCK_SIZE * ITEMS_PER_THREAD);
}

TEST_CASE("Sort large blocks", "[bitonic_sorter]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 4;
    constexpr std::size_t BLOCK_SIZE = 256;

    run_bitonic_sorter_test<BLOCK_SIZE, ITEMS_PER_THREAD>(32 * BLOCK_SIZE * ITEMS_PER_THREAD);
}

TEST_CASE("Benchmark old and new global memory bitonic sort implementation",
          "[bitonic_sorter][.microbenchmark]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 4;
    constexpr std::size_t BLOCK_SIZE = 256;
    constexpr std::size_t mult = 512;

    for (std::size_t i = 0; i < 3; ++i)
    {
        auto duration = run_bitonic_sorter_test<BLOCK_SIZE, ITEMS_PER_THREAD, false>(
            mult * BLOCK_SIZE * ITEMS_PER_THREAD);
        std::cout << "Old version: " << duration << " s" << std::endl;

        duration = run_bitonic_sorter_test<BLOCK_SIZE, ITEMS_PER_THREAD, true>(mult * BLOCK_SIZE *
                                                                               ITEMS_PER_THREAD);
        std::cout << "New version: " << duration << " s" << std::endl;
    }
}
