#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include "bits/cuch.hpp"
#include "bits/cuda_array.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/generator/data_generator.hpp"
#include "bits/generator/uniform_generator.hpp"
#include "bits/knn.hpp"

#include "bits/topk/bitonic_sort_regs.cuh"

template <std::size_t ITEMS_PER_THREAD, std::size_t BLOCK_SIZE, order_t ORDER = order_t::ascending>
__global__ void sort_regs_test(float* data, knn::pair_t* result)
{
    constexpr std::size_t VALUE_COUNT = BLOCK_SIZE * ITEMS_PER_THREAD;

    __shared__ float shm_dist[2 * BLOCK_SIZE];
    __shared__ std::int32_t shm_label[2 * BLOCK_SIZE];

    float dist[ITEMS_PER_THREAD];
    std::int32_t label[ITEMS_PER_THREAD];

#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto idx = threadIdx.x + i * BLOCK_SIZE;
        dist[i] = data[idx];
        label[i] = idx;
    }

    block_sort<1, BLOCK_SIZE, VALUE_COUNT, ITEMS_PER_THREAD, ORDER>(dist, label, shm_dist,
                                                                    shm_label);

#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto idx = threadIdx.x * ITEMS_PER_THREAD + i;
        result[idx].distance = dist[i];
        result[idx].index = label[i];
    }
}

template <std::size_t ITEMS_PER_THREAD, std::size_t BLOCK_SIZE, order_t order = order_t::ascending>
void run_sort_regs_test()
{
    constexpr std::size_t BLOCK_COUNT = 1;
    constexpr std::size_t VALUE_COUNT = ITEMS_PER_THREAD * BLOCK_SIZE;

    uniform_generator gen{17};
    auto input_cpu = gen.generate(1, VALUE_COUNT);

    std::vector<knn::pair_t> output_cpu(input_cpu.size());

    cuda_array<float, 2> input{{1, VALUE_COUNT}};
    cuda_array<knn::pair_t, 2> output{{1, VALUE_COUNT}};

    cuda_stream::make_default().copy_to_gpu_async(input.view(), input_cpu.data()).sync();

    sort_regs_test<ITEMS_PER_THREAD, BLOCK_SIZE, order>
        <<<BLOCK_COUNT, BLOCK_SIZE>>>(input.view().data(), output.view().data());
    CUCH(cudaGetLastError());

    cuda_stream::make_default().copy_from_gpu_async(output_cpu.data(), output.view()).sync();

    for (std::size_t i = 1; i < VALUE_COUNT; ++i)
    {
        if (order == order_t::ascending)
        {
            REQUIRE(output_cpu[i - 1].distance <= Catch::Approx(output_cpu[i].distance));
        }
        else // descending
        {
            REQUIRE(output_cpu[i - 1].distance >= Catch::Approx(output_cpu[i].distance));
        }
    }
}

TEST_CASE("Sort array with 32 threads, 1 value per thread", "[bitonic_sort_regs]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 1;
    constexpr std::size_t BLOCK_SIZE = 32;

    run_sort_regs_test<ITEMS_PER_THREAD, BLOCK_SIZE>();
}

TEST_CASE("Sort array with 64 threads, 1 value per thread", "[bitonic_sort_regs]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 1;
    constexpr std::size_t BLOCK_SIZE = 64;

    run_sort_regs_test<ITEMS_PER_THREAD, BLOCK_SIZE>();
}

TEST_CASE("Sort array with 128 threads, 1 value per thread", "[bitonic_sort_regs]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 1;
    constexpr std::size_t BLOCK_SIZE = 128;

    run_sort_regs_test<ITEMS_PER_THREAD, BLOCK_SIZE>();
}

TEST_CASE("Sort array with 1024 threads, 1 value per thread", "[bitonic_sort_regs]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 1;
    constexpr std::size_t BLOCK_SIZE = 1024;

    run_sort_regs_test<ITEMS_PER_THREAD, BLOCK_SIZE>();
}

TEST_CASE("Sort array with 32 threads, 2 values per thread", "[bitonic_sort_regs]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 2;
    constexpr std::size_t BLOCK_SIZE = 32;

    run_sort_regs_test<ITEMS_PER_THREAD, BLOCK_SIZE>();
}

TEST_CASE("Sort array with 32 threads, 4 values per thread", "[bitonic_sort_regs]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 4;
    constexpr std::size_t BLOCK_SIZE = 32;

    run_sort_regs_test<ITEMS_PER_THREAD, BLOCK_SIZE>();
}

TEST_CASE("Sort array with 64 threads, 8 values per thread", "[bitonic_sort_regs]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 8;
    constexpr std::size_t BLOCK_SIZE = 64;

    run_sort_regs_test<ITEMS_PER_THREAD, BLOCK_SIZE>();
}

TEST_CASE("Sort array with 1024 threads, 2 values per thread", "[bitonic_sort_regs]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 2;
    constexpr std::size_t BLOCK_SIZE = 1024;

    run_sort_regs_test<ITEMS_PER_THREAD, BLOCK_SIZE>();
}

TEST_CASE("Sort array with 1024 threads, 2 values per thread in descending order",
          "[bitonic_sort_regs]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 2;
    constexpr std::size_t BLOCK_SIZE = 1024;

    run_sort_regs_test<ITEMS_PER_THREAD, BLOCK_SIZE, order_t::descending>();
}
