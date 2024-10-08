#include <catch2/catch_test_macros.hpp>

#include "bits/cuda_array.hpp"
#include "bits/cuda_stream.hpp"

#include "bits/transpose.cuh"

template <std::size_t ITEMS_PER_THREAD>
__global__ void test_transpose_kernel(std::int32_t* output)
{
    std::int32_t values[ITEMS_PER_THREAD];

#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        values[i] = threadIdx.x * ITEMS_PER_THREAD + i;
    }

    transpose_warp(values);

#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        output[threadIdx.x * ITEMS_PER_THREAD + i] = values[i];
    }
}

TEST_CASE("Transpose register array in a warp", "[transpose]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 4;
    constexpr std::size_t THREAD_COUNT = 32;

    cuda_array<std::int32_t, 1> output_gpu{{ITEMS_PER_THREAD * THREAD_COUNT}};
    std::vector<std::int32_t> output(output_gpu.view().size());

    test_transpose_kernel<ITEMS_PER_THREAD><<<1, THREAD_COUNT>>>(output_gpu.view().data());

    cuda_stream::make_default()
        .copy_from_gpu_async(output.data(), output_gpu.view().data(), output_gpu.view().size())
        .sync();

    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        for (std::size_t j = 0; j < THREAD_COUNT; ++j)
        {
            const auto idx = j * ITEMS_PER_THREAD + i;
            REQUIRE(output[idx] == i * THREAD_COUNT + j);
        }
    }
}
