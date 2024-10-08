#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include "bits/cuda_array.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/data_generator.hpp"
#include "bits/knn.hpp"
#include "bits/memory.cuh"

template <std::size_t ITEMS_PER_THREAD, std::size_t BLOCK_SIZE, bool LOAD_STRIPED,
          bool STORE_STRIPED>
__global__ void load_store_test(std::int32_t* input, std::int32_t* output)
{
    constexpr std::size_t VALUE_COUNT = BLOCK_SIZE * ITEMS_PER_THREAD;

    __shared__ std::int32_t shm_values[VALUE_COUNT];

    std::int32_t values[ITEMS_PER_THREAD];

    if (LOAD_STRIPED)
    {
        load_striped<BLOCK_SIZE>(values, input);
    }
    else
    {
        load_blocked<BLOCK_SIZE>(values, input, shm_values);
    }

    __syncthreads();

    if (STORE_STRIPED)
    {
        store_striped<BLOCK_SIZE>(values, output);
    }
    else
    {
        store_blocked<BLOCK_SIZE>(values, output, shm_values);
    }
}

template <std::size_t ITEMS_PER_THREAD, std::size_t BLOCK_SIZE, bool LOAD_STRIPED,
          bool STORE_STRIPED>
std::vector<std::int32_t> run_load_store_test()
{
    constexpr std::size_t BLOCK_COUNT = 1;
    constexpr std::size_t VALUE_COUNT = ITEMS_PER_THREAD * BLOCK_SIZE;

    std::vector<std::int32_t> input_cpu(VALUE_COUNT);
    std::vector<std::int32_t> output_cpu(input_cpu.size());
    for (std::size_t i = 0; i < VALUE_COUNT; ++i)
    {
        input_cpu[i] = static_cast<std::int32_t>(i);
    }

    cuda_array<std::int32_t, 2> input{{1, VALUE_COUNT}};
    cuda_array<std::int32_t, 2> output{{1, VALUE_COUNT}};

    cuda_stream::make_default().copy_to_gpu_async(input.view(), input_cpu.data()).sync();

    load_store_test<ITEMS_PER_THREAD, BLOCK_SIZE, LOAD_STRIPED, STORE_STRIPED>
        <<<BLOCK_COUNT, BLOCK_SIZE>>>(input.view().data(), output.view().data());

    cuda_stream::make_default().copy_from_gpu_async(output_cpu.data(), output.view()).sync();

    return output_cpu;
}

TEST_CASE("Load and store a block of memory using striped arrangement", "[memory]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 4;
    constexpr std::size_t BLOCK_SIZE = 32;

    const auto output = run_load_store_test<ITEMS_PER_THREAD, BLOCK_SIZE, true, true>();
    for (std::size_t i = 0; i < output.size(); ++i)
    {
        REQUIRE(output[i] == static_cast<std::int32_t>(i));
    }
}

TEST_CASE("Load a block in blocked arrangement and store in striped arrangement", "[memory]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 4;
    constexpr std::size_t BLOCK_SIZE = 32;

    const auto output = run_load_store_test<ITEMS_PER_THREAD, BLOCK_SIZE, false, true>();
    for (std::size_t i = 0; i < output.size(); ++i)
    {
        const auto stripe = i / BLOCK_SIZE;
        const auto offset = i % BLOCK_SIZE;
        REQUIRE(output[i] == static_cast<std::int32_t>(offset * ITEMS_PER_THREAD + stripe));
    }
}

TEST_CASE("Load a block in striped arrangement and store in blocked arrangement", "[memory]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 4;
    constexpr std::size_t BLOCK_SIZE = 32;

    const auto output = run_load_store_test<ITEMS_PER_THREAD, BLOCK_SIZE, true, false>();
    for (std::size_t i = 0; i < output.size(); ++i)
    {
        const auto thread = i / ITEMS_PER_THREAD;
        const auto offset = i % ITEMS_PER_THREAD;
        REQUIRE(output[i] == static_cast<std::int32_t>(thread + offset * BLOCK_SIZE));
    }
}

TEST_CASE("Load a block in blocked arrangement and store in blocked arrangement", "[memory]")
{
    constexpr std::size_t ITEMS_PER_THREAD = 4;
    constexpr std::size_t BLOCK_SIZE = 32;

    const auto output = run_load_store_test<ITEMS_PER_THREAD, BLOCK_SIZE, false, false>();
    for (std::size_t i = 0; i < output.size(); ++i)
    {
        REQUIRE(output[i] == static_cast<std::int32_t>(i));
    }
}
