#include <cstddef>

#include "bits/cuda_stream.hpp"

namespace
{

template <typename T, std::size_t VALUES_PER_THREAD>
__global__ void fill_kernel(T* values, std::size_t size, T value)
{
    const auto idx = (blockIdx.x * blockDim.x + threadIdx.x) * VALUES_PER_THREAD;

#pragma unroll VALUES_PER_THREAD
    for (std::size_t i = 0; i < VALUES_PER_THREAD; ++i)
    {
        if (idx + i >= size)
        {
            return;
        }

        values[idx + i] = value;
    }
}

} // namespace

template <>
cuda_stream& cuda_stream::fill_async<float>(float* gpu_ptr, std::size_t size, float value)
{
    constexpr std::size_t BLOCK_SIZE = 512;
    constexpr std::size_t VALUES_PER_THREAD = 1;
    const auto thread_count = (size + VALUES_PER_THREAD - 1) / VALUES_PER_THREAD;
    const auto block_count = (thread_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fill_kernel<float, VALUES_PER_THREAD>
        <<<block_count, BLOCK_SIZE, 0, get()>>>(gpu_ptr, size, value);

    return *this;
}
