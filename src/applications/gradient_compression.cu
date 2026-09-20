#include "bits/applications/gradient_compression.hpp"
#include "bits/cuda_stream.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace applications
{
namespace
{
__global__ void magnitude_kernel(const float* gradient, float* scores, std::size_t n)
{
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n;
         i += std::size_t{blockDim.x} * gridDim.x)
        scores[i] = -fabsf(gradient[i]);
}

__global__ void gather_gradient(const float* gradient, const std::int32_t* indices,
                                gradient_entry* result, std::size_t n, std::size_t k)
{
    const auto i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (i < k)
    {
        const auto j = indices[i];
        // Keep invalid selector indices visible to host verification without an invalid read.
        if (j < 0 || static_cast<std::size_t>(j) >= n)
            result[i] = gradient_entry{-1, 0.0f};
        else
            result[i] = gradient_entry{j, gradient[j]};
    }
}
} // namespace

gradient_compression::gradient_compression(std::size_t elements, std::size_t k)
    : elements_(elements), k_(k)
{
    if (elements == 0 ||
        elements > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) || k == 0 ||
        k > elements || k > 2048)
        throw std::invalid_argument{
            "Gradient compression requires 0 < k <= min(elements, 2048) and int32 indexing"};
    gradient_ = cuda_array<float, 1>{{elements}};
    scores_ = cuda_array<float, 2>{{1, elements}};
    result_ = cuda_array<gradient_entry, 1>{{k}};
}

void gradient_compression::upload(std::span<const float> gradient)
{
    if (gradient.size() != elements_)
        throw std::invalid_argument{"Upload shape differs from the allocated gradient"};
    cuda_stream::make_default().copy_to_gpu_async(gradient_.view().data(), gradient.data(),
                                                  elements_);
}

void gradient_compression::transform()
{
    magnitude_kernel<<<std::min<std::size_t>((elements_ + 255) / 256, 65535), 256>>>(
        gradient_.view().data(), scores_.view().data(), elements_);
    CUCH(cudaGetLastError());
}

void gradient_compression::output(array_view<float, 2> scores, array_view<std::int32_t, 2> indices)
{
    if (!scores.data() || !indices.data() || scores.size(0) != 1 || indices.size(0) != 1 ||
        scores.size(1) != k_ || indices.size(1) != k_ || scores.stride(1) != 1 ||
        indices.stride(1) != 1)
        throw std::invalid_argument{"Selection output shape does not match gradient top-k"};
    // Only indices are needed: signed values must come from the original gradient.
    gather_gradient<<<(k_ + 255) / 256, 256>>>(gradient_.view().data(), indices.data(),
                                               result_.view().data(), elements_, k_);
    CUCH(cudaGetLastError());
}

void gradient_compression::download(std::span<gradient_entry> destination) const
{
    if (destination.size() != k_)
        throw std::invalid_argument{"Download requires exactly k gradient entries"};
    CUCH(cudaMemcpyAsync(destination.data(), result_.view().data(), k_ * sizeof(gradient_entry),
                         cudaMemcpyDeviceToHost, cuda_stream::make_default().get()));
}
} // namespace applications
