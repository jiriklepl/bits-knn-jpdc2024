#include "bits/applications/token_sampling.hpp"
#include "bits/cuda_stream.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace applications
{
namespace
{
constexpr unsigned threads = 256;

__global__ void negate_logits(const float* logits, float* scores, std::size_t count)
{
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < count;
         i += std::size_t{blockDim.x} * gridDim.x)
        scores[i] = -logits[i];
}

__global__ void sample_tokens(array_view<float, 2> logits, array_view<float, 2> scores,
                              array_view<std::int32_t, 2> indices, std::int32_t* candidate_ids,
                              float* probabilities, sampled_token* result, std::size_t k,
                              float temperature, std::uint64_t seed, std::uint64_t draw)
{
    extern __shared__ double storage[];
    auto* reduction = storage + k;
    __shared__ int invalid;
    for (std::size_t row = blockIdx.x; row < logits.size(0); row += gridDim.x)
    {
        if (threadIdx.x == 0)
            invalid = 0;
        __syncthreads();
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t j = threadIdx.x; j < k; j += blockDim.x)
        {
            const auto id = indices(row, j);
            const auto score = scores(row, j);
            const bool valid =
                id >= 0 && static_cast<std::size_t>(id) < logits.size(1) && isfinite(score);
            candidate_ids[row * k + j] = id;
            if (!valid)
                atomicExch(&invalid, 1);
            storage[j] = valid ? -static_cast<double>(score) : 0.0;
            maximum = fmax(maximum, storage[j]);
        }
        reduction[threadIdx.x] = maximum;
        __syncthreads();
        for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (threadIdx.x < stride)
                reduction[threadIdx.x] =
                    fmax(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
            __syncthreads();
        }
        maximum = reduction[0];
        // Double subtraction/division keep every finite FP32 logit and positive FP32
        // temperature representable, including opposite extrema and subnormal temperatures.
        double sum = 0;
        for (std::size_t j = threadIdx.x; j < k; j += blockDim.x)
        {
            storage[j] = exp((storage[j] - maximum) / static_cast<double>(temperature));
            sum += storage[j];
        }
        __syncthreads();
        reduction[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (threadIdx.x < stride)
                reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            __syncthreads();
        }
        for (std::size_t j = threadIdx.x; j < k; j += blockDim.x)
            probabilities[row * k + j] = invalid ? std::numeric_limits<float>::quiet_NaN()
                                                 : static_cast<float>(storage[j] / reduction[0]);
        __syncthreads();
        if (threadIdx.x == 0)
        {
            result[row] = {-1, std::numeric_limits<float>::quiet_NaN()};
            if (!invalid)
            {
                // Renormalize the stored FP32 probabilities for the CDF, including
                // their rounding error. Preserve the backend's candidate order.
                double total = 0;
                for (std::size_t j = 0; j < k; ++j)
                    total += probabilities[row * k + j];
                const double target = sampling_uniform(seed, draw, row) * total;
                double cumulative = 0;
                for (std::size_t j = 0; j < k; ++j)
                {
                    const auto probability = probabilities[row * k + j];
                    if (probability > 0)
                        result[row] = {candidate_ids[row * k + j], probability};
                    cumulative += probability;
                    if (target < cumulative)
                        break;
                }
            }
        }
        __syncthreads();
    }
}
} // namespace

token_sampling::token_sampling(std::size_t batch, std::size_t vocabulary, std::size_t k,
                               float temperature)
    : batch_(batch), vocabulary_(vocabulary), k_(k), temperature_(temperature)
{
    const auto maximum = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (batch == 0 || batch > maximum || vocabulary == 0 || vocabulary > maximum || k == 0 ||
        k > vocabulary || k > 2048 || !std::isfinite(temperature) || temperature <= 0 ||
        batch > std::numeric_limits<std::size_t>::max() / sizeof(float) / vocabulary)
        throw std::invalid_argument{
            "Sampling requires positive int32 dimensions, 0 < k <= min(vocabulary, 2048), "
            "and a positive finite temperature"};
    logits_ = cuda_array<float, 2>{{batch, vocabulary}};
    scores_ = cuda_array<float, 2>{{batch, vocabulary}};
    probabilities_ = cuda_array<float, 2>{{batch, k}};
    candidate_ids_ = cuda_array<std::int32_t, 2>{{batch, k}};
    result_ = cuda_array<sampled_token, 1>{{batch}};
}

void token_sampling::upload(std::span<const float> logits)
{
    if (logits.size() != batch_ * vocabulary_)
        throw std::invalid_argument{"Upload shape differs from the allocated logits"};
    cuda_stream::make_default().copy_to_gpu_async(logits_.view().data(), logits.data(),
                                                  logits.size());
}

void token_sampling::transform()
{
    const auto count = batch_ * vocabulary_;
    negate_logits<<<std::min<std::size_t>((count + threads - 1) / threads, 65535), threads>>>(
        logits_.view().data(), scores_.view().data(), count);
    CUCH(cudaGetLastError());
}

void token_sampling::output(array_view<float, 2> scores, array_view<std::int32_t, 2> indices,
                            std::uint64_t seed, std::uint64_t draw)
{
    if (!scores.data() || !indices.data() || scores.size(0) != batch_ ||
        indices.size(0) != batch_ || scores.size(1) != k_ || indices.size(1) != k_ ||
        scores.stride(1) != 1 || indices.stride(1) != 1 || scores.stride(0) < k_ ||
        indices.stride(0) < k_)
        throw std::invalid_argument{"Selection output shape does not match sampling"};
    sample_tokens<<<std::min<std::size_t>(batch_, 65535), threads,
                    (k_ + threads) * sizeof(double)>>>(
        logits_.view(), scores, indices, candidate_ids_.view().data(), probabilities_.view().data(),
        result_.view().data(), k_, temperature_, seed, draw);
    CUCH(cudaGetLastError());
}

void token_sampling::download(std::span<sampled_token> destination) const
{
    if (destination.size() != batch_)
        throw std::invalid_argument{"Download requires exactly one token per sequence"};
    CUCH(cudaMemcpyAsync(destination.data(), result_.view().data(), batch_ * sizeof(sampled_token),
                         cudaMemcpyDeviceToHost, cuda_stream::make_default().get()));
}

void token_sampling::download_distribution(std::span<std::int32_t> candidate_ids,
                                           std::span<float> probabilities) const
{
    if (candidate_ids.size() != batch_ * k_ || probabilities.size() != batch_ * k_)
        throw std::invalid_argument{"Distribution download requires batch x k entries"};
    auto stream = cuda_stream::make_default();
    stream.copy_from_gpu_async(candidate_ids.data(), candidate_ids_.view())
        .copy_from_gpu_async(probabilities.data(), probabilities_.view());
}
} // namespace applications
