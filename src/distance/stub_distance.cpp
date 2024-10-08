#include <algorithm>
#include <cstddef>
#include <random>

#include "bits/array_view.hpp"
#include "bits/cuda_array.hpp"
#include "bits/cuda_stream.hpp"

#include "bits/knn_args.hpp"

#include "bits/distance/stub_distance.hpp"

stub_distance::stub_distance(std::size_t seed) : seed_(seed), num_unique_(0) {}

stub_distance::stub_distance(std::size_t seed, std::size_t num_unique)
    : seed_(seed), num_unique_(num_unique)
{
}

void stub_distance::prepare(const knn_args& args)
{
    args_ = args;

    dist_gpu_ = cuda_array<float, 2>{{args.query_count, args.point_count}};

    dist_cpu_.resize(dist_gpu_.view().size());
    for (std::size_t i = 0; i < args_.query_count; ++i)
    {
        const float step = 1.0f / 32.0f;
        const auto stride =
            num_unique_ == 0 ? 1 : (args_.point_count + num_unique_ - 1) / num_unique_;

        float value = 0.0f;
        for (std::size_t j = 0; j < args_.point_count; j += stride)
        {
            const auto end = std::min<std::size_t>(j + stride, args_.point_count);
            for (std::size_t k = j; k < end; ++k)
            {
                dist_cpu_[i * args_.point_count + k] = value;
            }
            value += step;
        }
    }
}

void stub_distance::compute()
{
    std::default_random_engine eng{seed_};
    for (std::size_t i = 0; i < args_.query_count; ++i)
    {
        const auto begin = dist_cpu_.begin() + i * args_.point_count;
        const auto end = begin + args_.point_count;
        std::shuffle(begin, end, eng);
    }

    cuda_stream::make_default().copy_to_gpu_async(dist_gpu_.view(), dist_cpu_.data()).sync();
}

array_view<float, 2> stub_distance::matrix_cpu()
{
    return array_view<float, 2>{
        dist_cpu_.data(), {args_.query_count, args_.point_count}, {args_.point_count, 1}};
}
