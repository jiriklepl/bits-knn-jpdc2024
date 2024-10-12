#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "bits/array_view.hpp"
#include "bits/cuda_array.hpp"
#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/knn.hpp"
#include "bits/knn_args.hpp"

void cuda_knn::initialize(const knn_args& args)
{
    knn::initialize(args);

    out_dist_gpu_.release();
    out_label_gpu_.release();
    out_dist_gpu_ = cuda_array<float, 2>{{query_count(), k()}};
    out_label_gpu_ = cuda_array<std::int32_t, 2>{{query_count(), k()}};
}

array_view<float, 2> cuda_knn::in_dist_gpu() { return dist_impl_->matrix_gpu(); }

array_view<float, 2> cuda_knn::out_dist_gpu() { return out_dist_gpu_.view(); }

array_view<std::int32_t, 2> cuda_knn::out_label_gpu() { return out_label_gpu_.view(); }

std::vector<knn::pair_t> cuda_knn::finish()
{
    if (no_output_)
    {
        // make sure we measure something so that it is valid to query the elapsed time
        transfer_begin_.record();
        transfer_end_.record();
        transfer_end_.sync();
        return std::vector<knn::pair_t>{};
    }

    std::vector<knn::pair_t> result(query_count() * k());
    std::vector<float> dist(query_count() * k());
    std::vector<std::int32_t> label(query_count() * k());

    transfer_begin_.record();
    cuda_stream::make_default()
        .copy_from_gpu_async(dist.data(), out_dist_gpu())
        .copy_from_gpu_async(label.data(), out_label_gpu())
        .sync();
    transfer_end_.record();
    transfer_end_.sync();

    for (std::size_t i = 0; i < query_count(); ++i)
    {
        for (std::size_t j = 0; j < k(); ++j)
        {
            const auto idx = i * k() + j;
            auto& out = result[idx];
            out.distance = dist[idx];
            out.index = label[idx];
        }
    }

    return result;
}
