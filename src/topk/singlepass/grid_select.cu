#include <cassert>

#include <grid_select.h>

#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/topk/singlepass/grid_select.hpp"

void grid_select::selection()
{
    cuda_knn::selection();

    auto in_dist = in_dist_gpu();
    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();

    // COPIED FROM external/gpu_topK_benchmark/benchmark/benchmark.cu:80
    if (buf_size_ == 0)
    {
        nv::grid_select(nullptr, buf_size_, in_dist.data(), in_dist.size(0), in_dist.size(1), k(),
                        out_dist.data(), out_label.data(),
                        false, // greater (false == compute the smallest k values)
                        cuda_stream::make_default().get());

        assert(buf_size_);
        CUCH(cudaMalloc((void**)&buf_, buf_size_));
    }

    nv::grid_select(buf_, buf_size_, in_dist.data(), in_dist.size(0), in_dist.size(1), k(),
                    out_dist.data(), out_label.data(),
                    false, // greater (false == compute the smallest k values)
                    cuda_stream::make_default().get());

    cuda_stream::make_default().sync();
}
