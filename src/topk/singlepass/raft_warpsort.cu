#include <cstddef>
#include <cstdint>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/detail/select_warpsort.cuh>

#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/topk/singlepass/raft_warpsort.hpp"

void raft_warpsort::selection()
{
    cuda_knn::selection();

    auto in_dist = in_dist_gpu();
    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();

    raft::resources resources;
    raft::resource::set_cuda_stream(resources,
                                    rmm::cuda_stream_view{cuda_stream::make_default().get()});

    raft::matrix::detail::select::warpsort::select_k<
        float, std::int32_t, raft::matrix::detail::select::dense_layout<std::int32_t>>(
        resources, in_dist.data(), static_cast<std::int32_t*>(nullptr), in_dist.size(0),
        in_dist.size(1), k(), out_dist.data(), out_label.data(),
        true,                                 // select_min
        static_cast<std::int32_t*>(nullptr)); // not used in this case

    cuda_stream::make_default().sync();
}
