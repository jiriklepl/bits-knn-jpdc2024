#include <limits>
#include <stdexcept>

#include <grid_select.h>

#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/topk/singlepass/grid_select.hpp"

void grid_select::initialize(const knn_args& args)
{
    if (args.point_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        args.query_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument{"GridSelect requires point/query counts representable by int"};

    buf_.reset();
    buf_size_ = 0;
    cuda_knn::initialize(args);

    auto in_dist = in_dist_gpu();
    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();

    // Query and allocate scratch space outside repeated selection calls.
    nv::grid_select(nullptr, buf_size_, in_dist.data(), in_dist.size(0), in_dist.size(1), k(),
                    out_dist.data(), out_label.data(),
                    false, // greater (false == compute the smallest k values)
                    cuda_stream::make_default().get());
    CUCH(cudaGetLastError());
    if (buf_size_ == 0)
        throw std::runtime_error{"GridSelect returned an empty workspace requirement"};
    buf_ = make_cuda_ptr<std::byte>(buf_size_);
}

void grid_select::selection()
{
    cuda_knn::selection();

    auto in_dist = in_dist_gpu();
    auto out_dist = out_dist_gpu();
    auto out_label = out_label_gpu();

    nv::grid_select(buf_.get(), buf_size_, in_dist.data(), in_dist.size(0), in_dist.size(1), k(),
                    out_dist.data(), out_label.data(),
                    false, // greater (false == compute the smallest k values)
                    cuda_stream::make_default().get());
    CUCH(cudaGetLastError());

    cuda_stream::make_default().sync();
}
