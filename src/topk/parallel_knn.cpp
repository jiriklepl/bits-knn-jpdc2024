#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "bits/knn.hpp"
#include "bits/knn_args.hpp"

#include "bits/distance/eigen_distance.hpp"
#include "bits/topk/parallel_knn.hpp"

void parallel_knn::initialize(const knn_args& args)
{
    if (!dist_impl_)
    {
        set_dist_impl(std::make_unique<eigen_distance>());
    }
    knn::initialize(args);
}

void parallel_knn::prepare()
{
    knn::prepare();

    dist_.clear();
    dist_.resize(query_count() * k());
}

void parallel_knn::selection()
{
#pragma omp parallel for
    for (std::size_t i = 0; i < query_count(); ++i)
    {
        auto dist = dist_impl_->matrix_cpu();
        auto dist_cmp = [](const distance_pair& lhs, const distance_pair& rhs) {
            return lhs.distance < rhs.distance;
        };
        const auto topk_begin = dist_.begin() + (i * k());
        const auto topk_end = topk_begin + k();
        const auto topk_last = topk_begin + (k() - 1);

        // create heap from the first k elements
        auto topk_it = topk_begin;
        for (std::size_t j = 0; j < k(); ++j)
        {
            *topk_it++ = distance_pair{.distance = dist(i, j), .index = static_cast<int>(j)};
        }
        std::make_heap(topk_begin, topk_end, dist_cmp);

        // add all other elements to the heap
        for (std::size_t j = k(); j < point_count(); ++j)
        {
            if (topk_begin->distance > dist(i, j))
            {
                std::pop_heap(topk_begin, topk_end, dist_cmp);
                *topk_last = distance_pair{.distance = dist(i, j), .index = static_cast<int>(j)};
                std::push_heap(topk_begin, topk_end, dist_cmp);
            }
        }

        // sort the final result
        std::sort(topk_begin, topk_end, dist_cmp);
    }
}

std::vector<distance_pair> parallel_knn::finish() { return dist_; }
