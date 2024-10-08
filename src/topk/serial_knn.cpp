#include <algorithm>
#include <cstddef>
#include <vector>

#include "bits/knn.hpp"

#include "bits/topk/serial_knn.hpp"

void serial_knn::prepare()
{
    knn::prepare();

    dist_.clear();
    dist_.reserve(point_count() * query_count());
}

void serial_knn::distances()
{
    knn::distances();

    auto dist = dist_impl_->matrix_cpu();

    dist_.clear();
    for (std::size_t i = 0; i < query_count(); ++i)
    {
        for (std::size_t j = 0; j < point_count(); ++j)
        {
            auto& pair = dist_.emplace_back();
            pair.distance = dist(i, j);
            pair.index = j;
        }
    }
}

void serial_knn::selection()
{
    for (std::size_t i = 0; i < query_count(); ++i)
    {
        const auto begin = dist_.begin() + i * point_count();
        const auto end = begin + point_count();
        std::sort(begin, end, [](auto&& lhs, auto&& rhs) { return lhs.distance < rhs.distance; });
    }
}

std::vector<distance_pair> serial_knn::finish()
{
    std::vector<distance_pair> result(query_count() * k());
    for (std::size_t i = 0; i < query_count(); ++i)
    {
        const auto begin = dist_.begin() + i * point_count();
        std::copy(begin, begin + k(), result.begin() + k() * i);
    }
    return result;
}
