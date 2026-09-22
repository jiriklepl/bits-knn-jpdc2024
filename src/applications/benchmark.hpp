#ifndef BITS_APPLICATIONS_BENCHMARK_HPP_
#define BITS_APPLICATIONS_BENCHMARK_HPP_

#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "bits/cuda_stream.hpp"
#include "bits/topk/multipass/air_topk.hpp"
#include "bits/topk/singlepass/bits_knn.hpp"
#include "bits/topk/singlepass/grid_select.hpp"
#include "bits/topk/singlepass/warp_select.hpp"

namespace applications::benchmark
{
template <class T>
std::vector<T> read_column(const std::string& path, std::size_t rows)
{
    if (std::endian::native != std::endian::little)
        throw std::runtime_error{"The raw column loader currently requires a little-endian host"};
    if (std::filesystem::file_size(path) != rows * sizeof(T))
        throw std::invalid_argument{"Column byte count does not match rows: " + path};
    std::vector<T> result(rows);
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(result.data()), result.size() * sizeof(T)))
        throw std::runtime_error{"Cannot read column: " + path};
    return result;
}

template <class F>
double measure(F&& operation)
{
    const auto begin = std::chrono::steady_clock::now();
    operation();
    cuda_stream::make_default().sync();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

struct backend
{
    std::unique_ptr<cuda_knn> selector;
    std::size_t degree, block_size, items;
    bool sorted;
};

inline backend make_backend(const std::string& name, std::size_t k, std::size_t degree,
                            std::optional<std::size_t> requested_items, std::size_t bits_block_size)
{
    const auto items = requested_items.value_or(name == "bits-sq" ? 4 : 7);
    if (name == "bits" || name == "bits-prefetch" || name == "bits-sq")
    {
        if (items != 1 && items != 4 && items != 7 && items != 8 && items != 13 && items != 16)
            throw std::invalid_argument{"bits items-per-thread must be 1, 4, 7, 8, 13 or 16"};
        if (bits_block_size != 128 && bits_block_size != 256 && bits_block_size != 512)
            throw std::invalid_argument{"bits block size must be 128, 256 or 512"};
    }
    if (name == "bits")
        return {std::make_unique<bits_knn>(), 1, bits_block_size, items, true};
    if (name == "bits-prefetch")
        return {std::make_unique<bits_prefetch_knn>(), 1, bits_block_size, items, true};
    if (name == "bits-sq")
        return {std::make_unique<single_query_bits>(), degree, bits_block_size, items, true};
    if (name == "air-topk")
        return {std::make_unique<air_topk>(), 1, 512, 0, false};
    if (name == "grid-select")
        // The bundled library chooses its launch configuration internally.
        return {std::make_unique<grid_select>(), 1, 0, 0, true};
    if (name == "block-select")
    {
        if (k < 32 || k > 1024 || (k & (k - 1)) != 0)
            throw std::invalid_argument{"BlockSelect supports k = 32,64,128,256,512,1024"};
        const auto queue = k == 32 ? 2u : k <= 128 ? 3u : k == 256 ? 4u : 8u;
        return {std::make_unique<block_select>(), 1, 128, queue, true};
    }
    throw std::invalid_argument{"Unknown backend: " + name};
}
} // namespace applications::benchmark
#endif
