#include "bits/applications/database_topn.hpp"
#include "bits/cuda_stream.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace applications
{
namespace
{
__global__ void score_kernel(const float* price, const float* discount, float* scores,
                             std::size_t n)
{
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n;
         i += std::size_t{blockDim.x} * gridDim.x)
        scores[i] = -__fmul_rn(price[i], __fsub_rn(1.0f, discount[i]));
}

__global__ void sort_selected(float* scores, std::int32_t* indices, unsigned k, unsigned capacity)
{
    extern __shared__ float storage[];
    auto* labels = reinterpret_cast<std::int32_t*>(storage + capacity);
    for (unsigned i = threadIdx.x; i < capacity; i += blockDim.x)
    {
        storage[i] = i < k ? scores[i] : std::numeric_limits<float>::infinity();
        labels[i] = i < k ? indices[i] : -1;
    }
    __syncthreads();
    for (unsigned width = 2; width <= capacity; width *= 2)
    {
        for (unsigned stride = width / 2; stride > 0; stride /= 2)
        {
            for (unsigned i = threadIdx.x; i < capacity; i += blockDim.x)
            {
                const auto j = i ^ stride;
                if (j > i && ((i & width) == 0 ? storage[i] > storage[j] : storage[i] < storage[j]))
                {
                    const auto score = storage[i];
                    storage[i] = storage[j];
                    storage[j] = score;
                    const auto label = labels[i];
                    labels[i] = labels[j];
                    labels[j] = label;
                }
            }
            __syncthreads();
        }
    }
    for (unsigned i = threadIdx.x; i < k; i += blockDim.x)
    {
        scores[i] = storage[i];
        indices[i] = labels[i];
    }
}

__global__ void gather_rows(const std::uint64_t* row_ids, const float* payload, const float* scores,
                            const std::int32_t* indices, database_row* result, std::size_t n,
                            std::size_t k)
{
    const auto i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (i < k)
    {
        const auto j = indices[i];
        // Surface an invalid selector result to host verification without an invalid read.
        if (j < 0 || static_cast<std::size_t>(j) >= n)
            result[i] = database_row{0, 0, 0, -1};
        else
            result[i] = database_row{row_ids[j], -scores[i], payload[j], j};
    }
}
} // namespace

database_topn::database_topn(std::size_t rows, std::size_t k) : rows_(rows), k_(k)
{
    if (rows == 0 || rows > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        k == 0 || k > rows || k > 2048)
        throw std::invalid_argument{
            "Database top-N requires 0 < k <= min(rows, 2048) and int32 row indexing"};
    price_ = cuda_array<float, 1>{{rows}};
    discount_ = cuda_array<float, 1>{{rows}};
    payload_ = cuda_array<float, 1>{{rows}};
    row_id_ = cuda_array<std::uint64_t, 1>{{rows}};
    scores_ = cuda_array<float, 2>{{1, rows}};
    result_ = cuda_array<database_row, 1>{{k}};
}

void database_topn::upload(const database_columns& columns)
{
    // Value validation is a host preparation step; callers invoke validate_columns once.
    if (columns.size() != rows_ || columns.discount.size() != rows_ ||
        columns.payload.size() != rows_ || columns.row_id.size() != rows_)
        throw std::invalid_argument{"Upload shape differs from the allocated table"};
    auto stream = cuda_stream::make_default();
    stream.copy_to_gpu_async(price_.view().data(), columns.price.data(), rows_)
        .copy_to_gpu_async(discount_.view().data(), columns.discount.data(), rows_)
        .copy_to_gpu_async(payload_.view().data(), columns.payload.data(), rows_)
        .copy_to_gpu_async(row_id_.view().data(), columns.row_id.data(), rows_);
}

void database_topn::transform()
{
    score_kernel<<<std::min<std::size_t>((rows_ + 255) / 256, 65535), 256>>>(
        price_.view().data(), discount_.view().data(), scores_.view().data(), rows_);
    CUCH(cudaGetLastError());
}

void database_topn::output(array_view<float, 2> scores, array_view<std::int32_t, 2> indices,
                           bool already_sorted)
{
    if (!scores.data() || !indices.data() || scores.size(0) != 1 || indices.size(0) != 1 ||
        scores.size(1) != k_ || indices.size(1) != k_ || scores.stride(1) != 1 ||
        indices.stride(1) != 1)
        throw std::invalid_argument{"Selection output shape does not match top-N"};
    if (!already_sorted)
    {
        unsigned capacity = 1;
        while (capacity < k_)
            capacity *= 2;
        sort_selected<<<1, 256, capacity * (sizeof(float) + sizeof(std::int32_t))>>>(
            scores.data(), indices.data(), k_, capacity);
        CUCH(cudaGetLastError());
    }
    gather_rows<<<(k_ + 255) / 256, 256>>>(row_id_.view().data(), payload_.view().data(),
                                           scores.data(), indices.data(), result_.view().data(),
                                           rows_, k_);
    CUCH(cudaGetLastError());
}

void database_topn::download(std::span<database_row> destination) const
{
    if (destination.size() != k_)
        throw std::invalid_argument{"Download requires exactly k rows"};
    CUCH(cudaMemcpyAsync(destination.data(), result_.view().data(), k_ * sizeof(database_row),
                         cudaMemcpyDeviceToHost, cuda_stream::make_default().get()));
}
} // namespace applications
