#ifndef BITONIC_SORT_LAYOUT_HPP_
#define BITONIC_SORT_LAYOUT_HPP_

#include <cstdint>

#include "bits/knn.hpp"

#include "bits/memory.cuh"

// array of structures (distance, label pairs)
struct aos_layout
{
    knn::pair_t* values;

    inline __host__ __device__ aos_layout offset(std::size_t i) const
    {
        return aos_layout{values + i};
    }

    inline __host__ __device__ float dist(std::size_t i) const { return values[i].distance; }

    inline __host__ __device__ float& dist(std::size_t i) { return values[i].distance; }

    inline __host__ __device__ std::int32_t label(std::size_t i) const { return values[i].index; }

    inline __host__ __device__ std::int32_t& label(std::size_t i) { return values[i].index; }

    inline __host__ __device__ void swap(std::size_t i, std::size_t j)
    {
        swap_values(values[i], values[j]);
    }
};

// structure of arrays
struct soa_layout
{
    float* distances;
    std::int32_t* labels;

    inline __host__ __device__ soa_layout offset(std::size_t i) const
    {
        return soa_layout{distances + i, labels + i};
    }

    inline __host__ __device__ float dist(std::size_t i) const { return distances[i]; }

    inline __host__ __device__ float& dist(std::size_t i) { return distances[i]; }

    inline __host__ __device__ std::int32_t label(std::size_t i) const { return labels[i]; }

    inline __host__ __device__ std::int32_t& label(std::size_t i) { return labels[i]; }

    inline __host__ __device__ void swap(std::size_t i, std::size_t j)
    {
        swap_values(distances[i], distances[j]);
        swap_values(labels[i], labels[j]);
    }
};

#endif // BITONIC_SORT_LAYOUT_HPP_
