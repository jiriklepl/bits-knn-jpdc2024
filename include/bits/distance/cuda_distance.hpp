#ifndef BITS_DISTANCE_CUDA_DISTANCE_HPP_
#define BITS_DISTANCE_CUDA_DISTANCE_HPP_

#include <string>
#include <vector>

#include "bits/array_view.hpp"
#include "bits/cuda_array.hpp"
#include "bits/cuda_event.hpp"
#include "bits/knn_args.hpp"

class cuda_distance
{
public:
    virtual ~cuda_distance() = default;

    cuda_distance() = default;

    cuda_distance(const cuda_distance&) = delete;
    cuda_distance& operator=(const cuda_distance&) = delete;

    cuda_distance(cuda_distance&&) noexcept = default;
    cuda_distance& operator=(cuda_distance&&) noexcept = default;

    /** Allocate and initialize memory for the distance computation
     *
     * @param args kNN instance
     */
    virtual void prepare(const knn_args& args);

    /** Compute distances
     */
    virtual void compute() = 0;

    /** Get the time in seconds of the last data transfer to the GPU
     *
     * @return time in seconds of the last data transfer
     */
    virtual float transfer_seconds() const { return 0; }

    /** Get pointer to the distance matrix of size `query_count * points_count`
     *
     * @returns distance matrix view in GPU memory space
     */
    virtual array_view<float, 2> matrix_gpu() const;

    /** Allocate memory for the distance matrix on CPU and copy it there.
     *
     * @returns distance matrix view in CPU memory space
     */
    virtual array_view<float, 2> matrix_cpu();

    /** Name of this distance function.
     *
     * @returns name of this distance algorithm
     */
    virtual std::string name() const = 0;

protected:
    knn_args args_{};
    // memory allocated on the GPU
    cuda_array<float, 2> points_gpu_;
    cuda_array<float, 2> queries_gpu_;
    cuda_array<float, 2> dist_gpu_;
    // memory for matrix_cpu()
    std::vector<float> dist_cpu_;
    // events used to measure data transfer time
    cuda_event transfer_begin_;
    cuda_event transfer_end_;
};

#endif // BITS_DISTANCE_CUDA_DISTANCE_HPP_
