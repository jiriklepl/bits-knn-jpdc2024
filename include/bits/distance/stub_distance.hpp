#ifndef STUB_DISTANCE_HPP_
#define STUB_DISTANCE_HPP_

#include <cstddef>
#include <string>

#include "bits/array_view.hpp"
#include "bits/distance/cuda_distance.hpp"

/** Generate random distances instead of computing the real distances for testing purposes.
 */
class stub_distance : public cuda_distance
{
public:
    /** Create random distance function with distinct value using @p seed for shuffling
     *
     * @param seed Seed used for generating distances
     */
    explicit stub_distance(std::size_t seed = 17);

    /** Create a random distance function using @p seed
     *
     * @param seed Seed for generating distances
     * @param num_unique Number of unique distances. Value 0 indicates that all distances should be
     * unique
     */
    stub_distance(std::size_t seed, std::size_t num_unique);

    /** Allocate and initialize memory for the distance computation
     *
     * @param args kNN instance
     */
    void prepare(const knn_args& args) override;

    /** Compute distances
     */
    void compute() override;

    /** Allocate memory for the distance matrix on CPU and copy it there.
     *
     * @returns distance matrix view in CPU memory space
     */
    array_view<float, 2> matrix_cpu() override;

    /** Name of this distance function.
     *
     * @returns name of this distance algorithm
     */
    std::string name() const override { return "stub-dist"; }

private:
    std::size_t seed_;
    std::size_t num_unique_;
};
#endif // STUB_DISTANCE_HPP_
