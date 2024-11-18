#ifndef BITS_DISTANCE_TILED_DISTANCE_HPP_
#define BITS_DISTANCE_TILED_DISTANCE_HPP_

#include <string>

#include "bits/distance/cuda_distance.hpp"

/** Tiled CUDA kernel for computing distances.
 *
 * Data matrices are separated to tiles which are then loaded to shared memory.
 */
class tiled_distance : public cuda_distance
{
public:
    void compute() override;

    std::string name() const override { return "tiled-dist"; }
};

#endif // BITS_DISTANCE_TILED_DISTANCE_HPP_
