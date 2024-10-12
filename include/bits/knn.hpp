#ifndef KNN_HPP_
#define KNN_HPP_

#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "bits/distance/cuda_distance.hpp"
#include "bits/knn_args.hpp"

struct distance_pair
{
    float distance;
    std::int32_t index;
};

class knn
{
public:
    using pair_t = distance_pair;

    knn() = default;

    knn(const knn& args) = delete;
    knn& operator=(const knn& args) = delete;

    knn(knn&&) noexcept = default;
    knn& operator=(knn&&) noexcept = default;

    virtual ~knn() = default;

    virtual void initialize(const knn_args& args);

    virtual std::string id() const = 0;

    /** Allocate memory and initialize internal structures
     */
    virtual void prepare() {}

    /** Compute the distance matrix
     */
    virtual void distances();

    /** Select the (labeled) top k values from the distance matrix
     */
    virtual void selection() {}

    /** Postprocessing step to provide a sorted output (e.g., if selection() does not sort the
     * values)
     */
    virtual void postprocessing() {}

    /** Prepare the result in a CPU accessible memory
     *
     * @return vector of distance label pairs of the top k values
     */
    virtual std::vector<distance_pair> finish() { return std::vector<distance_pair>{}; }

    /** Get duration of transferring the input from host to device.
     *
     * @return duration of transferring the input
     */
    virtual float transfer_in_seconds() const { return 0; }

    /** Get duration of transferring the output from device to host.
     *
     * @return duration of transferring the output
     */
    virtual float transfer_out_seconds() const { return 0; }

    /** Set implementation of the distance function
     *
     * @param impl Implementation of the distance function
     */
    virtual void set_dist_impl(std::unique_ptr<cuda_distance>&& impl);

    /** Dimension of all objects
     *
     * @returns dimension of all objects
     */
    inline std::size_t dim() const { return args_.dim; }

    /** Number of nearest neighbors
     *
     * @returns number of nearest neighbors
     */
    inline std::size_t k() const { return args_.k; }

    /** Number of train objects
     *
     * @returns number of train objects
     */
    inline std::size_t point_count() const { return args_.point_count; }

    /** Number of test objects
     *
     * @returns number of test objects
     */
    inline std::size_t query_count() const { return args_.query_count; }

    /** Number of threads in a block
     *
     * @returns size of a CUDA block
     */
    inline std::size_t selection_block_size() const { return args_.selection_block_size; }

    /** Get pointer to points in CPU memory space
     *
     * @returns pointer to the points matrix
     */
    inline const float* points() const { return args_.points; }

    /** Get pointer to queries in CPU memory space
     *
     * @returns pointer to the query matrix
     */
    inline const float* queries() const { return args_.queries; }

    /** Do not copy the kNN result back to CPU
     */
    inline void no_output() { no_output_ = true; }

protected:
    bool no_output_ = false;
    knn_args args_{};
    std::unique_ptr<cuda_distance> dist_impl_;
};

#endif // KNN_HPP_
