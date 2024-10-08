#ifndef CUDA_KNN_HPP_
#define CUDA_KNN_HPP_

#include <vector>

#include "bits/array_view.hpp"
#include "bits/cuda_event.hpp"
#include "bits/knn.hpp"

/** Base class for brute force cuda solutions with Euclidean distance
 */
class cuda_knn : public knn
{
public:
    void initialize(const knn_args& args) override;

    /** Copy the computed result to CPU memory space
     *
     * @return top k result or an empty vector if you call `no_output()`
     */
    std::vector<pair_t> finish() override;

    /** Get view of the input distances
     *
     * @returns view of the distance matrix
     */
    virtual array_view<float, 2> in_dist_gpu();

    /** Get view of the output distances
     *
     * @returns view of the top k distances
     */
    virtual array_view<float, 2> out_dist_gpu();

    /** Get view of the output labels
     *
     * @returns view of the labels of the top k elements
     */
    virtual array_view<std::int32_t, 2> out_label_gpu();

    void set_random_distances() override;

    float transfer_in_seconds() const override { return dist_impl_->transfer_seconds(); }

    float transfer_out_seconds() const override
    {
        return transfer_end_.elapsed_seconds(transfer_begin_);
    }

protected:
    cuda_array<float, 2> out_dist_gpu_;
    cuda_array<std::int32_t, 2> out_label_gpu_;
    cuda_event transfer_begin_;
    cuda_event transfer_end_;
};

#endif // CUDA_KNN_HPP_
