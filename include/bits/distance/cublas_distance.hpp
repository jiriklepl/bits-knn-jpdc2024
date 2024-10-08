#ifndef CUBLAS_DISTANCE_HPP_
#define CUBLAS_DISTANCE_HPP_

#include <string>

#include <cublas_v2.h>

#include "bits/cuda_ptr.hpp"
#include "bits/distance/cuda_distance.hpp"

/** This class uses GEMM routines from cuBLAS to compute the distance
 */
class cublas_distance : public cuda_distance
{
public:
    cublas_distance();
    ~cublas_distance() override;

    void prepare(const knn_args& args) override;
    void compute() override;

    inline std::string name() const override { return "cublas-dist"; }

    /** Get pointer to the computed norms of database vectors in GPU memory space
     *
     * @return computed norms for each database vector in GPU memory space
     */
    inline float* lengths_gpu() const { return lengths_.get(); }

    /** Should this function finish distance computation by adding database vector norms to the
     * distance matrix
     *
     * @param val if true, this function will add vector norms to the distance matrix
     */
    inline void include_postprocessing(bool val) { postprocessing_ = val; }

private:
    bool postprocessing_ = true;
    cublasHandle_t handle_;
    cuda_ptr<float> lengths_;
};

#endif // CUBLAS_DISTANCE_HPP_
