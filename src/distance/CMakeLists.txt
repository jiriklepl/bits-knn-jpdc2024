set(DIST_SRC
    baseline_distance.cu
    eigen_distance.cpp
    cublas_distance.cu
    cuda_distance.cpp
    magma_distance.cu
    stub_distance.cpp
    tiled_distance.cu
    cutlass_distance.cu
)

target_sources(knn-static PRIVATE ${DIST_SRC})
