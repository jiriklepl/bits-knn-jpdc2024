set(DIST_ALL_SRC
    baseline_distance.cu
    eigen_distance.cpp
    cublas_distance.cu
    cuda_distance.cpp
    magma_distance.cu
    stub_distance.cpp
    tiled_distance.cu
    cutlass_distance.cu
)

set(DIST_MINIMAL_SRC
    baseline_distance.cu
    cublas_distance.cu
    cuda_distance.cpp
    magma_distance.cu
    tiled_distance.cu
)

target_sources(dist-all PRIVATE ${DIST_ALL_SRC})
target_sources(dist-minimal PRIVATE ${DIST_MINIMAL_SRC})
