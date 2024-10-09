set(TOPK_SINGLEPASS_SRC
    bits_knn.cu
    cub_knn.cu
    fused_cache_knn.cu
    fused_knn.cu
    fused_tc_knn.cu
    grid_select.cu
    partial_bitonic_buffered.cu
    partial_bitonic.cu
    rapidsai_fused.cu
    warp_select.cu
)

target_include_directories(topk-singlepass PRIVATE ${CMAKE_SOURCE_DIR}/external/gpu_topK_benchmark/include/)
target_link_directories(topk-singlepass PUBLIC ${CMAKE_SOURCE_DIR}/external/gpu_topK_benchmark/third_party/)

target_sources(topk-singlepass PRIVATE ${TOPK_SINGLEPASS_SRC})
target_link_libraries(topk-singlepass PRIVATE raft::raft gridselect)
