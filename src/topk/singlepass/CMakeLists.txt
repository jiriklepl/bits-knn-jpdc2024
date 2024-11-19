set(TOPK_SINGLEPASS_SRC
    bits_knn.cu
    cub_knn.cu
    fused_cache_knn.cu
    fused_knn.cu
    fused_tc_knn.cu
    grid_select.cu
    partial_bitonic_buffered.cu
    partial_bitonic.cu
    raft_warpsort.cu
    rapidsai_fused.cu
    warp_select.cu
)

target_sources(topk-singlepass-all PRIVATE ${TOPK_SINGLEPASS_SRC})
target_sources(topk-singlepass-minimal PRIVATE ${TOPK_SINGLEPASS_SRC})

target_link_libraries(topk-singlepass-all PRIVATE raft::raft)
target_link_libraries(topk-singlepass-minimal PRIVATE raft::raft)
