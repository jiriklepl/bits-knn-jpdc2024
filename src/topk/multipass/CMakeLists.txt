set(MULTIPASS_SRC
    air_topk.cu
    radik_knn.cu
)

target_sources(topk-multipass-all PRIVATE ${MULTIPASS_SRC})
target_sources(topk-multipass-minimal PRIVATE ${MULTIPASS_SRC})

target_link_libraries(topk-multipass-all PRIVATE raft::raft)
target_link_libraries(topk-multipass-minimal PRIVATE raft::raft)
