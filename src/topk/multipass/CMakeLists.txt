set(MULTIPASS_SRC
    air_topk.cu
    bits_global.cu
    radik_knn.cu
)

target_sources(knn-static PRIVATE ${MULTIPASS_SRC})
