#ifndef TOPK_SINGLEPASS_DETAIL_DEFINITIONS_COMMON_HPP_
#define TOPK_SINGLEPASS_DETAIL_DEFINITIONS_COMMON_HPP_

#define TOPK_SINGLEPASS_K_VALUES 16, 32, 64, 128, 256, 512, 1024, 2048
#define TOPK_SINGLEPASS_FUSED_K_VALUES 4, 8, 16, 32, 64, 128, 256
#define TOPK_SINGLEPASS_FUSED_CACHE_K_VALUES 4, 8, 16, 32, 64, 128, 256

#define TOPK_SINGLEPASS_FAISS_BLOCK_SIZES 64, 128
#define TOPK_SINGLEPASS_FAISS_THREAD_QUEUES 2, 3, 4, 5, 6, 7, 8, 9, 10

#define TOPK_SINGLEPASS_FUSED_BLOCK_QUERY_DIMS 4, 8, 16
#define TOPK_SINGLEPASS_FUSED_QUERY_REGS 2, 4, 8
#define TOPK_SINGLEPASS_FUSED_POINTS_REGS 4

#define TOPK_SINGLEPASS_FUSED_TC_VAL_TYPES half, bfloat16, double
#define TOPK_SINGLEPASS_FUSED_TC_BLOCK_SIZES 128, 256, 512

#endif // TOPK_SINGLEPASS_DETAIL_DEFINITIONS_COMMON_HPP_
