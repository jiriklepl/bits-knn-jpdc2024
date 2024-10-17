#ifndef FUSED_TC_POLICY_HPP_
#define FUSED_TC_POLICY_HPP_

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct fused_tc_half_policy
{
    using input_t = half;
    using output_t = float;

    static constexpr std::int32_t QUERY_TILE_SIZE = 8;
    static constexpr std::int32_t POINT_TILE_SIZE = 32;
    static constexpr std::int32_t DIM_TILE_SIZE = 16;

    static __forceinline__ __host__ __device__ float to_float(output_t value) { return value; }

    static __forceinline__ __host__ __device__ float to_float(input_t value)
    {
        return __half2float(value);
    }

    static __forceinline__ __host__ __device__ input_t from_float(float value)
    {
        return __float2half(value);
    }

    static __forceinline__ __host__ __device__ output_t zero_output() { return 0.0f; }

    static __forceinline__ __host__ __device__ input_t zero_input() { return __float2half(0.0f); }
};

struct fused_tc_bfloat16_policy
{
    using input_t = __nv_bfloat16;
    using output_t = float;

    static constexpr std::int32_t QUERY_TILE_SIZE = 8;
    static constexpr std::int32_t POINT_TILE_SIZE = 32;
    static constexpr std::int32_t DIM_TILE_SIZE = 16;

    static __forceinline__ __host__ __device__ float to_float(output_t value) { return value; }

    static __forceinline__ __host__ __device__ float to_float(input_t value)
    {
        return __bfloat162float(value);
    }

    static __forceinline__ __host__ __device__ input_t from_float(float value)
    {
        return __float2bfloat16(value);
    }

    static __forceinline__ __host__ __device__ output_t zero_output() { return 0.0f; }

    static __forceinline__ __host__ __device__ input_t zero_input()
    {
        return __float2bfloat16(0.0f);
    }
};

struct fused_tc_double_policy
{
    using input_t = double;
    using output_t = double;

    static constexpr std::int32_t QUERY_TILE_SIZE = 8;
    static constexpr std::int32_t POINT_TILE_SIZE = 8;
    static constexpr std::int32_t DIM_TILE_SIZE = 4;

    static __forceinline__ __host__ __device__ float to_float(output_t /* input_t */ value)
    {
        return (float)value;
    }

    static __forceinline__ __host__ __device__ input_t from_float(float value) { return value; }

    static __forceinline__ __host__ __device__ output_t zero_output() { return 0.0; }

    static __forceinline__ __host__ __device__ input_t zero_input() { return 0.0; }
};

#endif // FUSED_TC_POLICY_HPP_
