#ifndef CUDA_ALLOCATORS_HPP_
#define CUDA_ALLOCATORS_HPP_

#include <cstddef>

#include <cuda_runtime.h>

#include "bits/cuch.hpp"

/** Allocator which allocates pinned memory (using cudaHostAlloc)
 */
template <typename T>
class pinned_memory_allocator
{
public:
    using value_type = T;

    pinned_memory_allocator() noexcept = default;

    template <typename U>
    pinned_memory_allocator(const pinned_memory_allocator<U>&) noexcept
    {
    }

    value_type* allocate(std::size_t n)
    {
        value_type* ptr = nullptr;
        CUCH(cudaHostAlloc(&ptr, n * sizeof(value_type), cudaHostAllocPortable));
        return ptr;
    }

    void deallocate(value_type* p, std::size_t) { CUCH(cudaFreeHost(p)); }

    template <typename U>
    bool operator==(const pinned_memory_allocator<U>&) const noexcept
    {
        return true;
    }

    template <typename U>
    bool operator!=(const pinned_memory_allocator<U>& other) const noexcept
    {
        return !(*this == other);
    }
};

/** Allocators which allocates memory using cudaHostAllocMapped.
 */
template <typename T>
class mapped_memory_allocator
{
public:
    using value_type = T;

    mapped_memory_allocator() noexcept = default;

    template <typename U>
    mapped_memory_allocator(const mapped_memory_allocator<U>&) noexcept
    {
    }

    value_type* allocate(std::size_t n)
    {
        value_type* ptr = nullptr;
        CUCH(cudaHostAlloc(&ptr, n * sizeof(value_type), cudaHostAllocMapped));
        return ptr;
    }

    void deallocate(value_type* p, std::size_t) { CUCH(cudaFreeHost(p)); }

    template <typename U>
    bool operator==(const mapped_memory_allocator<U>&) const noexcept
    {
        return true;
    }

    template <typename U>
    bool operator!=(const mapped_memory_allocator<U>& other) const noexcept
    {
        return !(*this == other);
    }
};

/** Allocators which allocates memory using cudaMallocManaged.
 * Such memory is accessible from both host and device.
 */
template <typename T>
class unified_memory_allocator
{
public:
    using value_type = T;

    unified_memory_allocator() noexcept = default;

    template <typename U>
    unified_memory_allocator(const unified_memory_allocator<U>&) noexcept
    {
    }

    value_type* allocate(std::size_t n)
    {
        value_type* ptr = nullptr;
        CUCH(cudaMallocManaged(&ptr, n * sizeof(value_type)));
        return ptr;
    }

    void deallocate(value_type* p, std::size_t) { CUCH(cudaFree(p)); }

    template <typename U>
    bool operator==(const unified_memory_allocator<U>&) const noexcept
    {
        return true;
    }

    template <typename U>
    bool operator!=(const pinned_memory_allocator<U>& other) const noexcept
    {
        return !(*this == other);
    }
};

#endif // CUDA_ALLOCATORS_HPP_
