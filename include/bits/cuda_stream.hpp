#ifndef CUDA_STREAM_HPP_
#define CUDA_STREAM_HPP_

#include <cassert>
#include <cstddef>
#include <utility>

#include <cuda_runtime.h>

#include "bits/array_view.hpp"
#include "bits/cuch.hpp"

/** cudaStream_t wrapper which will automatically destroy the stream
 */
class cuda_stream
{
public:
    inline cuda_stream() : stream_(0) { CUCH(cudaStreamCreate(&stream_)); }

    inline explicit cuda_stream(cudaStream_t stream) : stream_(stream) {}

    inline ~cuda_stream() { release(); }

    // Noncopyable
    cuda_stream(const cuda_stream&) = delete;
    cuda_stream& operator=(const cuda_stream&) = delete;

    // Movable
    inline cuda_stream(cuda_stream&& other) noexcept : stream_(other.stream_) { other.stream_ = 0; }

    inline cuda_stream& operator=(cuda_stream&& other) noexcept
    {
        release();
        stream_ = other.stream_;
        other.stream_ = 0;
        return *this;
    }

    /** Wait for all previous commands in the stream to finish.
     */
    inline void sync() const
    {
        CUCH(cudaPeekAtLastError());
        CUCH(cudaStreamSynchronize(stream_));
    }

    /** Wait for the GPU to finish all previous commands.
     *
     * @return this
     */
    inline cuda_stream& device_sync()
    {
        CUCH(cudaDeviceSynchronize());
        return *this;
    }

    /** Destroy this stream.
     */
    inline void release()
    {
        if (stream_ != 0)
        {
            CUCH(cudaStreamDestroy(stream_));
            stream_ = 0;
        }
    }

    /** Run a kernel in this stream.
     *
     * @tparam KernelRunner callable object which runs the kernel.
     * @tparam Args types of kernel arguments.
     * @param run function which runs the kernel.
     * @param args arguments for the kernel.
     * @return this
     */
    template <typename KernelRunner, typename... Args>
    cuda_stream& run(KernelRunner&& run, Args&&... args)
    {
        run(get(), std::forward<Args>(args)...);
        return *this;
    }

    /** Copy @p count values of type `T` from CPU memory space @p src_cpu to GPU memory space
     * @p dest_gpu
     *
     * @tparam T type of the values to copy.
     * @param dest_gpu pointer to a memory in GPU memory space.
     * @param src_cpu pointer the input data in CPU memory space.
     * @param count number of values to transfer.
     * @return this
     */
    template <typename T>
    cuda_stream& copy_to_gpu_async(T* dest_gpu, const T* src_cpu, std::size_t count)
    {
        if (count <= 0)
        {
            return *this;
        }
        assert(src_cpu != nullptr);
        assert(dest_gpu != nullptr);
        CUCH(cudaMemcpyAsync(dest_gpu, src_cpu, count * sizeof(T), cudaMemcpyHostToDevice, get()));

        return *this;
    }

    /** Copy contiguous block in host memory space to a 2D memory block (i.e., not necessarily
     * continuous block) in device memory space.
     *
     * @param dest_gpu View of the memory on the GPU
     * @param src_cpu Memory on the CPU
     */
    template <typename T>
    cuda_stream& copy_to_gpu_async(array_view<T, 2> dest_gpu, const T* src_cpu)
    {
        assert(src_cpu != nullptr);
        assert(dest_gpu.data() != nullptr);
        CUCH(cudaMemcpy2DAsync(dest_gpu.data(), dest_gpu.stride(0) * sizeof(T), src_cpu,
                               dest_gpu.size(1) * sizeof(T), dest_gpu.size(1) * sizeof(T),
                               dest_gpu.size(0), cudaMemcpyHostToDevice, get()));
        return *this;
    }

    /** Copy a data from GPU to CPU.
     *
     * @tparam T type of values to transfer.
     * @param dest_cpu pointer to a memory in CPU memory space.
     * @param src_gpu pointer to the data in GPU memory space.
     * @param count number of values to transfer.
     * @return this
     */
    template <typename T>
    cuda_stream& copy_from_gpu_async(T* dest_cpu, const T* src_gpu, std::size_t count)
    {
        if (count <= 0)
        {
            return *this;
        }
        assert(dest_cpu != nullptr);
        assert(src_gpu != nullptr);
        CUCH(cudaMemcpyAsync(dest_cpu, src_gpu, count * sizeof(T), cudaMemcpyDeviceToHost, get()));

        return *this;
    }

    /** Copy a 2D matrix to CPU.
     *
     * @tparam T type of the values to transfer.
     * @param dest_cpu pointer to a memory in CPU memory space.
     * @param src_gpu view of the data in GPU memory space.
     * @return this
     */
    template <typename T>
    cuda_stream& copy_from_gpu_async(T* dest_cpu, array_view<T, 2> src_gpu)
    {
        assert(src_gpu.data() != nullptr);
        assert(dest_cpu != nullptr);
        CUCH(cudaMemcpy2DAsync(dest_cpu, src_gpu.size(1) * sizeof(T), src_gpu.data(),
                               src_gpu.stride(0) * sizeof(T), src_gpu.size(1) * sizeof(T),
                               src_gpu.size(0), cudaMemcpyDeviceToHost, get()));
        return *this;
    }

    /** Prefetch data starting at @p ptr on the first GPU
     *
     * @param ptr Pointer to unified memory
     * @param count Prefetch memory range size
     *
     * @returns this
     */
    template <typename T>
    cuda_stream& prefetch_async(T* ptr, std::size_t count)
    {
        CUCH(cudaMemPrefetchAsync(ptr, count * sizeof(T), 0, get()));
        return *this;
    }

    /** Get stream handle
     *
     * @returns stream handle
     */
    inline cudaStream_t get() const { return stream_; }

    /** Fill @p gpu_ptr with copies of @p value
     *
     * @tparam T type of the values
     * @param gpu_ptr pointer to a memory in GPU memory space.
     * @param size number of values to initialize.
     * @param value the initialized value.
     * @return this
     */
    template <typename T>
    cuda_stream& fill_async(T* gpu_ptr, std::size_t size, T value);

    /** Create an object representing the default stream.
     *
     * @return object representing the default stream.
     */
    inline static cuda_stream make_default() { return cuda_stream{0}; }

private:
    cudaStream_t stream_;
};

#endif // CUDA_STREAM_HPP_
