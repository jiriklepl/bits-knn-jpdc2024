#ifndef CUDA_EVENT_HPP_
#define CUDA_EVENT_HPP_

#include <cuda_runtime.h>

#include "bits/cuch.hpp"

/** Wrapper for `cudaEvent_t`
 */
class cuda_event
{
public:
    cuda_event() : event_(nullptr), is_initialized_(true) { CUCH(cudaEventCreate(&event_)); }

    ~cuda_event() { destroy(); }

    // non-copyable
    cuda_event(const cuda_event&) = delete;
    cuda_event& operator=(const cuda_event&) = delete;

    // movable
    cuda_event(cuda_event&& other) noexcept
        : event_(other.event_), is_initialized_(other.is_initialized_)
    {
        other.is_initialized_ = false;
    }

    cuda_event& operator=(cuda_event&& other) noexcept
    {
        destroy();

        is_initialized_ = other.is_initialized_;
        event_ = other.event_;
        other.is_initialized_ = false;

        return *this;
    }

    /** Destroy the event.
     */
    void destroy()
    {
        if (is_initialized_)
        {
            is_initialized_ = false;
            CUCH(cudaEventDestroy(event_));
        }
    }

    /** Wait for the event.
     */
    void sync() { CUCH(cudaEventSynchronize(event_)); }

    /** Record the event in the default stream.
     */
    void record() { CUCH(cudaEventRecord(event_)); }

    /** Find the elapsed time in seconds between this event and @p other
     *
     * @param other other event
     * @return elapsed time in seconds between this and @p other
     */
    float elapsed_seconds(const cuda_event& other) const
    {
        float ms = 0;
        CUCH(cudaEventElapsedTime(&ms, other.event_, event_));
        return ms / 1e3f;
    }

private:
    cudaEvent_t event_;
    bool is_initialized_;
};

#endif // CUDA_EVENT_HPP_
