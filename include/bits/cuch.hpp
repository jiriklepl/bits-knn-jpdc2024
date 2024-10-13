#ifndef CUCH_HPP_
#define CUCH_HPP_

#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

#define CUCH(status) handle_cuda_error((status), __FILE__, __LINE__, #status)

inline void handle_cuda_error(cudaError_t error, const char* path, int line, const char* msg)
{
    if (error != cudaSuccess)
    {
        std::cerr << path << ":" << line << " " << cudaGetErrorName(error) << ": "
                  << cudaGetErrorString(error) << "\n\t" << msg << '\n';
        throw std::runtime_error{"CUDA error."};
    }
}


#endif // CUCH_HPP_
