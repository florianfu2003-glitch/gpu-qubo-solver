#pragma once
#include <cuda_runtime_api.h>
#include <iostream>
#include <cassert>

#define divRND(a, b) ((a+b-1)/(b))

#define CUDA_CALL(code) check_cuda_error((code), __FILE__, __LINE__)

inline bool check_cuda_error(cudaError_t code, const char* file, int line
) {
#ifdef DEBUG // check error code again after synchronizing in DEBUG mode
    if (code == cudaSuccess) {
        return check_cuda_error(cudaDeviceSynchronize(), file, line);
    }
#endif // DEBUG

    if (code != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorName(code) << " @ " << file << ":" << line << "\n";
        assert(false);
        return false;
    }
    return true;
}