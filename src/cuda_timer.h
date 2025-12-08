#pragma once
#include <cuda_runtime.h>
#include <string>
#include <iostream>

class CudaTimer {
public:
    cudaEvent_t begin, end;
    bool started = false;
    std::string name;

    CudaTimer(const std::string& n, bool blocking = false) : name(n) {
        unsigned flags = blocking ? cudaEventBlockingSync : cudaEventDefault;

        cudaEventCreateWithFlags(&begin, flags);
        cudaEventCreateWithFlags(&end, flags);

        // Record start immediately
        cudaEventRecord(begin, 0);
        started = true;
    }

    // Stop timer
    void stop() {
        if (started) {
            cudaEventRecord(end, 0);
            started = false;
        }
    }

    // Return elapsed ms
    float elapsed() {
        stop();
        cudaEventSynchronize(end);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, begin, end);
        return ms;
    }

    ~CudaTimer() {
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }
};
