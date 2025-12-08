#pragma once
#include "cuda_util.h"
#include <vector>
#include <string>

struct gpu_info {
    int id = -1;
    std::string name;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int maxThreadsPerBlock;
    int major;
    int minor;
    int multiProcessorCount;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int regsPerThread;
    int maxThreads;
    int blocks;
};

inline std::vector<gpu_info> get_gpu_info() {
	int num_gpus;
	CUDA_CALL(cudaGetDeviceCount(&num_gpus));
    std::vector<gpu_info> gpus(num_gpus);
	for (int i = 0; i < num_gpus; i++) {
		cudaDeviceProp p;
		CUDA_CALL(cudaGetDeviceProperties(&p, i));
        gpu_info gpu;
        gpu.id = i;
        gpu.name = p.name;
        gpu.totalGlobalMem = p.totalGlobalMem;
        gpu.sharedMemPerBlock = p.sharedMemPerBlock;
        gpu.regsPerBlock = p.regsPerBlock;
        gpu.warpSize = p.warpSize;
        gpu.maxThreadsPerBlock = p.maxThreadsPerBlock;
        gpu.major = p.major;
        gpu.minor = p.minor;
        gpu.multiProcessorCount = p.multiProcessorCount;
        gpu.l2CacheSize = p.l2CacheSize;
        gpu.persistingL2CacheMaxSize = p.persistingL2CacheMaxSize;
        gpu.maxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
        gpu.sharedMemPerMultiprocessor = p.sharedMemPerMultiprocessor;
        gpu.regsPerMultiprocessor = p.regsPerMultiprocessor;
        gpu.regsPerThread = gpu.regsPerMultiprocessor / gpu.maxThreadsPerBlock;
        gpu.maxThreads = gpu.maxThreadsPerMultiProcessor * gpu.multiProcessorCount;
        gpu.blocks = (gpu.maxThreadsPerMultiProcessor / gpu.maxThreadsPerBlock) * gpu.multiProcessorCount;
        gpus[i] = gpu;
	}
    return gpus;
}

inline void print_gpu_info(gpu_info const &  gpu) {
    std::cout << "#" << gpu.id << ": " << gpu.name
        << "\n\tSMs: " << gpu.multiProcessorCount
        << "\n\tshadermodel: " << gpu.major << "." << gpu.minor
        << "\n\tglobal mem: " << gpu.totalGlobalMem
        << "\n\tmaxThreadsPerSM: " << gpu.maxThreadsPerMultiProcessor
        << "\n\tthreadsPerBlock: " << gpu.maxThreadsPerBlock
        << "\n\tregsPerSM: " << gpu.regsPerMultiprocessor
        << "\n\tmaxThreads: " << gpu.maxThreads
        << "\n\tregsPerThreadWith1024Threads: " << gpu.regsPerThread
        << "\n\tmaxBlocksWith1024Threads: " << gpu.blocks
        << "\n\tsharedMemPerSM: "<< gpu.sharedMemPerMultiprocessor
        << "\n\tsharedMemPerBlock: " << gpu.sharedMemPerBlock
        << "\n";
}
inline void print_gpu_info(std::vector<gpu_info> const &  gpus) {
    for (auto const &  gpu : gpus)
        print_gpu_info(gpu);
}

template<typename T>
inline std::vector<T> copy_to_cpu(T const * data, int n) {
    std::vector<T> cpu(n);
    CUDA_CALL(cudaMemcpy(cpu.data(), data, sizeof(T) * n, cudaMemcpyDeviceToHost));
    return cpu;
}