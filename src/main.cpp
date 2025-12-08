#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "datatypes.h"
#include "matrix_reader.h"
#include "cpu_brute_force.h"
#include "gpu_brute_force.h"
#include "timer.h"
#include "cuda_timer.h"
#include "qubo_energy.h"
#include "state_vector.h"

#ifdef WITH_CUDA
void print_gpu_info()
{
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    std::cout << "========== GPU Information ==========\n";
    std::cout << "CUDA Devices: " << devCount << "\n";

    for (int i = 0; i < devCount; i++)
    {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        std::cout << "Device " << i << ": " << p.name << "\n";
        std::cout << "  Compute Capability: " << p.major << "." << p.minor << "\n";
        std::cout << "  SM Count: " << p.multiProcessorCount << "\n";
        std::cout << "  Global Memory: " << p.totalGlobalMem / (1024 * 1024) << " MB\n";
    }
    std::cout << "=====================================\n";
}
#endif

int main()
{
    using iT = IndexType;   // unsigned int
    using vT = ValueType;   // double
    using sT = StateType;   // unsigned char

#ifdef WITH_CUDA
    print_gpu_info();
#else
    std::cout << "[INFO] CUDA NOT AVAILABLE → CPU only mode.\n";
#endif

    std::vector<std::string> matrices = {
        // block encoding (sparse)
        "../data/block_encoding_10.mtx",
        "../data/block_encoding_20.mtx",
        "../data/block_encoding_30.mtx",

        // one-hot (dense)
        "../data/one_hot_encoding_10.mtx",
        "../data/one_hot_encoding_20.mtx",
        "../data/one_hot_encoding_25.mtx",

        // maxcut (sparse)
        "../data/maxcut_g000134_qubo.mtx",
        "../data/maxcut_g000538_qubo.mtx",
        "../data/maxcut_g000750_qubo.mtx",
        "../data/maxcut_g001046_qubo.mtx",
        "../data/maxcut_g001940_qubo.mtx",
        "../data/maxcut_g002768_qubo.mtx",

        // coloring (sparse)
        "../data/coloring_g000072_qubo.mtx",
        "../data/coloring_g001055_qubo.mtx",
        "../data/coloring_g001291_qubo.mtx",
        "../data/coloring_g001511_qubo.mtx",
        "../data/coloring_g002206_qubo.mtx"
    };


    for (auto const& file : matrices)
    {
        std::cout << "\n===========================================\n";
        std::cout << "Matrix: " << file << "\n";

        // MUST USE readMatrixMarket<vT, iT>
        auto sparseMat = readMatrixMarket<vT, iT>(file);

        size_t n = sparseMat.rows;
        std::cout << n << " x " << n << ", nnz = " << sparseMat.nnz << "\n";

        // ---------------- CPU ----------------
        CPUQUBOBruteForcer<iT, vT, sT, SparseMatrix<vT, iT>> cpu_solver;

        double cpu_time = 0.0;
        std::vector<std::vector<sT>> cpu_opt;

        {
            Timer t("CPU brute force");
            cpu_opt = cpu_solver.brute_force_optima(sparseMat);
            cpu_time = t.stop();
        }

        vT cpu_energy = compute_energy(sparseMat, cpu_opt[0].data());
        std::cout << "CPU best energy = " << cpu_energy << "\n";
        std::cout << "CPU time = " << cpu_time << " ms\n";

#ifdef WITH_CUDA
        // ---------------- GPU ----------------
        GPUQUBOBruteForcer<iT, vT, sT, SparseMatrix<vT, iT>> gpu_solver;

        float gpu_time = 0.0f;
        std::vector<std::vector<sT>> gpu_opt;

        {
            CudaTimer t("GPU brute force", true);
            gpu_opt = gpu_solver.brute_force_optima(sparseMat);
            cudaDeviceSynchronize();
            gpu_time = t.elapsed();
        }

        vT gpu_energy = compute_energy(sparseMat, gpu_opt[0].data());
        std::cout << "GPU best energy = " << gpu_energy << "\n";
        std::cout << "GPU time = " << gpu_time << " ms\n";

        bool correct = (cpu_energy == gpu_energy);
        std::cout << "Correctness: " << (correct ? "MATCH ✓" : "MISMATCH ✗") << "\n";

        if (correct) {
            std::cout << "Speedup = " << cpu_time / gpu_time << "x\n";
        }
#else
        std::cout << "[GPU disabled – skipping GPU]\n";
#endif
    }

    return 0;
}
