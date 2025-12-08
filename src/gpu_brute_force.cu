#include "gpu_brute_force.h"
#include "datatypes.h"
#include "matrix.h"
#include "qubo_energy.h"
#include "state_vector.h"
#include "cuda_util.h"

#include <cuda_runtime.h>
#include <limits>
#include <vector>
#include <cassert>
#include <iostream>
#include <stdexcept>

#define MAX_VARS_BRUTE_FORCE 63

template<typename vT>
__device__ inline vT d_get_state_entry_product(std::size_t state,
                                               std::size_t n,
                                               std::size_t row,
                                               std::size_t col)
{
    std::size_t row_idx = n - 1ULL - row;
    std::size_t col_idx = n - 1ULL - col;
    std::size_t b1 = (state >> row_idx) & 1ULL;
    std::size_t b2 = (state >> col_idx) & 1ULL;
    return static_cast<vT>(b1 & b2);
}

template<typename vT>
__device__ vT d_compute_energy_dense(const vT* mat,
                                     std::size_t n,
                                     std::size_t state)
{
    vT energy = 0;
    for (std::size_t r = 0; r < n; ++r)
    {
        for (std::size_t c = r; c < n; ++c)
        {
            vT val = mat[r * n + c];
            energy += val * d_get_state_entry_product<vT>(state, n, r, c);
        }
    }
    return energy;
}

template<typename vT>
__device__ vT d_compute_row_flip_energy_difference_dense(const vT* mat,
                                                         std::size_t n,
                                                         std::size_t state,
                                                         std::size_t row)
{
    std::size_t row_idx = n - 1ULL - row;
    std::size_t bit     = (state >> row_idx) & 1ULL;
    std::size_t flipped = 1ULL - bit;
    int change          = static_cast<int>(flipped - bit);

    vT diff = 0;
    for (std::size_t c = 0; c < n; ++c)
    {
        std::size_t col_idx = n - 1ULL - c;
        diff += mat[row * n + c] *
                static_cast<vT>((state >> col_idx) & 1ULL);
    }

    diff += mat[row * n + row] * static_cast<vT>(flipped);
    diff *= static_cast<vT>(change);
    return diff;
}

template<typename vT, typename iT>
__device__ vT d_compute_energy_sparse(const vT* values,
                                      const iT* offsets,
                                      const iT* columns,
                                      std::size_t n,
                                      std::size_t state)
{
    vT energy = 0;
    for (iT row = 0; row < static_cast<iT>(n); ++row)
    {
        iT first = offsets[row];
        iT last  = offsets[row + 1];
        for (iT idx = first; idx < last; ++idx)
        {
            iT col = columns[idx];
            if (col < row) continue; // only upper triangle
            vT val = values[idx];
            energy += val * d_get_state_entry_product<vT>(state, n, row, col);
        }
    }
    return energy;
}

template<typename vT, typename iT>
__device__ vT d_compute_row_flip_energy_difference_sparse(const vT* values,
                                                          const iT* offsets,
                                                          const iT* columns,
                                                          std::size_t n,
                                                          std::size_t state,
                                                          std::size_t row)
{
    std::size_t row_idx = n - 1ULL - row;
    std::size_t bit     = (state >> row_idx) & 1ULL;
    std::size_t flipped = 1ULL - bit;
    int change          = static_cast<int>(flipped - bit);

    vT diff = 0;

    iT first    = offsets[row];
    iT last     = offsets[row + 1];
    int diagIdx = -1;

    for (iT idx = first; idx < last; ++idx)
    {
        iT col = columns[idx];
        std::size_t col_idx = n - 1ULL - static_cast<std::size_t>(col);
        diff += values[idx] *
                static_cast<vT>((state >> col_idx) & 1ULL);
        if (col == static_cast<iT>(row))
            diagIdx = idx;
    }

    if (diagIdx != -1)
        diff += values[diagIdx] * static_cast<vT>(flipped);
    diff *= static_cast<vT>(change);
    return diff;
}

template<typename vT>
__global__ void dense_bruteforce_kernel(const vT* d_mat,
                                        std::size_t n,
                                        std::size_t numFixedBits,
                                        std::size_t statesPerThread,
                                        std::size_t numThreads,
                                        std::size_t* d_bestState,
                                        vT* d_bestEnergy)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numThreads) return;

    std::size_t state = tid << (n - numFixedBits);
    vT energy = d_compute_energy_dense(d_mat, n, state);

    vT bestEnergy   = energy;
    std::size_t bestState = state;

    for (std::size_t i = 1; i < statesPerThread; ++i)
    {
        std::size_t ctz = static_cast<std::size_t>(__ffsll((long long)i)) - 1ULL;
        std::size_t flipIdx = numFixedBits + ctz;

        vT diff = d_compute_row_flip_energy_difference_dense(d_mat, n, state, flipIdx);
        state ^= (1ULL << (n - 1ULL - flipIdx));
        energy += diff;

        if (energy < bestEnergy)
        {
            bestEnergy = energy;
            bestState  = state;
        }
    }

    d_bestState[tid]  = bestState;
    d_bestEnergy[tid] = bestEnergy;
}

template<typename vT, typename iT>
__global__ void sparse_bruteforce_kernel(const vT* d_values,
                                         const iT* d_offsets,
                                         const iT* d_columns,
                                         std::size_t n,
                                         std::size_t numFixedBits,
                                         std::size_t statesPerThread,
                                         std::size_t numThreads,
                                         std::size_t* d_bestState,
                                         vT* d_bestEnergy)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numThreads) return;

    std::size_t state = tid << (n - numFixedBits);
    vT energy = d_compute_energy_sparse(d_values, d_offsets, d_columns, n, state);

    vT bestEnergy   = energy;
    std::size_t bestState = state;

    for (std::size_t i = 1; i < statesPerThread; ++i)
    {
        std::size_t ctz = static_cast<std::size_t>(__ffsll((long long)i)) - 1ULL;
        std::size_t flipIdx = numFixedBits + ctz;

        vT diff = d_compute_row_flip_energy_difference_sparse(d_values, d_offsets, d_columns, n, state, flipIdx);
        state ^= (1ULL << (n - 1ULL - flipIdx));
        energy += diff;

        if (energy < bestEnergy)
        {
            bestEnergy = energy;
            bestState  = state;
        }
    }

    d_bestState[tid]  = bestState;
    d_bestEnergy[tid] = bestEnergy;
}

template<typename iT, typename vT, typename sT, typename MatrixType>
static std::vector<std::vector<sT>>
gpu_bruteforce_impl(MatrixType const &)
{
    throw std::runtime_error("GPUQUBOBruteForcer not implemented for this matrix type.");
}

// DenseMatrix implementation
template<typename iT, typename vT, typename sT>
static std::vector<std::vector<sT>>
gpu_bruteforce_impl(DenseMatrix<vT> const & mat)
{
    assert(mat.rows == mat.cols);
    assert(mat.rows <= MAX_VARS_BRUTE_FORCE);

    std::size_t n         = mat.rows;
    std::size_t numStates = 1ULL << n;

    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, 0));

    std::size_t maxThreadsGPU =
        (std::size_t)prop.multiProcessorCount *
        (std::size_t)prop.maxThreadsPerMultiProcessor;

    std::size_t numThreads   = 1;
    std::size_t numFixedBits = 0;

    while (numThreads * 2 <= maxThreadsGPU &&
           numThreads * 2 <= numStates &&
           numFixedBits < n)
    {
        numThreads  *= 2;
        numFixedBits += 1;
    }

    std::size_t statesPerThread = numStates >> numFixedBits;

    vT* d_mat = nullptr;
    CUDA_CALL(cudaMalloc(&d_mat, sizeof(vT) * n * n));
    CUDA_CALL(cudaMemcpy(d_mat, mat.data, sizeof(vT) * n * n, cudaMemcpyHostToDevice));

    std::size_t* d_bestState = nullptr;
    vT*          d_bestEnergy = nullptr;
    CUDA_CALL(cudaMalloc(&d_bestState,  sizeof(std::size_t) * numThreads));
    CUDA_CALL(cudaMalloc(&d_bestEnergy, sizeof(vT)          * numThreads));

    int blockSize = 256;
    int gridSize  = (numThreads + blockSize - 1) / blockSize;

    dense_bruteforce_kernel<<<gridSize, blockSize>>>(
        d_mat, n, numFixedBits, statesPerThread, numThreads,
        d_bestState, d_bestEnergy);

    CUDA_CALL(cudaDeviceSynchronize());

    std::vector<std::size_t> h_bestState(numThreads);
    std::vector<vT>          h_bestEnergy(numThreads);

    CUDA_CALL(cudaMemcpy(h_bestState.data(), d_bestState,
                         sizeof(std::size_t) * numThreads,
                         cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaMemcpy(h_bestEnergy.data(), d_bestEnergy,
                         sizeof(vT) * numThreads,
                         cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_mat));
    CUDA_CALL(cudaFree(d_bestState));
    CUDA_CALL(cudaFree(d_bestEnergy));

    vT globalBest       = std::numeric_limits<vT>::max();
    std::size_t bestState = 0;

    for (std::size_t t = 0; t < numThreads; ++t)
    {
        if (h_bestEnergy[t] < globalBest)
        {
            globalBest = h_bestEnergy[t];
            bestState  = h_bestState[t];
        }
    }

    std::vector<std::vector<sT>> result;
    result.emplace_back(binary_representation_to_state_vector<sT>(bestState, n));
    return result;
}

// SparseMatrix implementation
template<typename iT, typename vT, typename sT>
static std::vector<std::vector<sT>>
gpu_bruteforce_impl(SparseMatrix<vT, iT> const & mat)
{
    assert(mat.rows == mat.cols);
    assert(mat.rows <= MAX_VARS_BRUTE_FORCE);

    std::size_t n         = mat.rows;
    std::size_t numStates = 1ULL << n;

    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, 0));

    std::size_t maxThreadsGPU =
        (std::size_t)prop.multiProcessorCount *
        (std::size_t)prop.maxThreadsPerMultiProcessor;

    std::size_t numThreads   = 1;
    std::size_t numFixedBits = 0;

    while (numThreads * 2 <= maxThreadsGPU &&
           numThreads * 2 <= numStates &&
           numFixedBits < n)
    {
        numThreads  *= 2;
        numFixedBits += 1;
    }

    std::size_t statesPerThread = numStates >> numFixedBits;

    vT* d_values = nullptr;
    iT* d_offsets = nullptr;
    iT* d_columns = nullptr;

    CUDA_CALL(cudaMalloc(&d_values,  sizeof(vT) * mat.nnz));
    CUDA_CALL(cudaMalloc(&d_offsets, sizeof(iT) * (mat.rows + 1)));
    CUDA_CALL(cudaMalloc(&d_columns, sizeof(iT) * mat.nnz));

    CUDA_CALL(cudaMemcpy(d_values,  mat.values,
                         sizeof(vT) * mat.nnz, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_offsets, mat.offsets,
                         sizeof(iT) * (mat.rows + 1), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_columns, mat.columns,
                         sizeof(iT) * mat.nnz, cudaMemcpyHostToDevice));

    std::size_t* d_bestState = nullptr;
    vT*          d_bestEnergy = nullptr;

    CUDA_CALL(cudaMalloc(&d_bestState,  sizeof(std::size_t) * numThreads));
    CUDA_CALL(cudaMalloc(&d_bestEnergy, sizeof(vT)          * numThreads));

    int blockSize = 256;
    int gridSize  = (numThreads + blockSize - 1) / blockSize;

    sparse_bruteforce_kernel<<<gridSize, blockSize>>>(
        d_values, d_offsets, d_columns,
        n, numFixedBits, statesPerThread, numThreads,
        d_bestState, d_bestEnergy);

    CUDA_CALL(cudaDeviceSynchronize());

    std::vector<std::size_t> h_bestState(numThreads);
    std::vector<vT>          h_bestEnergy(numThreads);

    CUDA_CALL(cudaMemcpy(h_bestState.data(), d_bestState,
                         sizeof(std::size_t) * numThreads, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaMemcpy(h_bestEnergy.data(), d_bestEnergy,
                         sizeof(vT) * numThreads,
                         cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_values));
    CUDA_CALL(cudaFree(d_offsets));
    CUDA_CALL(cudaFree(d_columns));
    CUDA_CALL(cudaFree(d_bestState));
    CUDA_CALL(cudaFree(d_bestEnergy));

    vT globalBest       = std::numeric_limits<vT>::max();
    std::size_t bestState = 0;

    for (std::size_t t = 0; t < numThreads; ++t)
    {
        if (h_bestEnergy[t] < globalBest)
        {
            globalBest = h_bestEnergy[t];
            bestState  = h_bestState[t];
        }
    }

std::vector<std::vector<sT>> result;
for (std::size_t t = 0; t < numThreads; ++t)
{
    if (h_bestEnergy[t] == globalBest)
    {
        result.emplace_back(
            binary_representation_to_state_vector<sT>(h_bestState[t], n)
        );
    }
}
return result;

}

template<typename iT, typename vT, typename sT, typename MatrixType>
std::vector<std::vector<sT>>
GPUQUBOBruteForcer<iT, vT, sT, MatrixType>::brute_force_optima(MatrixType const & mat)
{
    return gpu_bruteforce_impl<iT, vT, sT>(mat);
}

template struct GPUQUBOBruteForcer<IndexType, ValueType, StateType, DenseMatrix<ValueType>>;
template struct GPUQUBOBruteForcer<IndexType, ValueType, StateType, SparseMatrix<ValueType, IndexType>>;

