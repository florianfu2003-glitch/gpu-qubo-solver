#pragma once
#include "matrix.h"
#include "qubo_energy.h"
#include "qubo_brute_forcer.h"
#include "state_vector.h"

#include <iostream>
#include <vector>

#ifdef WITH_OPENMP
#include <omp.h> 
#endif

#define MAX_VARS_BRUTE_FORCE 63

/** 
* Naive CPU implementation of the QUBO brute forcer that calculates the energy of every possible state from scratch 
*/
template<typename iT, typename vT, typename sT, typename MatrixType>
struct NaiveCPUQUBOBruteForcer : public QUBOBruteForcer<iT, vT, sT, MatrixType>
{

    std::vector<std::vector<sT>> brute_force_optima(MatrixType const & mat) override {
        assert(mat.rows == mat.cols); // only for square matrices
        assert(mat.rows <= MAX_VARS_BRUTE_FORCE); // limit to 63 variables for brute-force
        size_t n = mat.rows;
        long long num_states = 1ULL << n; // 2^n possible states
        std::vector<std::vector<sT>> best_states;
        vT best_energy = std::numeric_limits<vT>::max();

        std::cout << "Starting naive brute-force search over " << num_states << " states for " << n << " variables.\n";
        #pragma omp parallel
        {
            std::vector<std::vector<sT>> local_best_states;
            vT local_best_energy = std::numeric_limits<vT>::max();
            #pragma omp for
            for (long long state = 0; state < num_states; state++)
            {
                sT current_state[MAX_VARS_BRUTE_FORCE];
                for (size_t i = 0; i < n; i++)
                {
                    current_state[i] = ((static_cast<size_t>(state) >> (n - 1ULL - i)) & 1ULL) ? 1 : 0;
                }
                vT energy = compute_energy(mat, current_state);
                if (energy < local_best_energy)
                {
                    local_best_energy = energy;
                    local_best_states.clear();
                    local_best_states.emplace_back(current_state, current_state + n);
                } else if (energy == local_best_energy)
                {
                    local_best_states.emplace_back(current_state, current_state + n);
                }
            }
            #pragma omp critical
            {
                if (local_best_energy < best_energy)
                {
                    best_energy = local_best_energy;
                    best_states.clear();
                    best_states = std::move(local_best_states);
                } else if (local_best_energy == best_energy)
                {
                    best_states.insert(best_states.end(), local_best_states.begin(), local_best_states.end());
                }
            }
        }
        return best_states;
    }
};

/// CPU implementation of the QUBO brute forcer that uses incremental energy updates, based on https://arxiv.org/pdf/2310.19373
template<typename iT, typename vT, typename sT, typename MatrixType>
struct CPUQUBOBruteForcer : public QUBOBruteForcer<iT, vT, sT, MatrixType> {

    std::vector<std::vector<sT>> brute_force_optima(MatrixType const & mat) override {
    assert(mat.rows == mat.cols); // only for square matrices
    assert(mat.rows <= MAX_VARS_BRUTE_FORCE); // limit variables for brute-force

    size_t n = mat.rows;
    size_t num_states = 1ULL << n; // 2^n possible states


    //find largest power of two less than or equal to max_num_threads
    size_t num_threads = 1;
    size_t num_fixed_bits = 0;

    #ifdef WITH_OPENMP
        size_t max_num_threads = omp_get_max_threads();

        while (num_threads * 2 <= max_num_threads && num_fixed_bits < n)
        {
            num_threads *= 2;
            num_fixed_bits += 1;
        }
    #endif

    std::cout << "CPU using " << num_threads << " threads\n";
    size_t num_states_per_thread = num_states >> num_fixed_bits;

    // storage for best states and best energy
    std::vector<size_t> best_states;
    std::vector<std::vector<sT>> best_states_reconstructed;
    vT best_energy = std::numeric_limits<vT>::max();

    // parallel region with fixed number of threads
    #pragma omp parallel num_threads(num_threads)
    {
        #ifdef WITH_OPENMP
        size_t thread = omp_get_thread_num();
        #else
        size_t thread = 0;
        #endif

        // each thread keeps track of its local best states and energy
        std::vector<size_t> local_best_states;
        
        // fix the first num_fixed_bits according to thread id
        size_t initial_state = thread << (n - num_fixed_bits);

        // current state and energy of the thread, each bit represents a variable
        size_t state = initial_state;
        vT energy = compute_energy(mat, initial_state);

        // best energy and state found by the thread
        vT local_best_energy = energy;
        local_best_states.push_back(initial_state);
        
        // iterate over all states assigned to this thread
        for (size_t i = 0; i < num_states_per_thread - 1; i++)
        {
            // find index of bit to flip
            size_t flip_idx = num_fixed_bits + __builtin_ctzll(static_cast<unsigned long long>(i + 1));

            // compute energy difference by flipping bit at flip_idx. Calculate the new energy for this row since its the only on affected
            vT energy_diff = compute_row_flip_energy_difference(mat, state, flip_idx);
            // flip the bit in the state
            state = state ^ (1ULL << (n - 1ULL - flip_idx));

            // update energy
            energy += energy_diff;
            //check for new best energy
            if (energy_diff > vT(0))
                continue;
            else if (energy <= local_best_energy)
            {
                if (energy < local_best_energy)
                    local_best_states.clear();

                local_best_states.push_back(state);

                if (energy < local_best_energy)
                    local_best_energy = energy;
            }
        }
        // reduce local best states and energy to global best
        #ifdef WITH_OPENMP
        #pragma omp critical
        {
            if (local_best_energy < best_energy)
            {
                best_energy = local_best_energy;
                best_states.clear();
                best_states = std::move(local_best_states);
            } else if (local_best_energy == best_energy)
            {
                best_states.insert(best_states.end(), local_best_states.begin(), local_best_states.end());
            }
        }
        #else
        best_energy = local_best_energy;
        best_states = std::move(local_best_states);
        #endif
    }

    // reconstruct best states into vectors from the size_t representation
    best_states_reconstructed = std::vector<std::vector<sT>>(best_states.size(), std::vector<sT>(n, 0));

    for (int i = 0; i < best_states.size(); i++)
    {
        auto const & state = best_states[i];
        best_states_reconstructed[i] = binary_representation_to_state_vector<sT>(state, n);
    }

    return best_states_reconstructed;
    }
};