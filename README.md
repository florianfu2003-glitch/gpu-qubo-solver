# GPU-Accelerated QUBO Brute-Force Solver  
### CUDA • C++ • HPC • Dense & Sparse Matrices • Gray-Code Incremental Update

This project implements a **massively parallel brute-force solver** for  
**QUBO (Quadratic Unconstrained Binary Optimization)** problems using both  
CPU-based and **GPU-accelerated** approaches.

The focus is on **incremental energy evaluation** via **Gray-code traversal**,  
allowing high-performance enumeration of all \(2^n\) binary states for  
problems up to **63 variables**.

This work was completed as part of the  
_Programmierung Massiv-Paralleler Prozessoren (PMPP)_ course at  
**TU Darmstadt**, and further extended into a standalone research-grade system.

---

## Key Features

### GPU Acceleration

- Custom CUDA kernels for dense and sparse (CSR) QUBO matrices  
- Efficient memory access patterns  
- Fully device-side Gray-code bit-flip traversal  
- Incremental energy update reduces cost from `O(n²)` to `O(n)` per step  

### CPU Implementations

- Naive brute-force solver (full recomputation of the energy)
- Optimized CPU solver using incremental energy update
- OpenMP parallelization with dynamic partitioning

### Matrix Support

- `DenseMatrix` — row-major dense QUBO matrices  
- `SparseMatrix` — CSR-based storage for large sparse QUBOs  

### Performance
See detailed benchmarks in the Performance section below.

---

## Background

A QUBO problem minimizes the quadratic binary energy function

**E(x) = xᵀ Q x**, with **x ∈ {0,1}ⁿ**.

Brute-force search over all `2ⁿ` states becomes quickly infeasible, but  
for `n ≤ 30` GPUs can evaluate millions of states in parallel.

To avoid recomputing the energy from scratch for every state, we use:

### Gray-Code Traversal

We iterate over all binary states in Gray-code order, so that only **one bit changes** between consecutive states:

- Let `xₖ` be the current state.
- The next state is `xₖ₊₁ = xₖ ⊕ (1 << ctz(k + 1))`,  
  where `ctz` is the count of trailing zeros.

This is implemented via `std::countr_zero` on the CPU and `__ffsll` on the GPU.

### Row-Flip Incremental Update

When bit `i` flips, the energy difference can be computed as:

**ΔE = Σⱼ Qᵢⱼ xⱼ + Qᵢᵢ (1 - xᵢ) - Qᵢᵢ xᵢ**

so the new energy is just `E_new = E_old + ΔE`.

This incremental update is used in both CPU and GPU variants and reduces the
per-state cost from `O(n²)` to `O(n)`.

---

## Repository Structure

```text
gpu-qubo-solver/
│
├── CMakeLists.txt           # Top-level CMake build script
├── run.sh                   # Optional helper script (e.g. SLURM job)
│
├── src/
│   ├── CMakeLists.txt       # CMake configuration for executable
│   ├── main.cpp             # Entry point (CPU & GPU comparison)
│   ├── cpu_brute_force.h    # Naive + incremental CPU solvers
│   ├── gpu_brute_force.cu   # CUDA kernels + GPU solver
│   ├── gpu_brute_force.h
│   ├── qubo_energy.h        # Dense & sparse energy + incremental ΔE
│   ├── matrix.h             # Dense / sparse matrix structures
│   ├── matrix_reader.h      # MatrixMarket (.mtx) loader
│   ├── state_vector.h       # Bitset ↔ vector utilities
│   ├── datatypes.h          # Global typedefs
│   ├── cuda_util.h          # CUDA helper macros
│   ├── cuda_timer.h         # GPU timing helpers
│   ├── cuda_debug.h
│   └── qubo_brute_forcer.h  # Base class for CPU/GPU solvers
```


---

## Build Instructions

### Requirements

- CUDA Toolkit ≥ 12.x  
- GCC ≥ 11 or MSVC ≥ 19  
- CMake ≥ 3.20  
- Optional: OpenMP-enabled compiler  
- Linux or Windows (also tested under WSL)

---

### Build (Linux / WSL / Cluster Login Node)

```bash
mkdir build
cd build
cmake ..
make -j8
```

This produces an executable `QUBOBruteForcing` inside `build/`.

On HPC systems (e.g., Lichtenberg Cluster), load modules first:

```bash
module load cuda/12.5 gcc/13.1.0 cmake
```

---

## Run

After building, run the solver with a MatrixMarket `.mtx` file:

```bash
./QUBOBruteForcing path/to/matrix.mtx
```

Example:

```bash
./QUBOBruteForcing ../data/block_encoding_20.mtx
```

Typical output:

```text
========== GPU Information ==========
CUDA Devices: 1
Device 0: Tesla T4
  Compute Capability: 7.5
  SM Count: 40
  Global Memory: 14912 MB
=====================================

===========================================
Matrix: ../data/block_encoding_20.mtx
20 x 20, nnz = 60
CPU using 1 threads
Elapsed time for CPU brute force: 9.98522 milliseconds
CPU best energy = 0
CPU time = 9.98522 ms

GPU best energy = 0
GPU time = 2.0945 ms
Correctness: MATCH ✓

Speedup = 4.76736x
```


---

## Technical Overview

### Parallelization Strategy

- The state space is partitioned across `2ᵏ` GPU threads.
- Each thread enumerates a contiguous subspace of the full search space.
- Gray-code ordering ensures only **one bit flips** at each step.
- Bit index computed by:
  - `std::countr_zero(k+1)` on CPU  
  - `__ffsll(k+1) - 1` on GPU  
- Energy updates are computed incrementally:

```text
E_new = E_old + ΔE
```

- Complexity reduced from `O(n²·2ⁿ)` → `O(n·2ⁿ)`.

---

### Dense vs Sparse Handling

| Matrix Type | Storage | Kernel Characteristics |
|-------------|---------|------------------------|
| Dense       | Row-major `Q[n][n]` | Good for small/medium n; simple memory access |
| Sparse      | CSR (`values`, `offsets`, `columns`) | Skips zero entries; fast incremental update |

Both formats support up to **63 variables** (due to 64-bit encoding).

---

## Performance

All benchmarks were executed on an **NVIDIA Tesla T4 GPU** (40 SMs, 16 GB) and an Intel CPU (single-threaded baseline).  
The GPU solver consistently achieves significant acceleration for medium-to-large QUBO matrices (n ≥ 20), while very small matrices remain CPU-bound.

### **Summary of Observed Speedups**

| Category | Size (n) | CPU Time (ms) | GPU Time (ms) | Speedup | Notes |
|---------|----------|----------------|----------------|---------|-------|
| Block Encoding | 10 | 0.015 | 4.36 | 0.003× | CPU faster due to tiny workload |
| Block Encoding | 20 | 9.99 | 2.09 | 4.77× | GPU begins outperforming CPU |
| Block Encoding | 30 | 10248.8 | 273.49 | **37.47×** | Strong GPU acceleration |
| One-Hot Encoding | 10 | 0.025 | 1.29 | 0.02× | Very small, CPU dominates |
| One-Hot Encoding | 20 | 40.41 | 2.72 | **14.83×** | GPU advantage increases |
| One-Hot Encoding | 25 | 1518.1 | 22.47 | **67.56×** | Excellent GPU scaling |
| MaxCut | 8 | 0.008 | 0.85 | 0.009× | CPU trivial workload |
| MaxCut | 20 | 12.23 | 1.44 | **8.50×** | GPU clearly faster |
| MaxCut | 23 | 160.77 | 2.90 | **55.44×** | Sparse structure benefits GPU |
| MaxCut | 25 | 423.66 | 7.54 | **56.19×** | Strong GPU advantage |
| MaxCut | 30 | 10518.3 | 313.94 | **33.50×** | Large sparse QUBO → strong GPU scaling |
| Coloring | 16 | 0.95 | 1.21 | 0.78× | Small n, CPU wins |
| Coloring | 18 | 4.25 | 1.31 | 3.23× | GPU moderately faster |
| Coloring | 28 | 4154.2 | 157.63 | **26.35×** | Large QUBO → strong GPU scaling |

---

### **Key Observations**

- **GPU is slower than CPU for very small QUBOs (n < 12)**  
  Kernel launch overhead dominates.

- **Performance crossover occurs around n ≈ 18–20**  
  From this point on, the GPU solver consistently outperforms the CPU.

- **For large sparse QUBOs (n ≥ 25), GPU achieves 30–70× speedup**  
  - Sparse MaxCut cases show the best scaling  
  - Dense one-hot encoding also benefits greatly

- **Maximum observed speedup: 67.56× (One-hot, n=25)**  
- **Typical speedup range for n ≥ 20: 20× – 60×**

---

### **Why do speedups increase with problem size?**

- Gray-code incremental updates reduce per-state work to **O(n)**  
- GPU parallelism grows with the number of combinatorial states per thread  
- Memory access patterns (dense/sparse) become more efficient on larger workloads  
- CPU single-thread brute-force grows exponentially and becomes prohibitively slow

The results confirm that, beyond small trivial problem sizes, **GPU brute force is dramatically superior to CPU brute force—even with optimized CPU incremental updates.**

---

## Limitations

- Maximum variables: **63** (due to 64-bit state representation).  
- Dense GPU kernel uses `O(n²)` memory.  
- QUBOs with `n > 32` may still require significant runtime.  
- Sparse performance varies with matrix structure.

---

## Future Work

- Multi-GPU parallelization  
- Shared-memory optimized kernels  
- ELLPACK / sliced-ELL sparse formats  
- Heuristic solvers (SA, tabu search, evolutionary methods)  
- Hybrid CPU/GPU enumeration  
- Python bindings (PyBind11)

---

## License

Released under the **MIT License**.  
Template available at:  
https://opensource.org/licenses/MIT

---

## Acknowledgements

This project was developed as part of:

> PMPP – Programmierung Massiv-Paralleler Prozessoren  
> Technische Universität Darmstadt (WS 2025/26)

and expanded into an independent research-oriented system.

---

## Author

**Bo Fu**  
M.Sc. Informatik, TU Darmstadt  
GPU Computing • Optimization • High-Performance Computing  

GitHub: https://github.com/florianfu2003-glitch
