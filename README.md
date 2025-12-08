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

## 🚀 Key Features

### **GPU Acceleration**
- Custom CUDA kernels for dense and sparse (CSR) QUBO matrices  
- Efficient memory access patterns  
- Fully device-side Gray code bit-flip traversal  
- Incremental energy update reduces cost from \(O(n^2)\) → \(O(n)\) per step  

### **CPU Implementations**
- Naive brute force (full recomputation)
- Optimized CPU solver using incremental energy update
- OpenMP parallelization with dynamic partitioning

### **Matrix Support**
- `DenseMatrix` — row-major dense QUBO matrices  
- `SparseMatrix` — CSR-based storage for large sparse QUBOs  

### **Performance**
- Achieves **up to 20–50× speedup** over optimized CPU implementation  
- Designed for HPC systems (tested on Lichtenberg Cluster @ TU Darmstadt)

---

## 📘 Background

A QUBO problem minimizes:

\[
E(x) = x^T Q x,\quad x \in \{0,1\}^n.
\]

Brute-force search over all \(2^n\) states becomes quickly infeasible, but  
for \(n \le 30\) GPUs can evaluate millions of states in parallel.

To avoid recomputing energy from scratch, we use:

### **Gray Code Traversal**
Only **one bit changes** between consecutive states:

\[
x_{k+1} = x_k \oplus (1 \ll \text{ctz}(k+1))
\]

### **Row-Flip Incremental Update**
When bit \(i\) flips:

\[
\Delta E = \sum_{j} Q_{ij} x_j + Q_{ii}(1 - x_i) - Q_{ii} x_i.
\]

Used in both CPU and GPU variants.

---

## Repository Structure

gpu-qubo-solver/
│
├── CMakeLists.txt
├── README.md
├── run.sh # Optional helper script
│
├── src/
│ ├── main.cpp # Entry point (CPU & GPU comparison)
│ ├── cpu_brute_force.h # Naive + optimized incremental CPU solvers
│ ├── gpu_brute_force.cu # CUDA kernels + GPU solver
│ ├── gpu_brute_force.h
│ ├── qubo_energy.h # Dense + Sparse energy + incremental ΔE
│ ├── matrix.h # Dense/Sparse matrix representations
│ ├── matrix_reader.h # MatrixMarket (.mtx) loader
│ ├── state_vector.h # Bitset <-> vector conversions
│ ├── datatypes.h
│ ├── cuda_util.h
│ ├── cuda_timer.h
│ ├── cuda_debug.h
│ └── qubo_brute_forcer.h # Base class for CPU/GPU solvers

---

## ⚙️ Build Instructions

### **Requirements**
- CUDA Toolkit ≥ 12.x  
- GCC ≥ 11 or MSVC ≥ 19  
- CMake ≥ 3.20  
- Optional: OpenMP enabled compiler  
- Linux or Windows  

---

### **Build (Linux / WSL / macOS with GPU passthrough)**

```bash
mkdir build
cd build
cmake ..
make -j8

### **Run**

After building, run the solver by providing a MatrixMarket `.mtx` file:

```bash
./QUBOBruteForcing path/to/matrix.mtx
Example:
./QUBOBruteForcing ../data/block_encoding_20.mtx
You will see output similar to:
Matrix: block_encoding_20.mtx
20 x 20, nnz = 60

CPU (sparse):
Elapsed time: 15.23 ms

GPU (sparse):
Elapsed time: 0.31 ms
Speedup: 49.1x

Optimal states:
0: 1 0 1 0 1 1 0 0 1 1 ... Energy: -12.0

Technical Overview
Parallelization Strategy
State space is split across 2^k GPU threads
Each thread enumerates a contiguous block of states
Bit-flip index computed using CUDA intrinsic __ffsll()
Incremental update makes each step O(n) instead of O(n²)

Dense vs Sparse
Dense energy evaluation uses row-major Q storage
Sparse uses CSR format: (values, offsets, columns)
GPU sparse kernel skips all lower-triangular entries
Supports QUBO matrices up to dimension 63 × 63

Limitations
Maximum variables: 63 (due to 64-bit state representation)
GPU memory limits dense matrices to moderate n
Very large QUBOs (n > 32) still require significant runtime

Future Work
Potential extensions include:
Hybrid CPU/GPU parallel enumeration
Multi-GPU support
Improved sparse kernels (e.g., sliced ELLPACK format)
Heuristic solvers (annealing, tabu search)
Python bindings via PyBind11

License
MIT License (recommended).
Add a file named LICENSE with the following template:
https://opensource.org/licenses/MIT

Acknowledgements
This project was developed as part of
PMPP – Programmierung Massiv-Paralleler Prozessoren
at Technische Universität Darmstadt,
and extended into a standalone research-oriented implementation.

Author
Bo Fu
M.Sc. Informatik, TU Darmstadt
GPU Computing • Optimization • High-Performance Computing
GitHub: https://github.com/florianfu2003-glitch
