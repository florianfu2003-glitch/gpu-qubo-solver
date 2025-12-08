# pmpp_qubo_project

## Getting started

Use CMake to build your project (you might need to load respective modules first):

e.g.
```
mkdir build
cd build

cmake ../src -DCMAKE_BUILD_TYPE=Release
make -j8
```

## Compile options:

* `WITH_OPENMP`: Use OpenMP for CPU parallelization (default: OFF)
* `WITH_PROFILING`: Enable profiling information during CUDA compilation - compile with lineinfo (default: OFF)

## Usage
Try to stick to the framework structure and only alter framework files if really necessary.

The gpu implementation goes right into `gpu_qubo_brute_force.cu`.
The cpu implementation can be found in `cpu_brute_force.h`.
The `main.cpp` times all implementations and times them. You can reduce the `MAX_QUBO_SIZE` in `main.cpp` to test with smaller matrices.

It is required to provide an implementation for both dense and sparse matrices (csr format). 

The data folder needs to be in the same folder as the executable.

Use SLURM to run your program on the cluster, e.g. by using the provided `run.sh` script.

## Test matrices
There are 4 types of test matrices in the data folder:

* One hot encoding:
    * One-hot state results represent all possible vectors where exactly one element is set to 1 and all other elements are 0
* Block encoding:
    * There are always two solutions: All 1s or all 0s.
* Maximum cut:
    * The maximum cut problem involves partitioning the vertices of a graph into two disjoint subsets such that the number of edges between the subsets is maximized.
* Coloring:
    * The graph coloring problem involves assigning colors to the vertices of a graph such that no two adjacent vertices share the same color, using the minimum number of colors possible. A one hot encoding is used here to represent more than two colors.