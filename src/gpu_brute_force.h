#pragma once

#include "qubo_brute_forcer.h"
#include "matrix.h"
#include <stdexcept>


/****************************************
* DO NOT CHANGE THIS FILE               *
* USE THE INTERFACE AS IS!              *
* IMPLEMENTATION IN gpu_brute_force.cu  *
****************************************/


/**
* iT: index type
* vT: value type
* sT: state type
* MatrixType: matrix type (DenseMatrix<vT> or SparseMatrix<vT, i
* GPU implementation of the QUBO brute forcer that uses incremental energy updates, based on
*/
template<typename iT, typename vT, typename sT, typename MatrixType>
struct GPUQUBOBruteForcer : public QUBOBruteForcer<iT, vT, sT, MatrixType> {
    /// Brute force search for optimal states of the QUBO problem represented by the matrix. 
    /// DO NOT CHANGE!
    std::vector<std::vector<sT>> brute_force_optima(MatrixType const & mat) override;
};
