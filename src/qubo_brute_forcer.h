# pragma once

#include "matrix.h"
#include <vector>
/**
* QUBO Brute Forcer
* iT: index type
* vT: value type
* sT: state type
* MatrixType: matrix type (DenseMatrix<vT> or SparseMatrix<vT, iT>)
*/
template<typename iT, typename vT, typename sT, typename MatrixType>
struct QUBOBruteForcer {
    /**
     * Brute force search for optimal states of the QUBO problem represented by the matrix.
     * @param mat The QUBO matrix, stored on the CPU
     * @return A vector of vectors containing the optimal states.
     */
    virtual std::vector<std::vector<sT>> brute_force_optima(MatrixType const & mat) = 0;
};