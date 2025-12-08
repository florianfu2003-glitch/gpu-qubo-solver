#pragma once
#include "matrix.h"

#include <cassert>

/// calculate state[row] * state[col] for binary variables
template <typename vT, typename iT>
inline vT get_state_entry_product(size_t state, size_t n, iT row, iT col) {
    size_t row_idx = n - 1ULL - static_cast<size_t>(row);
    size_t col_idx = n - 1ULL - static_cast<size_t>(col);
    return static_cast<vT>(((state >> row_idx) & 1ULL) & ((state >> col_idx) & 1ULL));
}

/// Compute the energy of a given state for a QUBO problem represented by a matrix with an array of binary variables
template<typename vT, typename iT, typename sT>
vT compute_energy(SparseMatrix<vT, iT> const & mat, const sT* state) {
    assert(mat.rows == mat.cols); // energy computation only for square matrices
    vT energy = 0;
    for (iT row = 0; row < mat.rows; row++) {
        for (iT idx = mat.offsets[row]; idx < mat.offsets[row + 1]; idx++) {
            iT col = mat.columns[idx];
            vT value = mat.values[idx];
            if(col >= row) // to avoid double counting in symmetric matrices
                energy += value * (state[row] * state[col]);
        }
    }
    return energy;
}

/// Compute the energy of a given state for a QUBO problem represented by a matrix with binary variables encoded in a size_t
template<typename vT, typename iT>
vT compute_energy(SparseMatrix<vT, iT> const & mat, size_t state) {
    assert(mat.rows == mat.cols); // energy computation only for square matrices
    assert(mat.rows <= 64);
    vT energy = 0;
    for (iT row = 0; row < mat.rows; row++)
    {
        iT first = mat.offsets[row];
        iT last = mat.offsets[row + 1];
        for (iT idx = first; idx < last; idx++)
        {
            iT col = mat.columns[idx];
            vT value = mat.values[idx];
            if (col >= row) // to avoid double counting in symmetric matrices
                energy += value * get_state_entry_product<vT>(state, mat.rows, row, col);
        }
    }
    return energy;
}

/// Compute the energy of a given state for a QUBO problem represented by a dense matrix with an array of binary variables
template<typename vT, typename sT>
vT compute_energy(DenseMatrix<vT> const & mat, const sT* state) {
    assert(mat.rows == mat.cols); // energy computation only for square matrices
    vT energy = 0;
    for (size_t row = 0; row < mat.rows; row++) {
        for (size_t col = row; col < mat.cols; col++) {
            energy += mat.data[row * mat.cols + col] * (state[row] * state[col]);
        }
    }
    return energy;
}

/// Compute the energy of a given state for a QUBO problem represented by a dense matrix with binary variables encoded in a size_t
template<typename vT>
vT compute_energy(DenseMatrix<vT> const & mat, size_t state) {
    assert(mat.rows == mat.cols); // energy computation only for square matrices
    assert(mat.rows <= 64);
    vT energy = 0;
    for (size_t row = 0; row < mat.rows; row++)
    {
        for (size_t col = row; col < mat.cols; col++)
        {
            energy += mat.data[row * mat.cols + col] * get_state_entry_product<vT>(state, mat.rows, row, col);
        }
    }
    return energy;
}

/// Compute the energy difference for a certain row when flipping the row-th bit in the state for a QUBO problem represented by a dense matrix
template<typename vT>
vT compute_row_flip_energy_difference(DenseMatrix<vT> const & mat, size_t state, size_t row) {
    assert(mat.rows == mat.cols); // energy computation only for square matrices
    assert(mat.rows <= 64);
    size_t row_idx = (mat.rows - 1ULL - row);
    size_t bit = (state >> row_idx) & 1ULL;
    size_t flipped = 1ULL - bit;
    int change = flipped - bit;

    vT energy_diff = 0;
    for (size_t col = 0; col < mat.cols; col++)
    {
        int idx = row * mat.cols + col;
        size_t col_idx = (mat.rows - 1ULL - col);
        energy_diff += mat.data[idx] * static_cast<vT>((state >> col_idx) & 1ULL); // value * state[col]
    }

    energy_diff += mat.data[row * mat.cols + row] * static_cast<vT>(flipped);
    energy_diff *= static_cast<vT>(change);

    return energy_diff;
}

/// Compute the energy difference for a certain row when flipping the row-th bit in the state for a QUBO problem represented by a dense matrix
template<typename vT, typename iT>
vT compute_row_flip_energy_difference(SparseMatrix<vT, iT> const & mat, size_t state, size_t row) {
    assert(mat.rows == mat.cols); // energy computation only for square matrices
    assert(mat.rows <= 64);
    size_t row_idx = (mat.rows - 1ULL - row);
    size_t bit = (state >> row_idx) & 1ULL;
    size_t flipped = 1ULL - bit;
    int change = flipped - bit;

    vT energy_diff = 0;

    iT first = mat.offsets[row];
    iT last = mat.offsets[row + 1];
    int diag_idx = -1;

    for (iT idx = first; idx < last; idx++)
    {
        auto const & col = mat.columns[idx];
        size_t col_idx = (mat.rows - 1ULL - col);
        energy_diff += mat.values[idx] * static_cast<vT>((state >> col_idx) & 1ULL); // value * state[col]
        if (row == col)
            diag_idx = idx;
    }

    if(diag_idx != -1)
        energy_diff += mat.values[diag_idx] * static_cast<vT>(flipped);
    energy_diff *= static_cast<vT>(change);

    return energy_diff;
}