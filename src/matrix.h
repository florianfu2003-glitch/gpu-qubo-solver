#pragma once

//Dense matrix row-major format
template<typename T>
struct DenseMatrix {

    DenseMatrix(size_t r, size_t c) : rows(r), cols(c) {
        data = new T[rows * cols];
    }

    ~DenseMatrix() {
        delete[] data;
    }

    void set_zero() {

        for (size_t r = 0; r < rows; r++)
        {
            for (size_t c = 0; c < rows; c++)
            {
                at(r, c) = T(0);
            }
        }
    }
    
    T& at(size_t r, size_t c) {
        return data[r * cols + c];
    }

    size_t rows, cols;
    T* data;
};

//Sparse matrix in CSR format (row-major)
template<typename vT, typename iT>
struct SparseMatrix {
    SparseMatrix(size_t r, size_t c) : rows(r), cols(c) {}

    SparseMatrix(SparseMatrix<vT, iT> && mat) {
        rows = mat.rows;
        cols = mat.cols;
        nnz = mat.nnz;
        std::swap(values, mat.values);
        std::swap(offsets, mat.offsets);
        std::swap(columns, mat.columns);
    }

    SparseMatrix& operator=(SparseMatrix&& mat) {
        rows = mat.rows;
        cols = mat.cols;
        nnz = mat.nnz;
        std::swap(values, mat.values);
        std::swap(offsets, mat.offsets);
        std::swap(columns, mat.columns);
        return *this;
    }

    SparseMatrix(const SparseMatrix&) = delete;
    SparseMatrix& operator=(const SparseMatrix&) = delete;

    void allocate(size_t nonZeros) {
        nnz = nonZeros;
        values = new vT[nnz];
        offsets = new iT[rows + 1];
        columns = new iT[nnz];
    }

    ~SparseMatrix() {
        if (values)
            delete[] values;
        if (offsets)
            delete[] offsets;
        if (columns)
            delete[] columns;
    }

    size_t rows, cols, nnz;
    vT* values = nullptr;
    iT* offsets = nullptr;
    iT* columns = nullptr;
};

template<typename vT, typename iT>
DenseMatrix<vT> sparse_to_dense(const SparseMatrix<vT, iT>& sparse) {
    DenseMatrix<vT> dense(sparse.rows, sparse.cols);
    dense.set_zero();
    for (size_t r = 0; r < sparse.rows; r++) {
        for (iT idx = sparse.offsets[r]; idx < sparse.offsets[r + 1]; idx++) {
            iT c = sparse.columns[idx];
            dense.at(r, c) = sparse.values[idx];
        }
    }
    return dense;
}
