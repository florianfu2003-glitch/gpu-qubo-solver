#pragma once
#include <type_traits>

template<typename sT>
size_t state_vector_to_binary_reprensentation(std::vector<sT> const & state_vector) {
    size_t state = 0ULL;
    size_t n = state_vector.size();
    for (size_t i = 0; i < n; i++) {
        state |= static_cast<size_t>(state_vector[i]) << (n - 1ULL - i);
    }
    return state;
}



template<typename sT>
std::vector<sT> binary_representation_to_state_vector(size_t state, size_t n) {
    std::vector<sT> state_vector(n, 0);
    for (size_t i = 0; i < n; i++) {
        state_vector[i] = static_cast<sT>((state >> (n - i - 1ULL)) & 1ULL);
    }
    return state_vector;
}