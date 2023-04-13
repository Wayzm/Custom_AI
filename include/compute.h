#ifndef COMPUTE_H
#define COMPUTE_H

#include "nn_headers.h"

template <typename T> class compute{
public:
    compute();

    /// @brief y := a*x + y
    /// @param x
    /// @param a
    /// @param y
    void axpy(const std::vector<T> x, const T a, std::vector<T> y);

private:
    /// @brief Temporary vector for parallelisation purposes
    std::vector<T> temp_vector;

    /// @brief Temporary matrix for parallelisation purposes
    std::vector<std::vector<T>> temp_matrix;
};
#endif

