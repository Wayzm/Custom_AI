/*
*
*   @file compute.h
*   @brief tailor made blas function for the neural network
*
*
*/


#ifndef COMPUTE_H
#define COMPUTE_H

#include "nn_headers.h"

// So, as we though there are full matrices in weight/bias

template <class T> class compute{
public:
    compute();

    /// @brief y := a*x + y
    /// @param x
    /// @param a
    /// @param y
    void axpy(const std::vector<T> x, const T a, std::vector<T> y);

    /// @brief Z := a * X * Y + b * Z
    /// @param X
    /// @param rows_x
    /// @param cols_x
    /// @param Y
    /// @param rows_y
    /// @param cols_y
    /// @param a
    /// @param Z
    /// @param rows_z
    /// @param cols_z
    /// @param b
    void emm(const std::vector<T> X,
             const ui32 rows_x,
             const ui32 cols_x,
             const std::vector<T> Y,
             const ui32 rows_y,
             const ui32 cols_y,
             const T a,
             std::vector<T> Z,
             const ui32 rows_z,
             const ui32 cols_z,
             const T b);

    /// @brief z = a * X * y + b * z
    /// @param X
    /// @param y
    /// @param a
    /// @param z
    /// @param b
    void emv(const std::vector<std::vector<T>> X,
             const std::vector<T> y,
             const T a,
             std::vector<T> z,
             const T b);

    /// @brief Use the first norm
    /// @param vector
    void normalisation(std::vector<T> vector);
private:
    /// @brief Temporary vector for parallelisation purposes
    std::vector<T> temp_vector;

    /// @brief Temporary matrix for parallelisation purposes
    std::vector<T> temp_matrix;
};
#endif

