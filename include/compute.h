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



template <typename T> class compute{
public:
    compute();

    /// @brief y := a*x + y
    /// @param x
    /// @param a
    /// @param y
    void axpy(const std::vector<T> x, const T a, std::vector<T> y);

    /// @brief Z := a * X * Y + b * Z
    /// @param X
    /// @param Y
    /// @param a
    /// @param Z
    /// @param b
    void emm(const std::vector<std::vector<T>> X,
             const std::vector<std::vector<T>> Y,
             const T a,
             std::vector<std::vector<T>> Z,
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
private:
    /// @brief Temporary vector for parallelisation purposes
    std::vector<T> temp_vector;

    /// @brief Temporary matrix for parallelisation purposes
    std::vector<std::vector<T>> temp_matrix;
};
#endif

