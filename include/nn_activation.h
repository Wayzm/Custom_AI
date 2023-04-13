/*
* @file nn_activation.h
* @brief Declaration of the class with all activation related methods
*
*
*
*/

#ifndef NN_ACTIVATION
#define NN_ACTIVATION

#include "nn_headers.h"

/// @brief  Class with the different methods for node activation in the NN
template <typename T> class NN_activation{

public:

    /// @brief Sub class with the list of the different implemented activation functions
    enum class activation_functions{sigmoid, tanh, relu, linear};

private:
    /// @brief Sigmoid function
    /// @param x
    /// @return 1 / (1 + e^(-x))
    T sigmoid(T x);

    /// @brief Tanh function
    /// @param x
    /// @return 2 / (1 + e^(-2 * x)) -1
    T tanh(T x);

    /// @brief Relu function
    /// @param x
    /// @return WIP
    T relu(T x);

    /// @brief Simple linear function
    /// @param x
    /// @return 3.14 * x
    T linear(T x);

    /// @brief
    /// @param x
    /// @return
    T reciprocal_derivative_tanh(T x);

    /// @brief
    /// @param x
    /// @return
    T reciprocal_derivative_sigmoid(T x);

    /// @brief
    /// @param void
    /// @return
    T reciprocal_derivative_linear();

    /// @brief
    /// @param x
    /// @return
    T derivative_relu(T x);

    T activation(T x);

    T last_layer_activation(T x);

    T reciprocal_derivative_activation(T x);

    T last_layer_reciprocal_derivative_activation(T x);

};

#endif