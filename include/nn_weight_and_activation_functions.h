/*
* @file nn_common.h
* @brief Interface / User functions for the neural network
*
*
*
*/

#ifndef NN_INTERFACE
#define NN_INTERFACE

#include "nn_headers.h"

/// @brief  Class with the different methods for weight and activation (waf = weight and activation functions, for short)
template <typename T> class NN_waf{

public:

    /// @brief Sub class with the list of the different implemented activation functions
    enum class activation_functions{sigmoid, tanh, relu, linear};

    /// @brief Sub class with the list of weight functions
    enum class weight_functions{inertie, standard};

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
    /// @param double x
    /// @return WIP
    T relu(T x);

    /// @brief Simple linear function
    /// @param double x
    /// @return 3.14 * x
    T linear(T x);

};

#endif