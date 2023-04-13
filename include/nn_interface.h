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

/// @brief Main class with which the user will interact with
class NeuralNetwork{

public:

    /// @brief Sub class with the list of the different implemented activation functions
    enum class activation_functions{sigmoid, tanh, relu, linear};
    /// @brief Sub class with the list of weight functions
    enum class weight_functions{inertie, standard};
    /// @brief Size of the input data pool
    ui32 number_of_elements;

private:
    /// @brief Sigmoid function
    /// @param double x
    /// @return 1 / (1 + e^(-x))
    f64 sigmoid(f64 x);
    /// @brief Tanh function
    /// @param double x
    /// @return 2 / (1 + e^(-2 * x)) -1
    f64 tanh(f64 x);
    /// @brief Relu function
    /// @param double x
    /// @return WIP
    f64 relu(f64 x);
    /// @brief Simple linear function
    /// @param double x
    /// @return 3.14 * x
    f64 linear(f64 x);

};

#endif