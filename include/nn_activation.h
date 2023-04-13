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
#include "nn.h"

/// @brief  Class with the different methods for node activation in the NN
template <typename T> class NN_activation:nn{

public:

    /// @brief Constructor
    NN_activation();

    /// @brief Destructor
    ~NN_activation();

    /// @brief Sub class with the list of the different implemented activation functions
    enum class activation_functions{sigmoid, tanh, relu, linear};

    /// @brief Fix the activation for the hidden layers
    /// @param selected_function
    void set_activation_method(activation_functions selected_function);

    /// @brief Fix the activation for the output layer
    /// @param selected_function
    void set_last_layer_activation_method(activation_functions selected_function);

private:
    /// @brief Current functions
    activation_functions current_activation_function, last_layer_current_activation_function;

    /// @brief Sigmoid function
    /// @param x
    /// @return 1 / (1 + e^(-x))
    T sigmoid(T x);

    /// @brief Tanh function
    /// @param x
    /// @return 2 / (1 + e^(-2 * x)) - 1
    T tanh(T x);

    /// @brief Relu function
    /// @param x
    /// @return max(0, x)
    T relu(T x);

    /// @brief Simple linear function
    /// @param x
    /// @return 3.14 * x
    T linear(T x);

    /// @brief derivative function of tanh(x)
    /// @param x
    /// @return 1 - tanh_x * tanh_x
    T derivative_tanh(T x);

    /// @brief derivative function of sigmoid
    /// @param x
    /// @return
    T derivative_sigmoid(T x);

    /// @brief Derivative of linear function
    /// @param x
    /// @return 3.14
    T derivative_linear(T x);

    /// @brief Derivative of relu function
    /// @param x
    /// @return 3.14
    T derivative_relu(T x);

    /// @brief Depending on the variable current_activation_function, will return the corresponding result
    /// @param x
    /// @return current_activation_function(x)
    T activation(T x);

    /// @brief Depending on the variable last_layer_current_activation_function, will return the corresponding result
    /// @param x
    /// @return last_layer_current_activation_function(x)
    T last_layer_activation(T x);

    /// @brief Depending on the variable current_activation_function, will return the derivative of that function
    /// @param x
    /// @return derivative of the current function (x)
    T derivative_activation(T x);

    /// @brief Depending on the variable last_layer_current_activation_function, will return the derivative of that function
    /// @param x
    /// @return derivative of the current function in the last layer (x)
    T last_layer_derivative_activation(T x);

};

#endif