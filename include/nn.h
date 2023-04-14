/*
* @file nn.h
* @brief Main class
*
*
*/
#ifndef NN
#define NN

#include "nn_headers.h"

template <typename T> class nn{

private:
    /// @brief Sub class with the list of the different implemented activation functions
    enum class activation_functions{sigmoid, tanh, relu, linear};

    /// @brief Sub class with the list of weight functions
    enum class weight_functions{inertie, standard};

    /// @brief Sub class with the list of loss functions
    enum class loss_functions{mean_squared_error,
                              mean_absolute_error,
                              hinge_error,
                              cross_entropy_error,
                              mean_bias_error};

public:

    /// @brief Default construction of the neural network
    nn();

    /// @brief Determine the shape and the activation function for the propagation
    /// @param nn_shape
    /// @param act_function
    nn(const std::vector<ui32> nn_shape, const activation_functions act_function);

    /// @brief Fix the shape and the weight actualisation method
    /// @param nn_shape
    /// @param weight_method
    nn(const std::vector<ui32> nn_shape, const weight_functions weight_method);

    /// @brief Fix the shape, the activation function for the propagation and the weight actualisation method
    /// @param nn_shape
    /// @param act_function
    /// @param weight_method
    nn(const std::vector<ui32> nn_shape, const activation_functions act_function, const weight_functions weight_method);

    /// @brief Fix the shape, all activation functions and the weight actualisation method
    /// @param nn_shape
    /// @param act_function
    /// @param last_layer_act_function
    /// @param weight_method
    nn(const std::vector<ui32> nn_shape,
       const activation_functions act_function,
       const activation_functions last_layer_act_function,
       const weight_functions weight_method);

    /// @brief Destructor
    ~nn(){};

    /// @brief User function to modify the learning rate, effective for both weight methods
    /// @param rate
    void set_learning_rate(T rate);

    /// @brief User function to modify the learning rate inertie, effictive only for the inertie weight method
    /// @param rate
    void set_learning_rate_inertie(T rate);

    /* -------------------         nn_weight.cpp        --------------------------       */

    /// @brief Fix the weight method
    /// @param selected_method
    void set_weight_method(weight_functions selected_method);

    /// @brief Fix the weight method and the desired seed number
    /// @param selected_method
    /// @param seed
    void set_weight_method(weight_functions selected_method, const ui32 seed);

    /*  -----------------       nn_activation.cpp      ---------------------- */
    /// @brief Fix the activation for the hidden layers
    /// @param selected_function
    void set_activation_method(activation_functions selected_function);

    /// @brief Fix the activation for the output layer
    /// @param selected_function
    void set_last_layer_activation_method(activation_functions selected_function);

    /*  ----------------    nn_solve.cpp -----------------  */

    /// @brief User function to train the neural network with an array of multiple inputs data
    /// @param Inputs
    void training_unsupervised(const std::vector<std::vector<T>> Inputs);

    /// @brief User function to train the neural network with an array of inputs and corresponding outputs
    /// @param Inputs
    /// @param Outputs
    void training_supervised(const std::vector<T> Input, const std::vector<T> Output);

    /// @brief User function to use the neural network with a given input
    /// @param Input
    std::vector<T> compute_output(const std::vector<T> Input);

    /*  ----------------    nn_error.cpp -----------------  */

    /// @brief User function to get the error
    /// @return err
    T get_err();

    /// @brief User function to select the loss function
    /// @param selected_method
    void set_loss_method(loss_functions selected_method);



private:

    /// @brief Shape of the neural network
    std::vector<ui32> nn_shape;

    /// @brief Matrices with the weight values in the neural network for the propagation and backpropagation
    std::vector<std::vector<T>> Weight_Matrix, Weight_Matrix_D;

    /// @brief Matrices with the bias values in the neural network for the propagation and the backpropagation
    std::vector<std::vector<T>> Bias_Matrix, Bias_Matrix_D;

    /// @brief Matrices with each node's value (input, hidden and output layers)
    std::vector<std::vector<T>> NN_layers, NN_layers_D;

    /// @brief Learning rates used for the weight update methods
    T learning_rate, learning_rate_inertie;

    /*  ------------------   nn_error.cpp   -------------------- */

    /// @brief Error computed at teh output layer during the learning phase
    T err;

    /// @brief Current loss function, default value being mean_squared_error
    loss_functions current_loss_method;

    /// @brief Compute the error depending on the current loss method
    void compute_error();

    /// @brief MSE method
    void mean_squared_error();

    /// @brief MAE method
    void mean_absolute_error();

    /// @brief MBE method
    void mean_bias_error();

    /// @brief HINGE method
    void hinge_error();

    /// @brief CE method
    void cross_entropy_error();

    /* --------------------   nn_weight.cpp  -------------------  */

    /// @brief Default weight seed
    ui32 weight_seed = 1337;

    /// @brief Current weight method in use
    weight_functions current_weight_method;

    /// @brief Default weight configuration
    void init_weight(const T min, const T max);

    /// @brief
    void inertie();

    /// @brief
    void standard();

    /// @brief Update the weight on the different nodes following the current_weight_method
    void update_weight();


    /*  ----------------------- nn_activation.cpp ----------------------    */

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
    /// @param tan_x
    /// @return 1 - tanh_x * tanh_x
    T derivative_tanh(T tan_x);

    /// @brief derivative function of sigmoid
    /// @param sigmoid_x
    /// @return sigmoid_x (1 - sigmoid_x)
    T derivative_sigmoid(T sigmoid_x);

    /// @brief Derivative of linear function
    /// @param x
    /// @return 3.14
    T derivative_linear();

    /// @brief Derivative of relu function
    /// @param x
    /// @return 1 or 0
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

    /*  -------------- nn_shaping.cpp -------------     */

    /// @brief Memory allocation of all structures in the nn
    void shaping();

    /// @brief Set the nn_shape from user input
    void set_shape();

    /*  -------------- nn_solve.cpp ---------------     */
    std::vector<T> Output;

    /// @brief Forward propagation for supervised model
    void propagation();

    /// @brief Backpropagation for supervised model
    void backpropagation();

};

#endif