/*
* @file nn.h
* @brief Main class
*
*
*/
#ifndef NN
#define NN

#include "nn_headers.h"

template <class T> class nn{

public:
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
    nn(){
        err = {};
        nn_shape = {};
        current_weight_method = weight_functions::standard;
        current_activation_function = activation_functions::sigmoid;
        last_layer_current_activation_function = activation_functions::linear;
        current_loss_method = loss_functions::mean_squared_error;
        learning_rate = 0.1;
        learning_rate_inertie = 0.05;
        set_shape();
        shaping();
        init_weight(0, 1);
    }

    /// @brief Determine the shape and the activation function for the propagation
    /// @param nn_shape
    /// @param act_function
    nn(const std::vector<ui32> nn_shape,
       const activation_functions act_function){
        err = {};
        this->nn_shape = nn_shape;
        current_weight_method = weight_functions::standard;
        current_activation_function = act_function;
        last_layer_current_activation_function = activation_functions::linear;
        current_loss_method = loss_functions::mean_squared_error;
        learning_rate = 0.1;
        learning_rate_inertie = 0.05;
        shaping();
        init_weight(0, 1);
    }

    /// @brief Fix the shape and the weight actualisation method
    /// @param nn_shape
    /// @param weight_method
    nn(const std::vector<ui32> nn_shape,
       const weight_functions weight_method){
        err = {};
        this->nn_shape = nn_shape;
        current_weight_method = weight_method;
        current_activation_function = activation_functions::sigmoid;
        last_layer_current_activation_function = activation_functions::linear;
        current_loss_method = loss_functions::mean_squared_error;
        learning_rate = 0.1;
        learning_rate_inertie = 0.05;
        shaping();
        init_weight(0, 1);
    }

    /// @brief Fix the shape, the activation function for the propagation and the weight actualisation method
    /// @param nn_shape
    /// @param act_function
    /// @param weight_method
    nn(const std::vector<ui32> nn_shape, const activation_functions act_function, const weight_functions weight_method){
        err = {};
        this->nn_shape = nn_shape;
        current_weight_method = weight_method;
        current_activation_function = act_function;
        last_layer_current_activation_function = activation_functions::linear;
        current_loss_method = loss_functions::mean_squared_error;
        learning_rate = 0.1;
        learning_rate_inertie = 0.05;
        shaping();
        init_weight(0, 1);
    };

    /// @brief Fix the shape, all activation functions and the weight actualisation method
    /// @param nn_shape
    /// @param act_function
    /// @param last_layer_act_function
    /// @param weight_method
    nn(const std::vector<ui32> nn_shape,
       const activation_functions act_function,
       const activation_functions last_layer_act_function,
       const weight_functions weight_method){
        err = {};
        this->nn_shape = nn_shape;
        current_weight_method = weight_method;
        current_activation_function = act_function;
        last_layer_current_activation_function = last_layer_act_function;
        current_loss_method = loss_functions::mean_squared_error;
        learning_rate = 0.1;
        learning_rate_inertie = 0.05;
        shaping();
        init_weight(0, 1);
    };

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
    void shaping(){
        assert(nn_shape.size() > 1); // Verification that our nn has an input/output layer
        const ui32 number_of_layers = nn_shape.size();

        /*  SHAPING THE THE NEURAL NETWORK  */
        NN_layers.resize(number_of_layers);
        NN_layers_D.resize(number_of_layers - 1);
        Weight_Matrix.resize(number_of_layers - 1);
        Weight_Matrix_D.resize(number_of_layers - 1);
        Bias_Matrix.resize(number_of_layers - 1);
        Bias_Matrix_D.resize(number_of_layers - 1);
        Output.resize(nn_shape[number_of_layers -1]);

        for(auto i = 0U; i < number_of_layers - 1; ++i){
            NN_layers.resize(nn_shape[i]);
            NN_layers_D.resize(nn_shape[i + 1]);
            Weight_Matrix.resize(nn_shape[i + 1] * nn_shape[i]);
            Weight_Matrix_D.resize(nn_shape[i + 1] * nn_shape[i]);
            Bias_Matrix.resize(nn_shape[i + 1]);
            Bias_Matrix_D.resize(nn_shape[i + 1]);
        }
        // OUTPUT LAYER
        NN_layers[number_of_layers - 1].resize(nn_shape[number_of_layers - 1]);

    };

    /// @brief Set the nn_shape from user input
    void set_shape(){
        ui32 number_of_layers;

        std::cout<<"How many different layers do you wish to have in the neural network? Input/Output included."<<std::endl;
        std::cin>>number_of_layers;
        nn_shape.reserve(number_of_layers);
        for(auto i = 0U; i < number_of_layers; ++i){
            if(i == 0U){
                std::cout<<"How many neurons(usually the size of the data) in the input layer?"<<std::endl;
                std::cin>>nn_shape[i];
                continue;
            }
            if(i == number_of_layers - 1){
                std::cout<<"How many neurons(usually the number of desired outcomes) in the output layer?"<<std::endl;
                std::cin>>nn_shape[i];
                continue;
            }
            std::cout<<"How many neurons in layer n°"<<i+1<<"?"<<std::endl;
            std::cin>>nn_shape[i];
        }
    };

    /*  -------------- nn_solve.cpp ---------------     */
    std::vector<T> Output;

    /// @brief Forward propagation for supervised model
    void propagation();

    /// @brief Backpropagation for supervised model
    void backpropagation();

};
#endif