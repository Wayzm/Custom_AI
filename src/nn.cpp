#include "nn.h"

template <typename T> nn<T>::nn(){
    err = {};
    nn_shape = {};
    current_weight_method = standard;
    current_activation_function = sigmoid;
    last_layer_current_activation_function = linear;
    learning_rate = 0.1;
    learning_rate_inertie = 0.05;
    set_shape();
    shape();
    init_weight();
}

template <typename T> nn<T>::nn(const std::vector<ui32> nn_shape,
                                const activation_functions act_function){
    err = {};
    current_weight_method = standard;
    current_activation_function = act_function;
    last_layer_current_activation_function = linear;
    learning_rate = 0.1;
    learning_rate_inertie = 0.05;
    shape();
    init_weight();
}

template <typename T> nn<T>::nn(const std::vector<ui32> nn_shape,
                                const weight_functions weight_method){
    err = {};
    current_weight_method = weight_method;
    current_activation_function = sigmoid;
    last_layer_current_activation_function = linear;
    learning_rate = 0.1;
    learning_rate_inertie = 0.05;
    shape();
    init_weight();
}

