#include "nn.h"

// template <class T> nn<T>::nn(){
//     err = {};
//     nn_shape = {};
//     current_weight_method = weight_functions::standard;
//     current_activation_function = activation_functions::sigmoid;
//     last_layer_current_activation_function = activation_functions::linear;
//     current_loss_method = loss_functions::mean_squared_error;
//     learning_rate = 0.1;
//     learning_rate_inertie = 0.05;
//     set_shape();
//     shaping();
//     init_weight(0, 1);
// }

// template <class T> nn<T>::nn(const std::vector<ui32> nn_shape,
//                                 const activation_functions act_function){
//     err = {};
//     this->nn_shape = nn_shape;
//     current_weight_method = weight_functions::standard;
//     current_activation_function = act_function;
//     last_layer_current_activation_function = activation_functions::linear;
//     current_loss_method = loss_functions::mean_squared_error;
//     learning_rate = 0.1;
//     learning_rate_inertie = 0.05;
//     shaping();
//     init_weight(0, 1);
// }

// template <class T> nn<T>::nn(const std::vector<ui32> nn_shape,
//                                 const weight_functions weight_method){
//     err = {};
//     this->nn_shape = nn_shape;
//     current_weight_method = weight_method;
//     current_activation_function = activation_functions::sigmoid;
//     last_layer_current_activation_function = activation_functions::linear;
//     current_loss_method = loss_functions::mean_squared_error;
//     learning_rate = 0.1;
//     learning_rate_inertie = 0.05;
//     shaping();
//     init_weight(0, 1);
// }

// template <class T> nn<T>::nn(const std::vector<ui32> nn_shape,
//                                 const activation_functions act_function,
//                                 const weight_functions weight_method){
//     err = {};
//     this->nn_shape = nn_shape;
//     current_weight_method = weight_method;
//     current_activation_function = act_function;
//     last_layer_current_activation_function = activation_functions::linear;
//     current_loss_method = loss_functions::mean_squared_error;
//     learning_rate = 0.1;
//     learning_rate_inertie = 0.05;
//     shaping();
//     init_weight(0, 1);
// }

// template <class T> nn<T>::nn(const std::vector<ui32> nn_shape,
//                                 const activation_functions act_function,
//                                 const activation_functions last_layer_act_function,
//                                 const weight_functions weight_method){
//     err = {};
//     this->nn_shape = nn_shape;
//     current_weight_method = weight_method;
//     current_activation_function = act_function;
//     last_layer_current_activation_function = last_layer_act_function;
//     current_loss_method = loss_functions::mean_squared_error;
//     learning_rate = 0.1;
//     learning_rate_inertie = 0.05;
//     shaping();
//     init_weight(0, 1);
// }

template class nn<f32>;
template class nn<f64>;
