#include "nn_activation.h"

template <typename T> NN_activation<T>::NN_activation(){
    current_activation_function = sigmoid;
    last_layer_current_activation_function = linear;
}

template <typename T> T NN_activation<T>::sigmoid(T x){
    return 1 / (1 + std::exp(-x));
}

template <typename T> T NN_activation<T>::tanh(T x){
    return 2 / (1 + std::exp(-2 * x)) -1;
}

template <typename T> T NN_activation<T>::linear(T x){
    return 3.14 * x;
}

template <typename T> T NN_activation<T>::relu(T x){
    return std::max(0, x);
}

template <typename T> T NN_activation<T>::derivative_sigmoid(T sigmoid_x){
  return sigmoid_x * (1 - sigmoid_x);
}

template <typename T> T NN_activation<T>::derivative_tanh(T tanh_x){
  return 1 - tanh_x * tanh_x;
}

template <typename T> T NN_activation<T>::derivative_relu(T x){
    return (0 < x) ? 1 : 0;
}

template <typename T> T NN_activation<T>::activation(T x){
  switch(current_activation_function){
    case activation_functions::sigmoid:
        return sigmoid(x);
    case activation_functions::tanh:
        return tanh(x);
    case activation_functions::relu:
        return relu(x);
    case activation_functions::linear:
        return linear(x);
    default:
        std::cerr<<"Activation function unknown."<<std::endl;
        break;
    }
    return 0;
}

template <typename T> T NN_activation<T>::last_layer_activation(T x){
  switch(last_layer_current_activation_function){
    case activation_functions::sigmoid:
        return sigmoid(x);
    case activation_functions::tanh:
        return tanh(x);
    case activation_functions::relu:
        return relu(x);
    case activation_functions::linear:
        return linear(x);
    default:
      std::cerr<<"Activation function unknown."<<std::endl;
      break;
  }
  return 0;
}

template <typename T> T NN_activation<T>::derivative_activation(T x){
    switch(current_activation_function){
    case activation_functions::sigmoid:
      return derivative_sigmoid(x);
    case activation_functions::tanh:
      return derivative_tanh(x);
    case activation_functions::relu:
      return derivative_relu(x);
    case activation_functions::linear:
      return derivative_linear(x);
    default:
      std::cerr << "Activation function unknown." << std::endl;
      break;
  }
  return 0;
}

template <typename T> T NN_activation<T>::last_layer_derivative_activation(T x){
    switch(last_layer_current_activation_function){
    case activations::sigmoid:
        return derivative_sigmoid(x);
    case activations::tanh:
        return derivative_tanh(x);
    case activations::relu:
        return derivative_relu(x);
    case activations::linear:
        return derivative_linear();
    default:
        std::cerr << "Activation function unknown." << std::endl;
        break;
    }
    return 0;
}

template <typename T> void NN_activation<T>::set_activation_method(activation_functions selected_function){
  current_activation_function = selected_function;
}

template <typename T> void NN_activation<T>::set_last_layer_activation_method(activation_functions selected_function){
  last_layer_current_activation_function = selected_function;
}