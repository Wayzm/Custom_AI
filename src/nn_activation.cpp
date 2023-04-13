#include "nn_activation.h"

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

template <typename T> T NN_activation<T>::reciprocal_derivative_sigmoid(T sigmoid_x){
  return sigmoid_x * (1 - sigmoid_x);
}

template <typename T> T NN_activation<T>::reciprocal_derivative_tanh(T tanh_x){
  return 1 - tanh_x * tanh_x;
}

template <typename T> T NN_activation<T>::reciprocal_derivative_linear(){
  return 3.14;
}

template <typename T> T NN_activation<T>::activation(T x){
  switch(current_activation){
    case activations::sigmoid:
      return sigmoid(x);
    case activations::tanh:
      return std::tanh(x);
    case activations::relu:
      return relu(x);
    default:
      std::cerr<<"Activation function unknown."<<std::endl;
      break;
  }
  return 0;
}

template <typename T> T NN_activation<T>::last_layer_activation(T x){
  switch(current_last_layer_activation){
    case activations::sigmoid:
      return sigmoid(x);
    case activations::tanh:
      return std::tanh(x);
    case activations::relu:
      return relu(x);
    case activations::linear:
      return linear(x);
    default:
      std::cerr<<"Activation function unknown."<<std::endl;
      break;
  }
  return 0;
}

template <typename T> T NN_activation<T>::reciprocal_derivative_activation(T x){
    switch(current_activation){
    case activations::sigmoid:
      return reciprocal_derivative_sigmoid(x);
    case activations::tanh:
      return reciprocal_derivative_tanh(x);
    case activations::relu:
      return derivative_relu(x);
    default:
      std::cerr << "Activation function unknown." << std::endl;
      break;
  }
  return 0;
}

template <typename T> T NN_activation<T>::last_layer_reciprocal_derivative_activation(T x){
    switch(current_last_layer_activation){
    case activations::sigmoid:
      return reciprocal_derivative_sigmoid(x);
    case activations::tanh:
      return reciprocal_derivative_tanh(x);
    case activations::relu:
      return reciprocal_relu(x);
    case activations::linear:
      return reciprocal_derivative_linear();
    default:
      std::cerr << "Activation function unknown." << std::endl;
      break;
  }
  return 0;
}
