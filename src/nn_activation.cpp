#include "nn.h"


template <class T> inline T nn<T>::sigmoid(T x){
    return 1 / (1 + std::exp(-x));
}

template <class T> inline T nn<T>::tanh(T x){
    return 2 / (1 + std::exp(-2 * x)) -1;
}

template <class T> inline T nn<T>::linear(T x){
    return 3.14 * x;
}

template <class T> inline T nn<T>::relu(T x){
    return (0 < x) ? x : 0;
}

template <class T> inline T nn<T>::derivative_sigmoid(T sigmoid_x){
  return sigmoid_x * (1 - sigmoid_x);
}

template <class T> inline T nn<T>::derivative_tanh(T tanh_x){
  return 1 - tanh_x * tanh_x;
}

template <class T> inline T nn<T>::derivative_relu(T x){
    return (0 < x) ? 1 : 0;
}

template <class T> inline T nn<T>::derivative_linear(){
    return 3.14;
}

template <class T> inline T nn<T>::activation(T x){
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

template <class T> inline T nn<T>::last_layer_activation(T x){
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

template <class T> inline T nn<T>::derivative_activation(T x){
    switch(current_activation_function){
    case activation_functions::sigmoid:
      return derivative_sigmoid(x);
    case activation_functions::tanh:
      return derivative_tanh(x);
    case activation_functions::relu:
      return derivative_relu(x);
    case activation_functions::linear:
      return derivative_linear();
    default:
      std::cerr << "Activation function unknown." << std::endl;
      break;
  }
  return 0;
}

template <class T> inline T nn<T>::last_layer_derivative_activation(T x){
    switch(last_layer_current_activation_function){
    case activation_functions::sigmoid:
        return derivative_sigmoid(x);
    case activation_functions::tanh:
        return derivative_tanh(x);
    case activation_functions::relu:
        return derivative_relu(x);
    case activation_functions::linear:
        return derivative_linear();
    default:
        std::cerr << "Activation function unknown." << std::endl;
        break;
    }
    return 0;
}

template <class T> void nn<T>::set_activation_method(activation_functions selected_function){
  current_activation_function = selected_function;
}

template <class T> void nn<T>::set_last_layer_activation_method(activation_functions selected_function){
  last_layer_current_activation_function = selected_function;
}

template class nn<f32>;
template class nn<f64>;