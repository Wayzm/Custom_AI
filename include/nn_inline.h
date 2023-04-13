#ifndef NN_INLINE.h
#define NN_INLINE.h

#include "nn_interface.h"

inline f64 NeuralNetwork::sigmoid(f64 x){
  return 1 / (1 + std::exp(-x));
}

inline f64 NeuralNetwork::tanh(f64 x){
  return 2 / (1 + std::exp(-2 * x)) -1;
}

inline f64 NeuralNetwork::linear(f64 x){
  return 3.14 * x;
}

#endif