#include "nn_weight_and_activation_functions.h"

template <typename T> T NN_waf<T>::sigmoid(T x){
    return 1 / (1 + std::exp(-x));
}

template <typename T> T NN_waf<T>::tanh(T x){
    return 2 / (1 + std::exp(-2 * x)) -1;
}

template <typename T> T NN_waf<T>::linear(T x){
    return 3.14 * x;
}