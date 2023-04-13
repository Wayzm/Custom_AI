/*
*
* @file nn_compute.h
* @brief Sub class with most compute intensive methods
*
*/

#ifndef NN_COMPUTE.H
#define NN_COMPUTE.H

#include "nn.h"

template <typename T> class NN_compute:nn{
public:

private:

    void propagation();

    void backpropagation();

    void compute_output();

    T compute_err();
};

#endif