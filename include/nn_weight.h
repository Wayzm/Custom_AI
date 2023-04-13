/*
* @file nn_weight.h
* @brief Declaration of the class with all weightrelated functions
*
*
*/

#ifndef NN_WEIGHT.H
#define NN_WEIGHT.H

#include "nn_headers.h"

template <typename T> class NN_weight{

public:
    /// @brief Sub class with the list of weight functions
    enum class weight_functions{inertie, standard};

private:
    void update_weight();


};

#endif