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

    /// @brief Fix the weight method
    /// @param selected_method
    void set_weight_method(weight_functions selected_method);

private:
    /// @brief Current weight method in use
    weight_functions current_weight_method;

    /// @brief Update the weight on the different nodes following the current_weight_method
    void update_weight();


};

#endif