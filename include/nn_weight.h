/*
* @file nn_weight.h
* @brief Declaration of the class with all weightrelated functions
*
*
*/

#ifndef NN_WEIGHT.H
#define NN_WEIGHT.H

#include "nn_headers.h"
#include "nn.h"

template <typename T> class NN_weight:nn{

public:
    /// @brief Fix the weight method
    /// @param selected_method
    void set_weight_method(weight_functions selected_method);

    /// @brief Constructor with default weight settings
    NN_weight();
    /// @brief Destructor
    ~NN_weight(){};

private:
    /// @brief Sub class with the list of weight functions
    enum class weight_functions{inertie, standard};

    /// @brief Default weight seed
    ui32 weight_seed = 1337;

    /// @brief Current weight method in use
    weight_functions current_weight_method;

    /// @brief Default weight configuration
    void init_weight(const ui32 seed, const T min, const T max);

    void inertie();

    void standard();


    /// @brief Update the weight on the different nodes following the current_weight_method
    void update_weight();

};

#endif