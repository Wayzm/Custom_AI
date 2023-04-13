#ifndef NN
#define NN

#include "nn_headers.h"

template <typename T> class nn{
public:

    /// @brief Default construction of the neural network
    nn();

    /// @brief Determine the shape and the number of elements
    /// @param nn_shape
    /// @param nbr_elements
    nn(const std::vector<ui32> nn_shape, const ui32 nbr_elements);

    /// @brief Destructor
    ~nn(){};

protected:

    T learning_rate, learning_rate_inertie;

private:

};

#endif