#include "nn.h"

template <typename T> nn<T>::shaping(){
    assert(nn_shape.size() > 1); // Verification that our nn has an input/output layer
    const ui32 number_of_layers = nn_shape.size() - 1;

    /*  SHAPING THE THE NEURAL NETWORK  */
    NN_layers.resize(number_of_layers + 1);
    NN_layers_D.resize(number_of_layers);
    Weight_Matrix.resize(number_of_layers);
    Wieght_Matrix_D.resize(number_of_layers);
    Bias_Matrix.resize(number_of_layers);
    Bias_Matrix_D.resize(number_of_layers);

    for(auto i = 0U; i < number_of_layers; ++i){
        NN_layers.resize(nn_shape[i]);
        NN_layers_D.resize(nn_shape[i + 1]);
        Weight_Matrix.resize(nn_shape[i + 1] * nn_shape[i]);
        Wieght_Matrix_D.resize(nn_shape[i + 1] * nn_shape[i]);
        Bias_Matrix.resize(nn_shape[i + 1]);
        Bias_Matrix_D.resize(nn_shape[i + 1]);
    }
    // OUTPUT LAYER
    NN_layers[number_of_layers].resize(nn_shape[number_of_layers]);

}