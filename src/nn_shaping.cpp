#include "nn.h"

template <typename T> void nn<T>::set_shape(){
    ui32 number_of_layers;

    std::cout<<"How many different layers do you wish to have in the neural network? Input/Output included."<<std::endl;
    std::cin>>number_of_layers;
    nn_shape.reserve(number_of_layers);
    for(auto i = 0U; i < number_of_layers; ++i){
        if(i == 0U){
            std::cout<<"How many neurons(usually the size of the data) in the input layer?"<<std::endl;
            std::cin>>nn_shape[i];
            continue;
        }
        if(i == number_of_layers - 1){
            std::cout<<"How many neurons(usually the number of desired outcomes) in the output layer?"<<std::endl;
            std::cin>>nn_shape[i];
            continue;
        }
        std::<<"How many neurons in layer n°"<<i<<"?"<<std::endl;
        std::cin>>nn_shape[i];
    }
}

template <typename T> void nn<T>::shaping(){
    assert(nn_shape.size() > 1); // Verification that our nn has an input/output layer
    const ui32 number_of_layers = nn_shape.size();

    /*  SHAPING THE THE NEURAL NETWORK  */
    NN_layers.resize(number_of_layers);
    NN_layers_D.resize(number_of_layers - 1);
    Weight_Matrix.resize(number_of_layers - 1);
    Weight_Matrix_D.resize(number_of_layers - 1);
    Bias_Matrix.resize(number_of_layers - 1);
    Bias_Matrix_D.resize(number_of_layers - 1);
    Output.resize(nn_shape[number_of_layers -1]);

    for(auto i = 0U; i < number_of_layers - 1; ++i){
        NN_layers.resize(nn_shape[i]);
        NN_layers_D.resize(nn_shape[i + 1]);
        Weight_Matrix.resize(nn_shape[i + 1] * nn_shape[i]);
        Weight_Matrix_D.resize(nn_shape[i + 1] * nn_shape[i]);
        Bias_Matrix.resize(nn_shape[i + 1]);
        Bias_Matrix_D.resize(nn_shape[i + 1]);
    }
    // OUTPUT LAYER
    NN_layers[number_of_layers - 1].resize(nn_shape[number_of_layers - 1]);

}
