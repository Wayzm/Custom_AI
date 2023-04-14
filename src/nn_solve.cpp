#include "nn.h"
#include "compute.h"



template <typename T> void nn<T>::propagation(){
    const ui32 last_layer_index = nn_shape.size() - 1;
    compute<T> c;
    for(ui32 i = 0U; i < last_layer_index - 1; ++i){
        // Propagation to the next layer with the weight's influence
        c.emm(Weight_Matrix[i], nn_shape[i + 1], nn_shape[i],
              NN_layers[i], nn_shape[i], 1,
              1.0,
              NN_layers[i + 1], nn_shape[i + 1], 1,
              1.0);
        // Activation on each neuron of the next layer
        #pragma omp parallel for schedule(dynamic, 1)
        for(ui32 j = 0U; j < nn_shape[i + 1]; ++j){
            NN_layers[i + 1][j] = activation(NN_layers[i + 1][j] + Bias_Matrix[i][j]);
        }
    }

    c.emm(Weight_Matrix[last_layer_index - 1], nn_shape[last_layer_index], nn_shape[last_layer_index - 1],
          NN_layers[last_layer_index - 1], nn_shape[last_layer_index - 1], 1,
          1.0,
          NN_layers[last_layer_index], nn_shape[last_layer_index], 1.0,
          1.0);

    #pragma omp parallel for schedule(dynamic, 1)
    for(ui32 i = 0U; i < nn_shape[last_layer_index]; ++i)
        NN_layers[last_layer_activation][i] = last_layer_activation(NN_layers[last_layer_activation][i] + Bias_Matrix[last_layer_index - 1][i]);
}

template <typename T> void nn<T>::backpropagation(){
    const ui32 last_layer_index = nn_shape.size() - 1;
    compute<T> c;
    for(ui32 i = last_layer_index - 1; i > 0U; --i){
        c.emm(Weight_Matrix[i], nn_shape[i], nn_shape[i + 1],
              NN_layers_D[i], nn_shape[i + 1], 1,
              1.0,
              NN_layers_D[i - 1], nn_shape[i], 1,
              1.0);
        #pragma omp parallel for schedule(dynamic, 1)
        for(ui32 j = 0U; j < nn_shape[i]; ++j)
            NN_layers_D[i - 1][j] *= derivative_activation(NN_layers[i][j]);
    }
}
