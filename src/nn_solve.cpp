#include "nn.h"
#include "compute.h"

template <typename T> T nn<T>::get_err(){
    return err;
}

template <typename T> void nn<T>::compute_err(){
    const ui32 last_layer_index = nn_shape.size() - 1;
    err = 0.0;
    /*  MEAN SQUARED ERROR METHOD   */
    for(ui32 i = 0U; i < nn_shape[last_layer_index]; ++i){
        const T diff = NN_layers[last_layer_index][i] - Output[i];
        err += diff * diff;
        NN_layers_D[last_layer_index - 1][i] = diff * last_layer_derivative_activation(NN_layers[last_layer_index][i]);
    }
    err /= nn_shape[last_layer_index];
}

template <typename T> void nn<T>::propagation(){

}
