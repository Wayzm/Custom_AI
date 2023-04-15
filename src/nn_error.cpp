#include "nn.h"

template <class T> T nn<T>::get_err(){
    return err;
}

template <class T>void nn<T>::set_loss_method(loss_functions selected_method){
    current_loss_method = selected_method;
}

template <class T> void nn<T>::mean_squared_error(){
    const ui32 last_layer_index = nn_shape.size() - 1;
    err = 0.0;
    /*  MEAN SQUARED ERROR METHOD   */
    for(ui32 i = 0U; i < nn_shape[last_layer_index]; ++i){
        const T diff = Output[i] - NN_layers[last_layer_index][i];
        err += diff * diff;
        NN_layers_D[last_layer_index - 1][i] = diff * last_layer_derivative_activation(NN_layers[last_layer_index][i]);
    }
    err /= nn_shape[last_layer_index];
}

template <class T> void nn<T>::mean_absolute_error(){
    const ui32 last_layer_index = nn_shape.size() - 1;
    err = 0.0;
    /*  MEAN ABSOLUTE ERROR METHOD  */
    for(ui32 i = 0U; i < nn_shape[last_layer_index; ++i]){
        const T diff = Output[i] - NN_layers[last_layer_index][i];
        err += std::fabs(diff);
        NN_layers_D[last_layer_index - 1][i] = diff * last_layer_derivative_activation(NN_layers[last_layer_index][i]);
    }
    err /= nn_shape[last_layer_index];
}

template <class T> void nn<T>::mean_bias_error(){
    const ui32 last_layer_index = nn_shape.size() - 1;
    err = 0.0;
    /*  MEAN BIAS ERROR METHOD  */
    for(ui32 i = 0U; i < nn_shape[last_layer_index; ++i]){
        const T diff = Output[i] - NN_layers[last_layer_index][i];
        err += diff;
        NN_layers_D[last_layer_index - 1][i] = diff * last_layer_derivative_activation(NN_layers[last_layer_index][i]);
    }
    err /= nn_shape[last_layer_index];
}

template <class T> void nn<T>::cross_entropy_error(){
    const ui32 last_layer_index = nn_shape.size() - 1;
    err = 0.0;
    /*  CROSS ENTROPY ERROR METHOD  */
    for(ui32 i = 0U; i < nn_shape[last_layer_index; ++i]){
        const T diff = Output[i] * std::log(std::fabs(NN_layers[last_layer_index][i])) + (1 - Output[i]) * std::log(std::fabs(1 - NN_layers[last_layer_index][i]));
        err += diff;
        NN_layers_D[last_layer_index - 1][i] = diff * last_layer_derivative_activation(NN_layers[last_layer_index][i]);
    }
    err /= nn_shape[last_layer_index];
}

template <class T> void nn<T>::hinge_error(){
    const ui32 last_layer_index = nn_shape.size() - 1;
    err = 0.0;
    /*  HINGE ERROR METHOD  */
    for(ui32 i = 0U; i < nn_shape[last_layer_index; ++i]){
        const T diff = std::max(0, 1 - Output[i] * NN_layers[last_layer_index][i]);
        err += diff;
        NN_layers_D[last_layer_index - 1][i] = diff * last_layer_derivative_activation(NN_layers[last_layer_index][i]);
    }
    err /= nn_shape[last_layer_index];
}

template <class T> void nn<T>::compute_error(){
    switch(current_loss_method){
    case loss_functions::mean_squared_error:
        mean_squared_error();
    case loss_functions::mean_absolute_error:
        mean_absolute_error();
    case loss_functions::mean_bias_error:
        mean_bias_error();
    case loss_functions::hinge_error:
        hinge_error();
    case loss_functions::cross_entropy_error:
        cross_entropy_error();
    default:
        std::cerr << "Loss function unknown." << std::endl;
        break;
    }
}

template class nn<f32>;
template class nn<f64>;