#include "nn_headers.h"
// interfacce
    f64 reciprocal_derivative_tanh(f64 x);
    f64 reciprocal_derivative_sigmoid(f64 x);
    f64 reciprocal_derivative_linear();
    f64 derivative_relu(f64 x);
    f64 activation(f64 x);
    f64 last_layer_activation(f64 x);
    f64 reciprocal_derivative_activation(f64 x);
    f64 last_layer_reciprocal_derivative_activation(f64 x);
    void update_weight();

    const f64 relu_parameter = 0.0;