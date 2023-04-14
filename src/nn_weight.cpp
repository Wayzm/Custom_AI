#include "nn.h"
#include "compute.h"

template <typename T> void  nn<T>::set_weight_method(weight_functions selected_method){
    current_weight_method = selected_method;
}

template <typename T> void nn<T>::set_weight_method(weight_functions selected_method, const ui32 seed){
    current_weight_method = selected_method;
    weight_seed = seed;
}

template <typename T> void nn<T>::set_learning_rate(T rate){
    learning_rate = rate;
}

template <typename T> void nn<T>::set_learning_rate_inertie(T rate){
    learning_rate_inertie = rate;
}

template <typename T> void nn<T>::init_weight(const T min, const T max){
    assert(min <= max);

    // Used to generate a random number engine
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<T> dis(min, max);

    // Default behavior
    if(weight_seed == 1337){
        std::seed_seq seeds({rd(), rd(), rd(), rd()});
        gen.seed(seeds);
    }
    else{
        gen.seed(weight_seed);
    }

    // Random initialisation for each node
    for(auto &vector_w : Weight_Matrix)
        for(auto &element : vector_w)
            element = dis(gen);
    for(auto &vector_w : Weight_Matrix_D)
        for(auto &element : vector_w)
            element = 0;
    for(auto &vector_b : Bias_Matrix)
        for(auto &element : vector_b)
            element = dis(gen);
    for(auto &vector_b : Bias_Matrix_D)
        for(auto &element : vector_b)
            element = 0;
}

template <typename T> void nn<T>::inertie(){
    /* CONSTRUCTION OF BLAS APPLICATION */
    compute<T> c;
    /* MAIN LOOP */
    const auto last_layer_index = nn_shape.size() - 1;
    #pragma omp parallel for schedule(dynamic, 1)
    for(ui32 i = 0U; i < last_layer_index; ++i){
        const std::vector<T> Old_bias_vector = Bias_Matrix[i];
        const std::vector<T> Old_weight_vector = Weight_Matrix[i];

        /*  UPDATE BIAS VALUES  */
        const auto tmp = NN_layers_D[i];
        for(ui32 j = 0U; j < nn_shape[i+1]; ++j){
            Bias_Matrix[i][j] -= learning_rate * learning_rate_inertie * tmp[j];
            Bias_Matrix[i][j] += (1 - learning_rate_inertie) * Bias_Matrix_D[i][j];
            Bias_Matrix_D[i][j] = Bias_Matrix[i][j] - Old_bias_vector[i];
        }

        for(ui32 j = 0U; j < Weight_Matrix[i].size(); ++j){
            Weight_Matrix[i][j] *= (1 - learning_rate_inertie);
        }
        /*      WEIGHT MODIFICATION     */
        c.emm(NN_layers_D[i], nn_shape[i + 1], 1,
              NN_layers[i], 1, nn_shape[i],
              (1 - learning_rate_inertie),
              Weight_Matrix[i], nn_shape[i + 1], nn_shape[i], 1.0);
        Weight_Matrix_D[i] = Weight_Matrix[i];
        for(ui32 j = 0U; j < Old_weight_vector.size(); ++j)
            Weight_Matrix_D[i][j] -= Old_weight_vector[j];
    }
}

template <typename T> void nn<T>::standard(){
    /* CONSTRUCTION OF BLAS APPLICATION */
    compute<T> c;
    /* MAIN LOOP */
    const auto last_layer_index = nn_shape.size() - 1;
    #pragma omp parallel for schedule(dynamic, 1)
    for(ui32 i = 0U; i < last_layer_index; ++i){
        /*  UPDATE BIAS VALUES  */
        const auto tmp = NN_layers_D[i];
        for(ui32 j = 0U; j < nn_shape[i+1]; ++j){
            Bias_Matrix[i][j] -= learning_rate * tmp[j];
        }

        /*      WEIGHT MODIFICATION     */
        c.emm(NN_layers_D[i], nn_shape[i + 1], 1,
              NN_layers[i], 1, nn_shape[i],
              (1 - learning_rate_inertie),
              Weight_Matrix[i], nn_shape[i + 1], nn_shape[i], 1.0);
    }
}

template <typename T> void nn<T>::update_weight(){
  switch(current_weight_method){
    case weight_functions::inertie:
      inertie();
      break;
    case weight_functions::standard:
      standard();
      break;
    default:
      std::cerr<<"Weight update method unknown."<<std::endl;
      break;
  }
}
