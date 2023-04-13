#include "nn.h"

template <typename T> void  nn<T>::set_weight_method(weight_functions selected_method){
    current_weight_method = selected_method;
}

template <typename T> void nn<T>::set_weight_method(weight_functions selected_method, const ui32 seed){
    current_weight_method = selected_method;
    weight_seed = seed;
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