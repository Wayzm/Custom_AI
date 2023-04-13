#include "nn.h"
#include "nn_weight.h"

template <typename T> NN_weight<T>::NN_weight(){
    init_weight(weight_seed, 0, 1);
}

template <typename T> void NN_weight<T>::set_weight_method(weight_functions selected_method){
    current_weight_method = selected_method;
}

template <typename T> void NN_weight<T>::init_weight(const ui32 seed, const T min, const T max){
    assert(min <= max);

    // Used to generate a random number engine
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<T> dis(min, max);

    // Default behavior
    if(seed == 1337){
        std::seed_seq seeds({rd(), rd(), rd(), rd()});
        gen.seed(seeds);
    }
    else{
        gen.seed(seed);
    }

    // Random initialisation for each node
    for(auto &element : Weight_Matrix)
        element = dis(gen);
    for(auto &element : Weight_Matrix_D)
        element = 0;
    for(auto &element : Bias_Matrix)
        element = dis(gen);
    for(auto &element : Bias_Matrix_D)
        element = 0;
}

template <typename T> void NN_weight<T>::update_weight(){
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