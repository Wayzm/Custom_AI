#include "nn_weight.h"

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