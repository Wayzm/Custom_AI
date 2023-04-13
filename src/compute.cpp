#include "compute.h"

template <typename T> compute<T>::compute(){
    temp_vector = {};
    temp_matrix = {};
}

template <typename T> void compute<T>::axpy(const std::vector<T> x,
                                            const T a,
                                            std::vector<T> y){
    /* MEMORY MANANGEMENT */
    assert(x.size() == y.size());
    if(temp_vector.size() < x.size())
        temp_vector.reserve(x.size() - temp_vector.size());
    if(temp_vector.size() > x.size())
        temp_vector.resize(x.size());

    /* AXPY MAIN LOOP */
    #pragma omp parallel for schedule(dynamic, 1)
    for(auto i = 0U; i < x.size(); ++i)
        temp_vector[i] = a * x[i] + y[i];

    y = temp_vector;
}