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

template <typename T> void compute<T>::emm(const std::vector<T> X,
                                           const ui32 rows_x,
                                           const ui32 cols_x,
                                           const std::vector<T> Y,
                                           const ui32 rows_y,
                                           const ui32 cols_y,
                                           const T a,
                                           std::vector<std::vector<T>> Z,
                                           const ui32 rows_z,
                                           const ui32 cols_z,
                                           const T b){
    /* MEMORY MANAGEMENT */
    assert(cols_x == rows_y);
    assert(rows_z == rows_x && cols_y == cols_z);
    assert(Z != NULL);
    temp_matrix.resize(rows_x * cols_y);

    /* EMM MAIN LOOP */
    T tmp = 0;
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1) private(tmp)
        for(ui32 i = 0; i < rows_x; ++i){
            for(ui32 j = 0; j < cols_y; ++j){
                for(ui32 k = 0; k < cols_x; ++k){
                    tmp += X[i * cols_x + k] * Y[j + cols_y * k];
                }
                temp_matrix[i * cols_y + j] = a * tmp + b * Z[i * cols_y + j];
                tmp = 0;
            }
        }
    }
    Z = temp_matrix;
}

template <typename T> void compute<T>::emv(const std::vector<std::vector<T>> X,
                                           const std::vector<T> y,
                                           const T a,
                                           std::vector<T> z,
                                           const T b){
    /* MEMORY MANAGEMENT */
}