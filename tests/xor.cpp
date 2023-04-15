#include "xor.h"

template <typename T> void XOR<T>::show_stats(nn<T> &nn, const ui32 iteration){
    std::cout << "erreur = " << nn.get_err() << std::endl;
    std::cout << "nombre d'iteration : " << iteration << std::endl;
}

template <typename T> void XOR<T>::Test(nn<T> &nn){
    std::cout << "Entrez 2 chiffres 0 ou 1 : " << std::endl << "--> ";

    std::vector<T> test_value_double(2);
    std::vector<ui32> test_value_int(2);

    for(ui32 j = 0U; j < 2; ++j){
        std::cin >> test_value_int[j];
        test_value_double[j] = static_cast<T>(test_value_int[j]);
    }

    auto result = nn.compute_output(test_value_double);

    T exact_result;
    if(test_value_int[0] == test_value_int[1]){
        exact_result = 0.0;
    }
    else{
        exact_result = 1.0;
    }

    std::vector<std::vector<T>> exact_result_v;
    exact_result_v.resize(1);
    exact_result_v[0].push_back(exact_result);
    auto error_value_while_testing = nn.get_err();

    std::cout << "Erreur du test : "<< error_value_while_testing<<std::endl;
    std::cout << "result du réseau de neurones : " << result.data() << std::endl;
    std::cout << "exact result : " << exact_result << std::endl;
}

template class XOR<f32>;
template class XOR<f64>;