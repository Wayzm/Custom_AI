#include "xor.h"
#include "nn.h"

#define eps 10e-3

int main(){
    nn<f32> nn({2, 3, 5, 1}, nn.activation_functions::sigmoid);
    XOR<f32> x;

    unsigned long long counter = 0;
    ui32 i;

    do{
        i = rand()%4;
        nn.training_supervised(x.Input_test[i], x.Output_test[i]);
        printf("Error value : %f \r", nn.get_err());
        ++counter;
    }while(nn.get_err() > eps);
    printf("Number of iteration needed : %lld \n", counter);
    return 0;
}