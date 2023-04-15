#ifndef TEST_XOR
#define TEST_XOR

#include "nn.h"

template <typename T> class XOR {
public:
	/// @brief Constructor
	XOR();

	/// @brief Launch the test with our neural network class
	/// @param nn
	void Test(nn<T> &nn){
    	Init();
	}

    /// @brief Displays the error value at a certain iteration
    /// @param nn
    /// @param iteration
    void show_stats(nn<T> &nn, const ui32 iteration);

    /// @brief Test arrays set in Init()
    std::vector<std::vector<T>> Input_test, Output_test;
private:
	/// @brief Creates the xor table used for testing purposes
	void Init(){
    	Input_test.resize(4);
    	Output_test.resize(4);
    	for(ui32 i = 0U; i < 4; ++i){
        	Input_test[i].resize(2);
        	Output_test[i].resize(1);
    	}

    	Input_test[0][0]  = 0;
    	Input_test[0][1]  = 0;
    	Output_test[0][0] = 0;

    	Input_test[1][0]  = 1;
    	Input_test[1][1]  = 1;
    	Output_test[1][0] = 0;

    	Input_test[2][0]  = 0;
    	Input_test[2][1]  = 1;
    	Output_test[2][0] = 1;

    	Input_test[3][0]  = 1;
    	Input_test[3][1]  = 0;
    	Output_test[3][0] = 1;

	}
};

#endif