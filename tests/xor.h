#ifndef TEST_XOR
#define TEST_XOR

#include "nn.h"

template <typename T> class XOR {
public:
	/// @brief Constructor
	XOR();

	/// @brief Launch the test with our neural network class
	/// @param nn
	void Test(nn<T> &nn);

    /// @brief Displays the error value at a certain iteration
    /// @param nn
    /// @param iteration
    void show_stats(nn<T> &nn, const ui32 iteration);

    /// @brief Test arrays set in Init()
    std::vector<std::vector<T>> Input_test, Output_test;
private:
	/// @brief Creates the xor table used for testing purposes
	void Init();
};

#endif