#ifndef NN_HEADER
#define NN_HEADER

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>
#include <ctime>
#include <cassert>
#include <random>

extern "C" {
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <cblas.h>
}

#define ui32 unsigned int
#define ui64 unsigned long int
#define f32 float
#define f64 double

#define cblas_gemm cblas_dgemm
#define cblas_gemv cblas_dgemv
#define cblas_axpy cblas_daxpy
#define cblas_asum cblas_dasum

#endif