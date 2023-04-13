cmake_minimum_required(VERSION 3.18)
project("Custom AI" C)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_library(MATH_LIBRARY REQUIRED)
find_package(MPI REQUIRED)
set(CMAKE_BUILD_TYPE Debug)

add_definitions(-DOMPI_SKIP_MPICXX)
add_library(lib_nn_cpu STATIC )
add_library(lib_nn_gpu STATIC )

add_link_options(-fsanitize=address -fsanitize=undefined)
add_compile_options(
                      -Wall -Wextra
                      -pedantic
                      -O3
                      -fno-omit-frame-pointer
                      -g
                      -lm
                      -march=native -mtune=native -mavx2
                      -fopenmp
                      -flto
                      -fsanitize=address -fsanitize=undefined
                      -finline-functions -funroll-loops
                      -ftree-loop-vectorize -ftree-vectorize
                      -fopt-info-all=miram.gcc.optrpt
                      )
