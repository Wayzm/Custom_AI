cmake_minimum_required(VERSION 3.18)
project("Custom AI" CXX)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(MPI REQUIRED)
set(CMAKE_BUILD_TYPE Debug)

add_definitions(-DOMPI_SKIP_MPICXX)
include_directories("include/")
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

add_library(custom_lib STATIC src/nn.cpp src/nn_weight.cpp src/nn_activation.cpp)
add_executable(AI.EXE src/main.cpp)

target_compile_options(custom_lib PRIVATE)
target_compile_options(AI.EXE PRIVATE)

target_link_libraries(custom_lib OpenMP::OpenMP_CXX)
target_link_libraries(custom_lib ${BLAS_LIBRARIES})
target_link_libraries(AI.EXE OpenMP::OpenMP_CXX)
target_link_libraries(AI.EXE custom_lib)
target_link_libraries(AI.EXE MPI::MPI_CXX)
