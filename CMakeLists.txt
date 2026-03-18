cmake_minimum_required(VERSION 3.26)

project(
    CholeskyFactorisation
    VERSION 0.0
    LANGUAGES CXX
    DESCRIPTION "Cholesky Factorisation Library"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_subdirectory(src)
add_subdirectory(benchmark)
add_subdirectory(test)