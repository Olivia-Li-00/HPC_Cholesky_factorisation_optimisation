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

find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_LATEX YES)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_EXTRACT_PRIVATE YES)
    set(DOXYGEN_RECURSIVE YES)
    set(DOXYGEN_QUIET YES)
    doxygen_add_docs(doxygen
        ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include
        COMMENT "Generating documentation"
    )
endif()

add_subdirectory(src)
add_subdirectory(benchmark)
add_subdirectory(test)
add_subdirectory(example)