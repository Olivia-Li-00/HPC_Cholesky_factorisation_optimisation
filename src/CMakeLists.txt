add_library(cholesky_factorisation_lib
    STATIC
    cholesky_factorisation.cpp
)

target_include_directories(cholesky_factorisation_lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(cholesky_factorisation_lib
    PUBLIC
    OpenMP::OpenMP_CXX
)