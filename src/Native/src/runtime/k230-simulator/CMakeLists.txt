cmake_minimum_required(VERSION 3.13)
project(nncase_simulator_k230 LANGUAGES CXX)

add_library(nncase.simulator.k230 SHARED
    dynamic_function.cpp
    dynamic_gnne_matmul.cpp
    gmodel_builder.cpp
    k230_runtime_function.cpp
    k230_runtime_module.cpp
    shared_memory.cpp)

target_link_libraries(nncase.simulator.k230 PUBLIC gsl::gsl-lite)
target_link_libraries(nncase.simulator.k230 PUBLIC nncaseruntime)
target_compile_definitions(nncase.simulator.k230 PRIVATE NNCASE_MODULES_K230_DLL)
