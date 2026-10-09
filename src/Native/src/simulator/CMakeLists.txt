cmake_minimum_required(VERSION 3.13)
project(k230_npu_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(K230_SIMULATOR_DLL "Build the simulator as a shared library exporting SimulatorMain (for debugging)" OFF)

file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS main.cpp globals.cpp isa/*.cpp engines/*.cpp math/*.cpp)

if(K230_SIMULATOR_DLL)
    add_library(nncase.simulator.k230.sc SHARED ${SRCS})
    target_compile_definitions(nncase.simulator.k230.sc PRIVATE K230_SIMULATOR_DLL)
else()
    add_executable(nncase.simulator.k230.sc ${SRCS})
endif()

target_include_directories(nncase.simulator.k230.sc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# shared_memory comes from the runtime library (also brings the nncase and gsl include paths)
target_link_libraries(nncase.simulator.k230.sc PRIVATE nncase.simulator.k230)
