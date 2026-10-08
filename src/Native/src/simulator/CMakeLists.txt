cmake_minimum_required(VERSION 3.10)
project(k230_npu_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -w")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

file(GLOB_RECURSE SIM_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/globals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/isa/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/engines/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/math/*.cpp)

add_executable(sim ${SIM_SOURCES})
target_include_directories(sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sim PRIVATE pthread rt)
