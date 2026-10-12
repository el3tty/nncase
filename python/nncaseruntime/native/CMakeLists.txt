cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
set(SRCS ffi.cpp)

if (WIN32)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_rc)
    if (_pybind11_rc EQUAL 0)
        file(TO_CMAKE_PATH "${_pybind11_cmakedir}" pybind11_DIR)
    endif()
endif()
find_package(pybind11 REQUIRED)

pybind11_add_module(nncaseruntime_python ${SRCS})
target_link_libraries(nncaseruntime_python PRIVATE nlohmann_json::nlohmann_json)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "riscv64")
    set_target_properties(nncaseruntime_python PROPERTIES
        PREFIX ""
        SUFFIX ".cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}-riscv64-linux-gnu.so")
    target_link_libraries(nncaseruntime_python PRIVATE nncaseruntime functional_k230 nncase.rt_modules.k230 mmz)
else()
    target_link_libraries(nncaseruntime_python PRIVATE nncaseruntime nncase.simulator.k230)
    target_compile_definitions(nncaseruntime_python PRIVATE $<$<CONFIG:Debug>:Py_DEBUG>)
endif()
target_include_directories(nncaseruntime_python PRIVATE
                  ${CMAKE_CURRENT_SOURCE_DIR}/../../common
                  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/Native/include
                  ${NNCASE_GENERATED_INCLUDE_DIR})
set_target_properties(nncaseruntime_python PROPERTIES
                                 OUTPUT_NAME _nncaseruntime)
install(TARGETS nncaseruntime_python DESTINATION lib)
