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
endif()

# By default the module is built for the release Python (python.exe / python312.dll) in every configuration, also
# Debug: pybind11 hides _DEBUG from Python.h, and the debug import library of FindPython is not used. Py_DEBUG would
# link python312_d.dll, and a release interpreter that loads such a module aborts ("the GIL is released").
# NNCASE_PYTHON_DEBUG=ON builds it for the debug interpreter (python_d.exe) instead, which only imports modules
# named <name>_d.<tag>.pyd and needs every other extension module it imports (numpy ...) as a debug build too.
option(NNCASE_PYTHON_DEBUG "Build the python module for the debug Python interpreter (python_d.exe)" OFF)
if (WIN32)
    if (NNCASE_PYTHON_DEBUG)
        target_compile_definitions(nncaseruntime_python PRIVATE Py_DEBUG)
        target_link_libraries(nncaseruntime_python PRIVATE "${Python3_LIBRARY_DEBUG}")
        get_target_property(_nncase_py_suffix nncaseruntime_python SUFFIX)
        set_target_properties(nncaseruntime_python PROPERTIES SUFFIX "_d${_nncase_py_suffix}")
    elseif (TARGET Python3::Module)
        set_target_properties(Python3::Module PROPERTIES
            MAP_IMPORTED_CONFIG_DEBUG Release
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
            MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    endif()
endif()
target_include_directories(nncaseruntime_python PRIVATE
                  ${CMAKE_CURRENT_SOURCE_DIR}/../../common
                  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/Native/include
                  ${NNCASE_GENERATED_INCLUDE_DIR})
set_target_properties(nncaseruntime_python PROPERTIES
                                 OUTPUT_NAME _nncaseruntime)
# The target is always built. Host builds install it into <prefix>/nncaseruntime from the root CMakeLists.txt,
# so this lib/ install is only for runtime (cross) builds.
if (BUILDING_RUNTIME)
    install(TARGETS nncaseruntime_python DESTINATION lib)
endif ()
