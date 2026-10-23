cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
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

set(SRCS ffi.cpp)

pybind11_add_module(nncase_python ${SRCS})
target_link_libraries(nncase_python PRIVATE nncaseruntime nlohmann_json::nlohmann_json)

# Same Python linkage as nncaseruntime_python (see python/nncaseruntime/native/CMakeLists.txt): release Python in
# every configuration unless NNCASE_PYTHON_DEBUG=ON.
if (WIN32)
    if (NNCASE_PYTHON_DEBUG)
        target_compile_definitions(nncase_python PRIVATE Py_DEBUG)
        target_link_libraries(nncase_python PRIVATE "${Python3_LIBRARY_DEBUG}")
        get_target_property(_nncase_py_suffix nncase_python SUFFIX)
        set_target_properties(nncase_python PROPERTIES SUFFIX "_d${_nncase_py_suffix}")
    elseif (TARGET Python3::Module)
        set_target_properties(Python3::Module PROPERTIES
            MAP_IMPORTED_CONFIG_DEBUG Release
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
            MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    endif()
endif()
target_include_directories(nncase_python PRIVATE
                  ${CMAKE_CURRENT_SOURCE_DIR}/../../common
                  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/Native/include
                  ${NNCASE_GENERATED_INCLUDE_DIR})
set_target_properties(nncase_python PROPERTIES
                                    OUTPUT_NAME _nncase
                                    INSTALL_RPATH "$ORIGIN/../nncaseruntime")
# Installed into <prefix>/nncaseruntime from the root CMakeLists.txt.
