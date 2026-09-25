cmake_minimum_required (VERSION 3.13)

include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.1.1 # или 9.1.0
)

FetchContent_Declare(
    gsl-lite
    GIT_REPOSITORY https://github.com/gsl-lite/gsl-lite.git
    GIT_TAG        v0.41.0
)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)

FetchContent_MakeAvailable(fmt gsl-lite json)

FetchContent_Declare(
    nethost
    URL      "https://www.nuget.org/api/v2/package/runtime.win-x64.Microsoft.NETCore.DotNetAppHost/8.0.0"
)

FetchContent_Populate(nethost)

add_library(nethost::nethost UNKNOWN IMPORTED)
set(NETHOST_NATIVE_DIR "${nethost_SOURCE_DIR}/runtimes/win-x64/native")
set_target_properties(nethost::nethost PROPERTIES
        IMPORTED_LOCATION "${NETHOST_NATIVE_DIR}/nethost.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${NETHOST_NATIVE_DIR}")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../Native/include")
if (MSVC)
    add_compile_options(/std:c++20 /wd4251)
else()
    add_compile_options(-std=c++20)
endif()

if (WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

set(SRCS 
        api.cpp
        object.cpp
        shape.cpp
        tensor.cpp
        type.cpp
        value.cpp)

add_library(nncasebase OBJECT ${SRCS})
if(BUILDING_RUNTIME)
    target_compile_definitions(nncasebase PUBLIC -DNNCASE_DLL)
else()
    target_compile_definitions(nncasebase PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
endif()

if (DEFAULT_BUILTIN_RUNTIMES)
    target_compile_definitions(nncasebase PRIVATE -DNNCASE_DEFAULT_BUILTIN_RUNTIMES)
endif ()
set_property(TARGET nncasebase PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(nncasebase PUBLIC gsl::gsl-lite)


add_subdirectory(compiler)
add_subdirectory(runtime)
add_subdirectory(kernels)

if (NOT BUILDING_RUNTIME)
    add_executable(nncasetest test.cpp)
    target_link_libraries(nncasetest PRIVATE nncaseruntime)
    target_link_libraries(nncasetest PRIVATE nlohmann_json::nlohmann_json)
    if (${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo" AND MSVC)
        set_target_properties(nncasetest PROPERTIES LINK_FLAGS "/PROFILE")
    endif()

    add_executable(nncasetest_cli test_cli.cpp)
    target_link_libraries(nncasetest_cli PRIVATE nncaseruntime)

    add_executable(nncasetest_dynamic_cli test_dynamic_cli.cpp)
    target_link_libraries(nncasetest_dynamic_cli PRIVATE nncaseruntime)
endif()

if(BUILDING_RUNTIME)
    install(TARGETS nncasebase EXPORT nncaseruntimeTargets)
else()
    install(TARGETS nncasebase EXPORT nncaseTargets
            COMPONENT nncase-runtime
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
            INCLUDES DESTINATION include
            )

    install(EXPORT nncaseTargets DESTINATION lib/cmake/nncase)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/nncaseConfig.cmake.in nncaseConfig.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nncaseConfig.cmake DESTINATION lib/cmake/nncase)
endif()