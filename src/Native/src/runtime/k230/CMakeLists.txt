cmake_minimum_required(VERSION 3.13)

set(COMMON_SOURCES
    gnne.c
    dynamic_function.device.cpp
    gnne_tile_utils.cpp
    host_allocator.cpp
    mmz_allocator.cpp
    runtime_function.cpp
    runtime_module.cpp
)

add_library(functional_k230 STATIC
    ${COMMON_SOURCES}
    dynamic_gnne_matmul.cpp
)

add_library(nncase.rt_modules.k230 STATIC
    ${COMMON_SOURCES}
    ai2d_builder.cpp
)

function(configure_k230_runtime_target target_name)
    target_compile_definitions(${target_name} PUBLIC -DBUILDING_RUNTIME)
    target_link_libraries(${target_name} PUBLIC gsl::gsl-lite)
    set_property(TARGET ${target_name} PROPERTY POSITION_INDEPENDENT_CODE ON)
    install(TARGETS ${target_name} EXPORT nncaseruntimeTargets)
endfunction()

configure_k230_runtime_target(functional_k230)
configure_k230_runtime_target(nncase.rt_modules.k230)

# mmz is required for python bindings and for ai2d_test, so build it as part of runtime
set(LIBMMZ_REV ea816b267c431edd9a1a80574bded229010b1f15)
set(LIBMMZ_BASE https://raw.githubusercontent.com/kendryte/k230_linux_sdk/${LIBMMZ_REV}/buildroot-overlay/package/libmmz)
FetchContent_Declare(mmz_c URL ${LIBMMZ_BASE}/mmz.c DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_Declare(mmz_h URL ${LIBMMZ_BASE}/mmz.h DOWNLOAD_NO_EXTRACT TRUE)

FetchContent_MakeAvailable(mmz_c mmz_h)

add_library(mmz STATIC ${mmz_c_SOURCE_DIR}/mmz.c)
set_target_properties(mmz PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mmz PUBLIC ${mmz_h_SOURCE_DIR})

add_subdirectory(ai2d_test)
