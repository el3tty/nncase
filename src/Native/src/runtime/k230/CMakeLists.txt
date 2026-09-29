cmake_minimum_required(VERSION 3.13)
#project(k230_runtime CXX C)

set(COMMON_SOURCES
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
