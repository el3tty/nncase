cmake_minimum_required (VERSION 3.13)

if (BUILDING_RUNTIME)
    # BUILDING_RUNTIME is not used for host builds
    if (NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
        message(FATAL_ERROR "The K230 runtime build requires CMAKE_SYSTEM_PROCESSOR=riscv64 (got '${CMAKE_SYSTEM_PROCESSOR}')")
    endif()

    add_compile_options(-mcpu=c908v -mabi=lp64d -mtune=c908 -mrvv-v0p10-compatible -mrvv-auto-vectorize)
endif()

add_subdirectory(src/Native/src)

if (RISCV64_SYSROOT)
    list(APPEND CMAKE_FIND_ROOT_PATH ${RISCV64_SYSROOT})
    set(Python3_EXECUTABLE ${RISCV64_SYSROOT}/../../bin/python3)

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(Python3_INCLUDE_DIR "${RISCV64_SYSROOT}/usr/include/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
endif()

add_subdirectory(python/nncaseruntime/native)
