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

# Windows install layout: every exe, dll and python file goes to
# <prefix>/nncaseruntime, so the exes find the dlls next to them and the
# python module finds them next to the .pyd.
if (WIN32)
    set(NNCASE_WIN_INSTALL_DIR nncaseruntime)
    set(NNCASE_WIN_INSTALL_TARGETS
        nncaseruntime
        nncase.simulator.k230
        nncase.simulator.k230.sc
        nncasetest
        nncasetest_cli
        nncasetest_dynamic_cli
        nncaseruntime_python)

    foreach (_target IN LISTS NNCASE_WIN_INSTALL_TARGETS)
        if (TARGET ${_target})
            install(TARGETS ${_target}
                    RUNTIME DESTINATION ${NNCASE_WIN_INSTALL_DIR}
                    LIBRARY DESTINATION ${NNCASE_WIN_INSTALL_DIR})
            # The subdirectories also install some of these to bin/ or lib/;
            # drop those copies so there is only one set of binaries.
            install(CODE "file(REMOVE \"\${CMAKE_INSTALL_PREFIX}/bin/$<TARGET_FILE_NAME:${_target}>\" \"\${CMAKE_INSTALL_PREFIX}/lib/$<TARGET_FILE_NAME:${_target}>\")")
        endif ()
    endforeach ()

    # nethost.dll comes from the package fetched in src/Native/src
    include(FetchContent)
    FetchContent_GetProperties(nethost)
    if (nethost_SOURCE_DIR)
        install(FILES "${nethost_SOURCE_DIR}/runtimes/win-x64/native/nethost.dll"
                DESTINATION ${NNCASE_WIN_INSTALL_DIR})
    else ()
        message(WARNING "nethost was not fetched; nethost.dll will not be installed")
    endif ()

    install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/python/nncaseruntime/__init__.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/nncaseruntime/base.py
            DESTINATION ${NNCASE_WIN_INSTALL_DIR})
endif ()
