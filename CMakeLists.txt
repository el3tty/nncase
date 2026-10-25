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

# compiler bindings (_nncase): host only, they start the .NET compiler through nethost
if (NOT BUILDING_RUNTIME)
    add_subdirectory(python/nncase/native)
endif ()

# Install layout (host builds): every exe, shared library and python file goes
# to <prefix>/nncaseruntime, so the exes find the shared libraries next to them
# and the python module finds them next to the extension module.
if (NOT BUILDING_RUNTIME)
    set(NNCASE_INSTALL_DIR nncaseruntime)
    set(NNCASE_INSTALL_TARGETS
        nncaseruntime
        nncase.simulator.k230
        nncase.simulator.k230.sc
        nncasetest_cli
        nncaseruntime_python k230_isa_test k230-as)

    foreach (_target IN LISTS NNCASE_INSTALL_TARGETS)
        if (TARGET ${_target})
            install(TARGETS ${_target}
                    RUNTIME DESTINATION ${NNCASE_INSTALL_DIR}
                    LIBRARY DESTINATION ${NNCASE_INSTALL_DIR})
        endif ()
    endforeach ()

    # nethost.dll / libnethost.so (NETHOST_RUNTIME_FILE is set in src/Native/src)
    if (EXISTS "${NETHOST_RUNTIME_FILE}")
        install(FILES "${NETHOST_RUNTIME_FILE}" DESTINATION ${NNCASE_INSTALL_DIR})
    else ()
        message(WARNING "nethost library not found (${NETHOST_RUNTIME_FILE}); it will not be installed")
    endif ()

    # assembly tests for k230_isa_test
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/Native/src/isa_test/asm/
            DESTINATION ${NNCASE_INSTALL_DIR}/isa_tests)

    install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/python/nncaseruntime/__init__.py
            DESTINATION ${NNCASE_INSTALL_DIR})

    # compiler package: <prefix>/nncase holds __init__.py and _nncase; the runtime libraries it links stay in
    # <prefix>/nncaseruntime (nncase/__init__.py adds that folder to the DLL search path on Windows)
    if (TARGET nncase_python)
        install(TARGETS nncase_python
                RUNTIME DESTINATION nncase
                LIBRARY DESTINATION nncase)
        install(FILES
                ${CMAKE_CURRENT_SOURCE_DIR}/python/nncase/__init__.py
                DESTINATION nncase)
    endif ()
endif ()
