# ISA tests of the K230 NPU: k230_isa_test assembles a .s file, sets up the memory, runs the program and checks the
# output. The same tests (asm/) run on two platforms; the platform specific part is run_test() in platform_*.cpp:
#
#   host build (default)   platform_simulator.cpp: the app does not link the simulator, like the host runtime
#                          (runtime/k230-simulator/platform.h) it loads the library (K230_SIMULATOR_DLL=ON) or starts
#                          the executable (K230_SIMULATOR_DLL=OFF)
#   BUILDING_RUNTIME       platform_k230.cpp: runs on a K230 board, memory from libmmz, registers through /dev/mmz
set(ISA_TEST_SOURCES main.cpp test_case.cpp test_runner.cpp assembler.cpp)

if(BUILDING_RUNTIME)
    # the register level driver of the runtime is used as it is (gnne.c is not part of a library the test could link
    # without pulling in the runtime module)
    add_executable(k230_isa_test ${ISA_TEST_SOURCES} platform_k230.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/k230/gnne.c)
    target_include_directories(k230_isa_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_link_libraries(k230_isa_test PRIVATE mmz)
    install(TARGETS k230_isa_test RUNTIME DESTINATION bin)
    add_executable(ai2d_mmio_probe ai2d_mmio_probe.cpp)
    target_include_directories(ai2d_mmio_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/k230)
    target_link_libraries(ai2d_mmio_probe PRIVATE mmz)
    install(TARGETS ai2d_mmio_probe RUNTIME DESTINATION bin)
else()
    add_executable(k230_isa_test ${ISA_TEST_SOURCES} platform_simulator.cpp)
    target_include_directories(k230_isa_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../runtime)
    # nncase.simulator.k230 provides shared_memory
    target_link_libraries(k230_isa_test PRIVATE nncase.simulator.k230)
    # the simulator has to be built, but is not linked
    add_dependencies(k230_isa_test nncase.simulator.k230.sc)

    if(K230_SIMULATOR_DLL)
        target_compile_definitions(k230_isa_test PRIVATE K230_SIMULATOR_DLL)
        if(NOT WIN32)
            target_link_libraries(k230_isa_test PRIVATE ${CMAKE_DL_LIBS})
        endif()
    else()
        target_compile_definitions(k230_isa_test PRIVATE "K230_SIMULATOR_EXE=\"$<TARGET_FILE:nncase.simulator.k230.sc>\"")
    endif()

    # Runs every .s test below asm/ (k230_isa_test accepts a directory): cmake --build . --target run_isa_tests
    # K230_SIMULATOR points the test app to the simulator that was just built (the library is not next to the test app).
    add_custom_target(run_isa_tests
        COMMAND ${CMAKE_COMMAND} -E env K230_SIMULATOR=$<TARGET_FILE:nncase.simulator.k230.sc>
                $<TARGET_FILE:k230_isa_test> ${CMAKE_CURRENT_SOURCE_DIR}/asm
        DEPENDS k230_isa_test
        USES_TERMINAL)
endif()
