# ISA tests of the K230 simulator: k230_isa_test assembles a .s file, sets up the memory, runs the simulator and checks
# the output. The test app does not link the simulator: like the host runtime (runtime/k230-simulator/platform.h) it
# loads the library (K230_SIMULATOR_DLL=ON) or starts the executable (K230_SIMULATOR_DLL=OFF).
add_executable(k230_isa_test test_main.cpp assembler.cpp)
target_include_directories(k230_isa_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime)
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

# Runs every asm/*.s test: cmake --build . --target run_isa_tests
# K230_SIMULATOR points the test app to the simulator that was just built (the library is not next to the test app).
add_custom_target(run_isa_tests
    COMMAND ${CMAKE_COMMAND} -E env K230_SIMULATOR=$<TARGET_FILE:nncase.simulator.k230.sc>
            ${CMAKE_COMMAND} -DTEST_EXE=$<TARGET_FILE:k230_isa_test> -DTEST_DIR=${CMAKE_CURRENT_SOURCE_DIR}/asm
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.cmake
    DEPENDS k230_isa_test
    USES_TERMINAL)
