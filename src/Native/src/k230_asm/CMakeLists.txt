# K230 NPU assembler: the k230_asm library (also used by the ISA tests, ../isa_test) and the k230-as tool, which turns
# a program (e.g. a compiler dump written one instruction per line) into the machine code of a K230 function.
add_library(k230_asm STATIC assembler.cpp)
target_include_directories(k230_asm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(k230-as k230_as.cpp)
target_link_libraries(k230-as PRIVATE k230_asm)

# k230-as needs no nncase or simulator library, so it goes to bin/ in host builds as well
install(TARGETS k230-as RUNTIME DESTINATION bin)
