#pragma once
// Platform independent part of a test run: memory layout, filling the memory, assembling the program and checking
// the result. The platform backends (platform_simulator.cpp, platform_k230.cpp) provide the memory and run the program.
#include "assembler.h"
#include "test_case.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace k230::isa_test {

// Exit codes of run_test() and of the test app.
enum ExitCode {
    kPass = 0,
    kMismatch = 1,      // the output differs from the expected values
    kBadTest = 2,       // bad usage, assembly error, bad test description
    kPlatformFailure = 3, // the simulator / device failed to run the program
};

struct RunOptions {
    bool verbose = false;
};

// Runs one test on the platform this test app was built for (simulator or k230 device) and returns an ExitCode.
// Prints the result ("PASS ..." on stdout, details on stderr). Throws TestError / AsmError for a bad test.
int run_test(TestCase &tc, const RunOptions &options);

constexpr size_t kGlbSize = 0x400000; // 4 MiB of L2 / GLB bank 0
constexpr size_t kInputBase = 0;      // automatic placement of IN tensors in the GLB
constexpr size_t kOutputBase = 0x1000; // automatic placement of OUT tensors in the GLB
constexpr size_t kDdrAlign = 16;

size_t align_to(size_t v, size_t a);

// Where the tensors of a test case live. GLB addresses are plain GLB addresses (bank 0); DDR tensors are placed back to
// back inside two regions (DDR_IN, DDR_OUT) and are given as offsets from the start of their region.
struct Layout {
    std::vector<size_t> in_at, out_at;
    std::vector<size_t> ddr_in_at, ddr_out_at;
    size_t ddr_in_size = 0, ddr_out_size = 0;
};

// Throws TestError if the tensors do not fit.
Layout make_layout(const TestCase &tc);

// The memory as the program sees it. The pointers are in the address space of the test app, the `*_addr` values are
// what the NPU program has to use (DDR offsets in the simulator, physical addresses on the device).
struct MemoryImage {
    uint8_t *glb = nullptr; // kGlbSize bytes
    uint8_t *ddr_in = nullptr;
    uint64_t ddr_in_addr = 0;
    uint8_t *ddr_out = nullptr;
    uint64_t ddr_out_addr = 0;
};

// Copies the inputs into the memory and fills the outputs with 0xA5, so that a missing store is detected.
void init_memory(const TestCase &tc, const Layout &layout, const MemoryImage &mem);

// Compares the outputs; prints up to 16 differing elements. Returns the number of mismatching elements.
int verify_outputs(const TestCase &tc, const Layout &layout, const MemoryImage &mem);

// Symbols the programs may use with `li`: DDR_IN, DDR_OUT (first address of the DDR tensors).
SymbolTable program_symbols(const MemoryImage &mem);

// Assembles the program. With verbose set the machine code listing is printed.
AsmProgram assemble_test(const TestCase &tc, const RunOptions &options, const SymbolTable &symbols);

} // namespace k230::isa_test
