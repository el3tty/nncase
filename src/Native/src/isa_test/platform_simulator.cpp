// run_test() for the K230 C model. The test app only assembles the program, sets up the memory objects and starts the
// simulator: as a shared library (K230_SIMULATOR_DLL) or as a separate executable. The simulator location can be
// overridden with the environment variable K230_SIMULATOR (path of the library / executable).
#include "test_runner.h"

#include <nncase/runtime/k230/shared_memory.h>
// run_cmodel(): loads the simulator library (K230_SIMULATOR_DLL) or starts the simulator executable (see platform.h)
#include <k230-simulator/platform.h>

#include <cstdio>
#include <cstring>
#include <random>

using nncase::runtime::k230::shared_memory;
using nncase::runtime::k230::shared_memory_openmode;

namespace k230::isa_test {

namespace {

constexpr size_t kDdrSize = 0x80000000ull;
constexpr size_t kTextOffset = 0x1000; // the program is placed at DDR offset 0x1000
constexpr size_t kDdrInBase = 0x100000;  // DDR_IN region: DDR offsets 0x100000 ..
constexpr size_t kDdrOutBase = 0x200000; // DDR_OUT region: DDR offsets 0x200000 ..

std::string unique_name(const char *tag) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    char buf[64];
    std::snprintf(buf, sizeof buf, "k230isa-%s-%016llx", tag, static_cast<unsigned long long>(rng()));
    return buf;
}

} // namespace

int run_test(TestCase &tc, const RunOptions &options) {
    const Layout layout = make_layout(tc);
    if (layout.ddr_in_size > kDdrOutBase - kDdrInBase)
        throw TestError("DDR input tensors do not fit before the DDR output area");

    // Memory: DDR holds the code and the DDR tensors, ctrl (GLB bank 0) holds the GLB tensors.
    const std::string ddr_name = unique_name("ddr");
    const std::string ctrl_name = unique_name("ctrl");
    shared_memory ddr(ddr_name, kDdrSize, shared_memory_openmode::create);
    shared_memory ctrl(ctrl_name, kGlbSize, shared_memory_openmode::create);
    auto *ddr_base = static_cast<uint8_t *>(ddr.data());

    MemoryImage mem;
    mem.glb = static_cast<uint8_t *>(ctrl.data());
    mem.ddr_in = ddr_base + kDdrInBase;
    mem.ddr_in_addr = kDdrInBase;
    mem.ddr_out = ddr_base + kDdrOutBase;
    mem.ddr_out_addr = kDdrOutBase;

    const AsmProgram prog = assemble_test(tc, options, program_symbols(mem));
    if (prog.text.size() > kDdrInBase - kTextOffset)
        throw TestError("the program is too large");
    std::memcpy(ddr_base + kTextOffset, prog.text.data(), prog.text.size());
    init_memory(tc, layout, mem);

    // The simulator is run exactly as the host runtime runs it: <ddr> <ctrl> <text offset> <dump path>.
    const int rc = nncase::runtime::k230::run_cmodel(ddr_name, ctrl_name, kTextOffset, ".");
    if (tc.expect_fail) {
        if (rc == 0) {
            std::fprintf(stderr, "FAIL %s: the simulator accepted a program that has to fail\n", tc.path.c_str());
            return kMismatch;
        }
        std::printf("PASS %s (the simulator rejected it with %d, as expected)\n", tc.path.c_str(), rc);
        return kPass;
    }
    if (rc != 0) {
        std::fprintf(stderr, "FAIL %s: the simulator returned %d\n", tc.path.c_str(), rc);
        return kPlatformFailure;
    }

    if (const int mismatches = verify_outputs(tc, layout, mem)) {
        std::fprintf(stderr, "FAIL %s: %d element(s) differ\n", tc.path.c_str(), mismatches);
        return kMismatch;
    }
    std::printf("PASS %s (%zu bytes of code)\n", tc.path.c_str(), prog.text.size());
    return kPass;
}

} // namespace k230::isa_test
