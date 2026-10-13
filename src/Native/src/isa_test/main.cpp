// Test runner for the K230 NPU ISA tests: assembles one .s file, runs it on the simulator or on a K230 board
// (depending on how the app was built) and verifies the output.
//
//   k230_isa_test [-v] <program.s | directory>
//
// With a directory, every *.s file below it is run (sorted by path), each in a separate process (the simulator keeps
// global state and a misbehaving program must not take the other tests down); a summary is printed and the exit code is
// the worst per-test code (or 1 if nothing was found).
//
// Test description (comments in the assembly file):
//
//   # IN:  [3, 3]            input tensor, one line per tensor, optional element type before the list
//   # IN:  u8 [128, 255]     types: i8 u8 i16 u16 i32 (default) u32 f32 f16
//   # OUT: [9]               expected output tensor(s), same syntax
//   # EXPECT_FAIL            the program has to be rejected (non-zero return value, e.g. `intr`, a missing `end`, a bad
//                            configuration); the outputs are not checked. Faults that kill the host process
//                            (division by zero) cannot be tested this way.
//   # IN @0x4000: [1, 2]     optional explicit GLB address instead of the automatic layout
//   # DDR_IN:  u8 [1, 2, 3]  tensors placed in DDR (for l2_load); fp16 is available as 'f16'
//   # DDR_OUT: u8 [1, 2, 3]  expected DDR contents after the run (for l2_store)
//
// Memory seen by the program:
//
//   GLB (L2, bank 0, addresses are plain GLB addresses)
//     0x0000 ..  input tensors, back to back, each aligned to 4 bytes
//     0x1000 ..  output tensors, back to back, each aligned to 4 bytes (pre-filled with 0xA5)
//   DDR: the addresses of the tensors are not known when the test is written (they are DDR offsets in the simulator
//   and physical addresses on the device), so the program gets them through the symbols DDR_IN and DDR_OUT
//   (`li x1, DDR_IN+16`). Inside a region the tensors are back to back, each aligned to 16 bytes; DDR_OUT is pre-filled
//   with 0xA5.
//
// The program has to finish with `end`.
//
// Exit code: 0 = pass, 1 = output mismatch, 2 = bad usage / assembly / test description error, 3 = simulator / device failure.
#include "test_runner.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace k230::isa_test;

namespace {

std::string quote(const std::string &s) { return "\"" + s + "\""; }


int run_directory(const char *self, const std::string &dir, const std::vector<std::string> &forward) {
    namespace fs = std::filesystem;
    std::vector<std::string> files;
    for (const auto &e : fs::recursive_directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".s")
            files.push_back(e.path().generic_string());
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        std::fprintf(stderr, "%s: no .s files found\n", dir.c_str());
        return 1;
    }

    int worst = 0, failed = 0;
    for (const auto &file : files) {
        std::string cmd = quote(self) + " ";
        for (const auto &arg : forward)
            cmd += quote(arg) + " ";
        cmd += quote(file);
#ifdef _WIN32
        cmd = quote(cmd); // cmd.exe /c strips the outer quotes
#endif
        std::fflush(stdout);
        std::fflush(stderr);
        int rc = std::system(cmd.c_str());
#ifndef _WIN32
        rc = (rc >= 0 && (rc & 0x7f) == 0) ? ((rc >> 8) & 0xff) : 3; // killed by a signal -> simulator failure
#endif
        if (rc != 0) {
            ++failed;
            worst = std::max(worst, rc);
        }
    }
    std::printf("\n%zu test(s), %zu passed, %d failed\n", files.size(), files.size() - failed, failed);
    return worst;
}

int run_file(const std::string &path, const RunOptions &options) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw TestError("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    TestCase tc = parse_test_case(ss.str());
    tc.path = path;
    return run_test(tc, options);
}

} // namespace

int main(int argc, char **argv) {
    std::string path;
    RunOptions options;
    std::vector<std::string> forward; // options handed on to the per-test processes of a directory run
    bool usage_error = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-v") == 0) {
            options.verbose = true;
            forward.push_back("-v");
        } else if (path.empty())
            path = argv[i];
        else
            usage_error = true;
    }
    if (path.empty() || usage_error) {
        std::fprintf(stderr, "usage: %s [-v] <program.s | directory>\n", argv[0]);
        return kBadTest;
    }
    try {
        if (std::filesystem::is_directory(path))
            return run_directory(argv[0], path, forward);
        return run_file(path, options);
    } catch (const AsmError &e) {
        std::fprintf(stderr, "%s:%s\n", path.c_str(), e.what());
        return kBadTest;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s: %s\n", path.c_str(), e.what());
        return kBadTest;
    }
}
