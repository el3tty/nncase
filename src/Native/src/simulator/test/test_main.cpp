// Test runner for the K230 simulator: assembles one .s file, runs it in the simulator and verifies the output.
//
//   k230_isa_test [-v] <program.s | directory>
//
// With a directory, every *.s file below it is run (sorted by path), each in a separate process because the simulator
// keeps global state; a summary is printed and the exit code is the worst per-test code (or 1 if nothing was found).
//
// Test description (comments in the assembly file):
//
//   # IN:  [3, 3]            input tensor, one line per tensor, optional element type before the list
//   # IN:  u8 [128, 255]     types: i8 u8 i16 u16 i32 (default) u32 f32 f16
//   # OUT: [9]               expected output tensor(s), same syntax
//   # EXPECT_FAIL            the simulator has to reject the program (non-zero return value, e.g. `intr`, a missing `end`,
//                            a bad configuration); the outputs are not checked. Faults that kill the host process
//                            (division by zero) cannot be tested this way.
//   # IN @0x4000: [1, 2]     optional explicit address (GLB address in bank 0 / DDR offset) instead of the automatic layout
//   # DDR_IN:  u8 [1, 2, 3]  tensors placed in DDR (for l2_load); fp16 is available as 'f16'
//   # DDR_OUT: u8 [1, 2, 3]  expected DDR contents after the run (for l2_store)
//
// Memory layout seen by the program (GLB bank 0 = the "ctrl" shared memory, addresses are plain GLB addresses):
//
//   0x0000 .. : input tensors, back to back, each aligned to 4 bytes
//   0x1000 .. : output tensors, back to back, each aligned to 4 bytes (pre-filled with 0xA5)
//
// DDR (offsets as used by l2_load / l2_store; the registers hold plain offsets, not physical addresses):
//
//   0x100000 .. : DDR_IN tensors, each aligned to 16 bytes
//   0x200000 .. : DDR_OUT tensors, each aligned to 16 bytes (pre-filled with 0xA5)
//
// The program text is placed at DDR offset 0x1000 and has to finish with `end`.
//
// The test app only assembles the program, sets up the memory objects and starts the simulator: as a shared library
// (K230_SIMULATOR_DLL) or as a separate executable. The simulator location can be overridden with the environment
// variable K230_SIMULATOR (path of the library / executable).
//
// Exit code: 0 = pass, 1 = output mismatch, 2 = bad usage / assembly / test description error, 3 = simulator failure.
#include "assembler.h"

#include <nncase/runtime/k230/shared_memory.h>
// run_cmodel(): loads the simulator library (K230_SIMULATOR_DLL) or starts the simulator executable (see platform.h)
#include <k230-simulator/platform.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace k230::isa_test;
using nncase::runtime::k230::shared_memory;
using nncase::runtime::k230::shared_memory_openmode;

namespace {

constexpr size_t kDdrSize = 0x80000000ull;
constexpr size_t kCtrlSize = 0x400000;
constexpr size_t kTextOffset = 0x1000;
constexpr size_t kInputBase = 0;
constexpr size_t kOutputBase = 0x1000;
constexpr size_t kDdrInBase = 0x100000;
constexpr size_t kDdrOutBase = 0x200000;
constexpr size_t kDdrAlign = 16;

struct TestError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class ElemType { I8, U8, I16, U16, I32, U32, F32, F16 };

struct TypeInfo {
    const char *name;
    ElemType type;
    size_t size;
};

constexpr TypeInfo kTypes[] = {
    {"i8", ElemType::I8, 1}, {"u8", ElemType::U8, 1}, {"i16", ElemType::I16, 2}, {"u16", ElemType::U16, 2},
    {"i32", ElemType::I32, 4}, {"u32", ElemType::U32, 4}, {"f32", ElemType::F32, 4}, {"f16", ElemType::F16, 2},
};

struct Tensor {
    const TypeInfo *type;
    size_t count;
    std::vector<uint8_t> bytes;
    int line;
    int64_t addr = -1; // explicit address or -1 = automatic layout
};

// float -> IEEE half, round to nearest even
uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint16_t sign = uint16_t((x >> 16) & 0x8000);
    const uint32_t abs = x & 0x7fffffff;
    if (abs >= 0x7f800000) // inf / nan
        return uint16_t(sign | 0x7c00 | (abs > 0x7f800000 ? 0x200 : 0));
    if (abs >= 0x477ff000) // rounds to or beyond 65536
        return uint16_t(sign | 0x7c00);
    if (abs < 0x33000001) // rounds to zero
        return sign;
    const uint32_t exp = abs >> 23;
    const uint32_t man = (abs & 0x7fffff) | 0x800000;
    uint32_t r;
    int shift;
    if (exp < 113) { // subnormal half
        shift = int(126 - exp);
        r = 0;
    } else { // normal: the exponent field is added below, the implicit bit is dropped
        shift = 13;
        r = (exp - 113) << 10;
    }
    const uint32_t rem = man & ((1u << shift) - 1);
    r += man >> shift; // for normals the implicit bit adds exactly 1 << 10, i.e. the +1 of the biased exponent
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (r & 1)))
        ++r;
    return uint16_t(sign | r);
}

float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, x;
    if (exp == 0x1f)
        x = sign | 0x7f800000 | (man << 13);
    else if (exp != 0)
        x = sign | ((exp + 112) << 23) | (man << 13);
    else if (man == 0)
        x = sign;
    else { // subnormal
        exp = 113;
        while (!(man & 0x400)) {
            man <<= 1;
            --exp;
        }
        x = sign | (exp << 23) | ((man & 0x3ff) << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
        s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
        s.remove_suffix(1);
    return s;
}

// "i32 [1, 2, 3]" / "[1 2 3]" -> tensor
Tensor parse_tensor(std::string_view text, int line) {
    text = trim(text);
    int64_t addr = -1;
    if (!text.empty() && text[0] == '@') { // "@0x4000: u8 [..]" (the key before the ':' was already removed)
        const size_t colon = text.find(':');
        if (colon == std::string_view::npos)
            throw TestError("line " + std::to_string(line) + ": expected ':' after the address");
        auto v = parse_integer(trim(text.substr(1, colon - 1)));
        if (!v || *v < 0)
            throw TestError("line " + std::to_string(line) + ": bad address");
        addr = *v;
        text = trim(text.substr(colon + 1));
    }
    const size_t open = text.find('[');
    const size_t close = text.rfind(']');
    if (open == std::string_view::npos || close == std::string_view::npos || close < open)
        throw TestError("line " + std::to_string(line) + ": expected a value list in [..]");
    if (!trim(text.substr(close + 1)).empty())
        throw TestError("line " + std::to_string(line) + ": unexpected text after ']'");

    const TypeInfo *type = &kTypes[4];
    auto type_name = trim(text.substr(0, open));
    if (!type_name.empty()) {
        auto it = std::find_if(std::begin(kTypes), std::end(kTypes), [&](const TypeInfo &t) { return type_name == t.name; });
        if (it == std::end(kTypes))
            throw TestError("line " + std::to_string(line) + ": unknown element type '" + std::string(type_name) + "'");
        type = &*it;
    }

    Tensor t{type, 0, {}, line, addr};
    std::string body(text.substr(open + 1, close - open - 1));
    std::replace(body.begin(), body.end(), ',', ' ');
    std::istringstream is(body);
    std::string tok;
    while (is >> tok) {
        uint8_t raw[4];
        if (type->type == ElemType::F32 || type->type == ElemType::F16) {
            float f;
            try {
                size_t used = 0;
                f = std::stof(tok, &used);
                if (used != tok.size())
                    throw std::invalid_argument(tok);
            } catch (const std::exception &) {
                throw TestError("line " + std::to_string(line) + ": bad float value '" + tok + "'");
            }
            if (type->type == ElemType::F16) {
                const uint16_t h = float_to_half(f);
                std::memcpy(raw, &h, 2);
            } else
                std::memcpy(raw, &f, 4);
        } else {
            auto v = parse_integer(tok);
            if (!v)
                throw TestError("line " + std::to_string(line) + ": bad integer '" + tok + "'");
            int64_t lo = 0, hi = 0;
            switch (type->type) {
            case ElemType::I8: lo = INT8_MIN; hi = INT8_MAX; break;
            case ElemType::U8: lo = 0; hi = UINT8_MAX; break;
            case ElemType::I16: lo = INT16_MIN; hi = INT16_MAX; break;
            case ElemType::U16: lo = 0; hi = UINT16_MAX; break;
            case ElemType::I32: lo = INT32_MIN; hi = INT32_MAX; break;
            default: lo = 0; hi = UINT32_MAX; break;
            }
            if (*v < lo || *v > hi)
                throw TestError("line " + std::to_string(line) + ": " + tok + " is out of range for " + type->name);
            const uint32_t u = static_cast<uint32_t>(*v);
            std::memcpy(raw, &u, 4); // little endian host
        }
        t.bytes.insert(t.bytes.end(), raw, raw + type->size);
        ++t.count;
    }
    if (t.count == 0)
        throw TestError("line " + std::to_string(line) + ": empty tensor");
    return t;
}

struct TestCase {
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    std::vector<Tensor> ddr_inputs;
    std::vector<Tensor> ddr_outputs;
    bool expect_fail = false;
};

TestCase parse_test_case(const std::string &source) {
    TestCase tc;
    std::istringstream is(source);
    std::string line;
    int no = 0;
    while (std::getline(is, line)) {
        ++no;
        auto text = trim(line);
        if (text.empty() || text[0] != '#')
            continue;
        text = trim(text.substr(1));
        if (text.rfind("EXPECT_FAIL", 0) == 0) {
            tc.expect_fail = true;
            continue;
        }
        // "<key>: ..." or "<key> @addr: ..."
        const size_t colon = text.find(':');
        if (colon == std::string_view::npos)
            continue;
        std::string_view head = trim(text.substr(0, colon));
        std::string_view rest = text.substr(colon + 1);
        std::string_view key = head.substr(0, head.find('@'));
        key = trim(key);
        if (key != "IN" && key != "OUT" && key != "DDR_IN" && key != "DDR_OUT")
            continue;
        std::string with_addr = head.find('@') == std::string_view::npos ? std::string(rest)
                                                                          : std::string(head.substr(head.find('@'))) + ":" + std::string(rest);
        Tensor t = parse_tensor(with_addr, no);
        if (key == "IN")
            tc.inputs.push_back(std::move(t));
        else if (key == "OUT")
            tc.outputs.push_back(std::move(t));
        else if (key == "DDR_IN")
            tc.ddr_inputs.push_back(std::move(t));
        else
            tc.ddr_outputs.push_back(std::move(t));
    }
    if (tc.outputs.empty() && tc.ddr_outputs.empty() && !tc.expect_fail)
        throw TestError("the test has no '# OUT:' / '# DDR_OUT:' line");
    return tc;
}

size_t align_to(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

// Size of the automatically laid out tensors (tensors with an explicit address are not counted).
size_t layout_size(const std::vector<Tensor> &tensors, size_t align = 4) {
    size_t n = 0;
    for (const auto &t : tensors)
        if (t.addr < 0)
            n = align_to(n, align) + t.bytes.size();
    return align_to(n, align);
}

// Address of every tensor: explicit or the next free slot after `base`.
std::vector<size_t> place(const std::vector<Tensor> &tensors, size_t base, size_t align) {
    std::vector<size_t> r;
    size_t off = base;
    for (const auto &t : tensors) {
        if (t.addr >= 0) {
            r.push_back(size_t(t.addr));
            continue;
        }
        off = align_to(off, align);
        r.push_back(off);
        off += t.bytes.size();
    }
    return r;
}

std::string unique_name(const char *tag) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    char buf[64];
    std::snprintf(buf, sizeof buf, "k230isa-%s-%016llx", tag, static_cast<unsigned long long>(rng()));
    return buf;
}

std::string format_element(const TypeInfo &t, const uint8_t *p) {
    char buf[64];
    switch (t.type) {
    case ElemType::I8: std::snprintf(buf, sizeof buf, "%d (0x%02x)", int(int8_t(*p)), unsigned(*p)); break;
    case ElemType::U8: std::snprintf(buf, sizeof buf, "%u (0x%02x)", unsigned(*p), unsigned(*p)); break;
    case ElemType::I16: { int16_t v; std::memcpy(&v, p, 2); std::snprintf(buf, sizeof buf, "%d (0x%04x)", int(v), unsigned(uint16_t(v))); break; }
    case ElemType::U16: { uint16_t v; std::memcpy(&v, p, 2); std::snprintf(buf, sizeof buf, "%u (0x%04x)", unsigned(v), unsigned(v)); break; }
    case ElemType::I32: { int32_t v; std::memcpy(&v, p, 4); std::snprintf(buf, sizeof buf, "%d (0x%08x)", v, unsigned(uint32_t(v))); break; }
    case ElemType::U32: { uint32_t v; std::memcpy(&v, p, 4); std::snprintf(buf, sizeof buf, "%u (0x%08x)", v, v); break; }
    case ElemType::F16: { uint16_t v; std::memcpy(&v, p, 2); std::snprintf(buf, sizeof buf, "%g (0x%04x)", double(half_to_float(v)), unsigned(v)); break; }
    case ElemType::F32: { float v; uint32_t u; std::memcpy(&v, p, 4); std::memcpy(&u, p, 4); std::snprintf(buf, sizeof buf, "%g (0x%08x)", double(v), u); break; }
    }
    return buf;
}

int run(const std::string &path, bool verbose) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw TestError("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string source = ss.str();

    TestCase tc = parse_test_case(source);
    AsmProgram prog = assemble(source);

    if (layout_size(tc.inputs) > kOutputBase - kInputBase)
        throw TestError("input tensors do not fit before the output area (0x1000 bytes)");
    if (kOutputBase + layout_size(tc.outputs) > kCtrlSize)
        throw TestError("output tensors do not fit into the control memory");
    if (layout_size(tc.ddr_inputs, kDdrAlign) > kDdrOutBase - kDdrInBase)
        throw TestError("DDR input tensors do not fit before the DDR output area");

    if (verbose) {
        for (size_t i = 0; i < prog.offsets.size(); ++i) {
            const uint32_t off = prog.offsets[i];
            const uint32_t end = i + 1 < prog.offsets.size() ? prog.offsets[i + 1] : uint32_t(prog.text.size());
            std::printf("%04x:", off);
            for (uint32_t b = end; b > off; --b)
                std::printf("%02x", prog.text[b - 1]);
            std::printf("\n");
        }
    }

    // Memory: DDR holds the code, ctrl (GLB bank 0) holds inputs and outputs.
    const std::string ddr_name = unique_name("ddr");
    const std::string ctrl_name = unique_name("ctrl");
    shared_memory ddr(ddr_name, kDdrSize, shared_memory_openmode::create);
    shared_memory ctrl(ctrl_name, kCtrlSize, shared_memory_openmode::create);
    auto *ddr_base = static_cast<uint8_t *>(ddr.data());
    auto *ctrl_base = static_cast<uint8_t *>(ctrl.data());

    std::memcpy(ddr_base + kTextOffset, prog.text.data(), prog.text.size());

    auto check_range = [](const std::vector<Tensor> &ts, const std::vector<size_t> &at, size_t limit, const char *what) {
        for (size_t i = 0; i < ts.size(); ++i)
            if (at[i] + ts[i].bytes.size() > limit)
                throw TestError(std::string("line ") + std::to_string(ts[i].line) + ": " + what + " tensor is outside of the memory");
    };
    const auto in_at = place(tc.inputs, kInputBase, 4);
    const auto out_at = place(tc.outputs, kOutputBase, 4);
    const auto ddr_in_at = place(tc.ddr_inputs, kDdrInBase, kDdrAlign);
    const auto ddr_out_at = place(tc.ddr_outputs, kDdrOutBase, kDdrAlign);
    check_range(tc.inputs, in_at, kCtrlSize, "IN");
    check_range(tc.outputs, out_at, kCtrlSize, "OUT");
    check_range(tc.ddr_inputs, ddr_in_at, kDdrSize, "DDR_IN");
    check_range(tc.ddr_outputs, ddr_out_at, kDdrSize, "DDR_OUT");

    // pre-fill the expected outputs with a pattern, so a test that forgets to write a result fails
    for (size_t i = 0; i < tc.outputs.size(); ++i)
        std::memset(ctrl_base + out_at[i], 0xA5, tc.outputs[i].bytes.size());
    for (size_t i = 0; i < tc.ddr_outputs.size(); ++i)
        std::memset(ddr_base + ddr_out_at[i], 0xA5, tc.ddr_outputs[i].bytes.size());
    for (size_t i = 0; i < tc.inputs.size(); ++i)
        std::memcpy(ctrl_base + in_at[i], tc.inputs[i].bytes.data(), tc.inputs[i].bytes.size());
    for (size_t i = 0; i < tc.ddr_inputs.size(); ++i)
        std::memcpy(ddr_base + ddr_in_at[i], tc.ddr_inputs[i].bytes.data(), tc.ddr_inputs[i].bytes.size());

    // The simulator is run exactly as the host runtime runs it: <ddr> <ctrl> <text offset> <dump path>.
    const int rc = nncase::runtime::k230::run_cmodel(ddr_name, ctrl_name, kTextOffset, ".");
    if (tc.expect_fail) {
        if (rc == 0) {
            std::fprintf(stderr, "FAIL %s: the simulator accepted a program that has to fail\n", path.c_str());
            return 1;
        }
        std::printf("PASS %s (the simulator rejected it with %d, as expected)\n", path.c_str(), rc);
        return 0;
    }
    if (rc != 0) {
        std::fprintf(stderr, "FAIL %s: the simulator returned %d\n", path.c_str(), rc);
        return 3;
    }

    // Verify.
    int mismatches = 0;
    auto verify = [&](const std::vector<Tensor> &tensors, const std::vector<size_t> &at, const uint8_t *base, const char *what) {
        for (size_t ti = 0; ti < tensors.size(); ++ti) {
            const auto &t = tensors[ti];
            const size_t o = at[ti];
            const uint8_t *actual = base + o;
            for (size_t i = 0; i < t.count; ++i) {
                const uint8_t *e = t.bytes.data() + i * t.type->size;
                const uint8_t *a = actual + i * t.type->size;
                if (std::memcmp(e, a, t.type->size) != 0) {
                    if (++mismatches <= 16)
                        std::fprintf(stderr, "  %s #%zu [%zu] (+0x%zx): expected %s, got %s\n", what, ti, i, o + i * t.type->size,
                            format_element(*t.type, e).c_str(), format_element(*t.type, a).c_str());
                }
            }
        }
    };
    verify(tc.outputs, out_at, ctrl_base, "OUT");
    verify(tc.ddr_outputs, ddr_out_at, ddr_base, "DDR_OUT");
    if (mismatches) {
        std::fprintf(stderr, "FAIL %s: %d element(s) differ\n", path.c_str(), mismatches);
        return 1;
    }
    std::printf("PASS %s (%zu bytes of code)\n", path.c_str(), prog.text.size());
    return 0;
}

std::string quote(const std::string &s) { return "\"" + s + "\""; }

int run_directory(const char *self, const std::string &dir, bool verbose) {
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
        std::string cmd = quote(self) + (verbose ? " -v " : " ") + quote(file);
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

} // namespace

int main(int argc, char **argv) {
    std::string path;
    bool verbose = false;
    bool usage_error = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-v") == 0)
            verbose = true;
        else if (path.empty())
            path = argv[i];
        else
            usage_error = true;
    }
    if (path.empty() || usage_error) {
        std::fprintf(stderr, "usage: %s [-v] <program.s | directory>\n", argv[0]);
        return 2;
    }
    try {
        if (std::filesystem::is_directory(path))
            return run_directory(argv[0], path, verbose);
        return run(path, verbose);
    } catch (const AsmError &e) {
        std::fprintf(stderr, "%s:%s\n", path.c_str(), e.what());
        return 2;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s: %s\n", path.c_str(), e.what());
        return 2;
    }
}
