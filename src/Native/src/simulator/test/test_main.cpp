// Test runner for the K230 simulator: assembles one .s file, runs it in the simulator and verifies the output.
//
//   k230_isa_test [-v] <program.s>
//
// Test description (comments in the assembly file):
//
//   # IN:  [3, 3]            input tensor, one line per tensor, optional element type before the list
//   # IN:  u8 [128, 255]     types: i8 u8 i16 u16 i32 (default) u32 f32
//   # OUT: [9]               expected output tensor(s), same syntax
//
// Memory layout seen by the program (GLB bank 0 = the "ctrl" shared memory, addresses are plain GLB addresses):
//
//   0x0000 .. : input tensors, back to back, each aligned to 4 bytes
//   0x1000 .. : output tensors, back to back, each aligned to 4 bytes (pre-filled with 0xA5)
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
#include <cstring>
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

struct TestError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class ElemType { I8, U8, I16, U16, I32, U32, F32 };

struct TypeInfo {
    const char *name;
    ElemType type;
    size_t size;
};

constexpr TypeInfo kTypes[] = {
    {"i8", ElemType::I8, 1}, {"u8", ElemType::U8, 1}, {"i16", ElemType::I16, 2}, {"u16", ElemType::U16, 2},
    {"i32", ElemType::I32, 4}, {"u32", ElemType::U32, 4}, {"f32", ElemType::F32, 4},
};

struct Tensor {
    const TypeInfo *type;
    size_t count;
    std::vector<uint8_t> bytes;
    int line;
};

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

    Tensor t{type, 0, {}, line};
    std::string body(text.substr(open + 1, close - open - 1));
    std::replace(body.begin(), body.end(), ',', ' ');
    std::istringstream is(body);
    std::string tok;
    while (is >> tok) {
        uint8_t raw[4];
        if (type->type == ElemType::F32) {
            float f;
            try {
                size_t used = 0;
                f = std::stof(tok, &used);
                if (used != tok.size())
                    throw std::invalid_argument(tok);
            } catch (const std::exception &) {
                throw TestError("line " + std::to_string(line) + ": bad f32 value '" + tok + "'");
            }
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
        if (text.rfind("IN:", 0) == 0)
            tc.inputs.push_back(parse_tensor(text.substr(3), no));
        else if (text.rfind("OUT:", 0) == 0)
            tc.outputs.push_back(parse_tensor(text.substr(4), no));
    }
    if (tc.outputs.empty())
        throw TestError("the test has no '# OUT:' line");
    return tc;
}

size_t align4(size_t v) { return (v + 3) & ~size_t(3); }

size_t layout_size(const std::vector<Tensor> &tensors) {
    size_t n = 0;
    for (const auto &t : tensors)
        n = align4(n) + t.bytes.size();
    return align4(n);
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

    size_t off = kInputBase;
    for (const auto &t : tc.inputs) {
        off = align4(off);
        std::memcpy(ctrl_base + off, t.bytes.data(), t.bytes.size());
        off += t.bytes.size();
    }
    std::memset(ctrl_base + kOutputBase, 0xA5, layout_size(tc.outputs));

    // The simulator is run exactly as the host runtime runs it: <ddr> <ctrl> <text offset> <dump path>.
    const int rc = nncase::runtime::k230::run_cmodel(ddr_name, ctrl_name, kTextOffset, ".");
    if (rc != 0) {
        std::fprintf(stderr, "FAIL %s: the simulator returned %d\n", path.c_str(), rc);
        return 3;
    }

    // Verify.
    int mismatches = 0;
    off = kOutputBase;
    for (size_t ti = 0; ti < tc.outputs.size(); ++ti) {
        const auto &t = tc.outputs[ti];
        off = align4(off);
        const uint8_t *actual = ctrl_base + off;
        for (size_t i = 0; i < t.count; ++i) {
            const uint8_t *e = t.bytes.data() + i * t.type->size;
            const uint8_t *a = actual + i * t.type->size;
            if (std::memcmp(e, a, t.type->size) != 0) {
                if (++mismatches <= 16)
                    std::fprintf(stderr, "  OUT #%zu [%zu] (ctrl+0x%zx): expected %s, got %s\n", ti, i, off + i * t.type->size,
                        format_element(*t.type, e).c_str(), format_element(*t.type, a).c_str());
            }
        }
        off += t.bytes.size();
    }
    if (mismatches) {
        std::fprintf(stderr, "FAIL %s: %d element(s) differ\n", path.c_str(), mismatches);
        return 1;
    }
    std::printf("PASS %s (%zu bytes of code)\n", path.c_str(), prog.text.size());
    return 0;
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
        std::fprintf(stderr, "usage: %s [-v] <program.s>\n", argv[0]);
        return 2;
    }
    try {
        return run(path, verbose);
    } catch (const AsmError &e) {
        std::fprintf(stderr, "%s:%s\n", path.c_str(), e.what());
        return 2;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s: %s\n", path.c_str(), e.what());
        return 2;
    }
}
