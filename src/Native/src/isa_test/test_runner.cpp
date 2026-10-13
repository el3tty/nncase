#include "test_runner.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace k230::isa_test {

size_t align_to(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

namespace {

// All accesses to the test memory are single byte accesses. On the device the GLB is a mapping of device memory
// (/dev/mmz): library memcpy / memset / memcmp may use wide or misaligned accesses, which can fault there.
void write_bytes(uint8_t *dst, const uint8_t *src, size_t n) {
    volatile uint8_t *d = dst;
    for (size_t i = 0; i < n; ++i)
        d[i] = src[i];
}

void fill_bytes(uint8_t *dst, uint8_t value, size_t n) {
    volatile uint8_t *d = dst;
    for (size_t i = 0; i < n; ++i)
        d[i] = value;
}

std::vector<uint8_t> read_bytes(const uint8_t *src, size_t n) {
    const volatile uint8_t *s = src;
    std::vector<uint8_t> r(n);
    for (size_t i = 0; i < n; ++i)
        r[i] = s[i];
    return r;
}

// Size of the automatically laid out tensors (tensors with an explicit address are not counted).
size_t layout_size(const std::vector<Tensor> &tensors, size_t align) {
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

void check_range(const std::vector<Tensor> &ts, const std::vector<size_t> &at, size_t limit, const char *what) {
    for (size_t i = 0; i < ts.size(); ++i)
        if (at[i] + ts[i].bytes.size() > limit)
            throw TestError(std::string("line ") + std::to_string(ts[i].line) + ": " + what + " tensor is outside of the memory");
}

} // namespace

Layout make_layout(const TestCase &tc) {
    if (layout_size(tc.inputs, 4) > kOutputBase - kInputBase)
        throw TestError("input tensors do not fit before the output area (0x1000 bytes)");
    if (kOutputBase + layout_size(tc.outputs, 4) > kGlbSize)
        throw TestError("output tensors do not fit into the GLB");

    Layout l;
    l.in_at = place(tc.inputs, kInputBase, 4);
    l.out_at = place(tc.outputs, kOutputBase, 4);
    l.ddr_in_at = place(tc.ddr_inputs, 0, kDdrAlign);
    l.ddr_out_at = place(tc.ddr_outputs, 0, kDdrAlign);
    l.ddr_in_size = layout_size(tc.ddr_inputs, kDdrAlign);
    l.ddr_out_size = layout_size(tc.ddr_outputs, kDdrAlign);
    check_range(tc.inputs, l.in_at, kGlbSize, "IN");
    check_range(tc.outputs, l.out_at, kGlbSize, "OUT");
    return l;
}

void init_memory(const TestCase &tc, const Layout &l, const MemoryImage &mem) {
    for (size_t i = 0; i < tc.outputs.size(); ++i)
        fill_bytes(mem.glb + l.out_at[i], 0xA5, tc.outputs[i].bytes.size());
    for (size_t i = 0; i < tc.ddr_outputs.size(); ++i)
        fill_bytes(mem.ddr_out + l.ddr_out_at[i], 0xA5, tc.ddr_outputs[i].bytes.size());
    for (size_t i = 0; i < tc.inputs.size(); ++i)
        write_bytes(mem.glb + l.in_at[i], tc.inputs[i].bytes.data(), tc.inputs[i].bytes.size());
    for (size_t i = 0; i < tc.ddr_inputs.size(); ++i)
        write_bytes(mem.ddr_in + l.ddr_in_at[i], tc.ddr_inputs[i].bytes.data(), tc.ddr_inputs[i].bytes.size());
}

int verify_outputs(const TestCase &tc, const Layout &l, const MemoryImage &mem) {
    int mismatches = 0;
    auto verify = [&](const std::vector<Tensor> &tensors, const std::vector<size_t> &at, const uint8_t *base, const char *what) {
        for (size_t ti = 0; ti < tensors.size(); ++ti) {
            const auto &t = tensors[ti];
            const size_t o = at[ti];
            const std::vector<uint8_t> snapshot = read_bytes(base + o, t.bytes.size());
            const uint8_t *actual = snapshot.data();
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
    verify(tc.outputs, l.out_at, mem.glb, "OUT");
    verify(tc.ddr_outputs, l.ddr_out_at, mem.ddr_out, "DDR_OUT");
    return mismatches;
}

SymbolTable program_symbols(const MemoryImage &mem) {
    return {{"DDR_IN", static_cast<int64_t>(mem.ddr_in_addr)}, {"DDR_OUT", static_cast<int64_t>(mem.ddr_out_addr)}};
}

AsmProgram assemble_test(const TestCase &tc, const RunOptions &options, const SymbolTable &symbols) {
    AsmProgram prog = assemble(tc.source, symbols);

    if (options.verbose) {
        for (size_t i = 0; i < prog.offsets.size(); ++i) {
            const uint32_t off = prog.offsets[i];
            const uint32_t end = i + 1 < prog.offsets.size() ? prog.offsets[i + 1] : uint32_t(prog.text.size());
            std::printf("%04x:", off);
            for (uint32_t b = end; b > off; --b)
                std::printf("%02x", prog.text[b - 1]);
            std::printf("\n");
        }
    }
    return prog;
}

} // namespace k230::isa_test
