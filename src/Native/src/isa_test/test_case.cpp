#include "test_case.h"

#include "assembler.h" // parse_integer

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace k230::isa_test {

namespace {

constexpr TypeInfo kTypes[] = {
    {"i8", ElemType::I8, 1}, {"u8", ElemType::U8, 1}, {"i16", ElemType::I16, 2}, {"u16", ElemType::U16, 2},
    {"i32", ElemType::I32, 4}, {"u32", ElemType::U32, 4}, {"f32", ElemType::F32, 4}, {"f16", ElemType::F16, 2},
};

} // namespace

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

namespace {

std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
        s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
        s.remove_suffix(1);
    return s;
}

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

} // namespace

TestCase parse_test_case(const std::string &source) {
    TestCase tc;
    tc.source = source;
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
        if (t.addr >= 0 && key != "IN" && key != "OUT")
            throw TestError("line " + std::to_string(no) + ": DDR tensors cannot have an explicit address (the program gets their address through the DDR_IN / DDR_OUT symbols)");
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

} // namespace k230::isa_test
