#include "test_case.h"

#include "assembler.h" // parse_integer

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace k230::isa_test {

using k230::assembler::parse_integer;

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

std::string at_line(int line) { return "line " + std::to_string(line) + ": "; }

// Text after a second '#' on a tensor line is a comment.
std::string_view strip_comment(std::string_view s) { return s.substr(0, s.find('#')); }

int bracket_depth(std::string_view s) {
    int depth = 0;
    for (char c : s)
        depth += c == '[' ? 1 : c == ']' ? -1 : 0;
    return depth;
}

// One element: a number of the element type; in f16 / f32 tensors a hex number is the bit pattern.
void append_value(Tensor &t, std::string_view tok, int line) {
    uint8_t raw[4];
    const ElemType type = t.type->type;
    const bool is_float = type == ElemType::F32 || type == ElemType::F16;
    const bool hex = tok.size() > 2 && tok[0] == '0' && (tok[1] == 'x' || tok[1] == 'X');
    if (is_float && !hex) {
        const std::string s(tok);
        float f;
        try {
            size_t used = 0;
            f = std::stof(s, &used);
            if (used != s.size())
                throw std::invalid_argument(s);
        } catch (const std::exception &) {
            throw TestError(at_line(line) + "bad float value '" + s + "'");
        }
        if (type == ElemType::F16) {
            const uint16_t h = float_to_half(f);
            std::memcpy(raw, &h, 2);
        } else
            std::memcpy(raw, &f, 4);
    } else {
        auto v = parse_integer(tok);
        if (!v)
            throw TestError(at_line(line) + "bad integer '" + std::string(tok) + "'");
        int64_t lo = 0, hi = 0;
        switch (type) {
        case ElemType::I8: lo = INT8_MIN; hi = INT8_MAX; break;
        case ElemType::U8: lo = 0; hi = UINT8_MAX; break;
        case ElemType::I16: lo = INT16_MIN; hi = INT16_MAX; break;
        case ElemType::U16: case ElemType::F16: lo = 0; hi = UINT16_MAX; break;
        case ElemType::I32: lo = INT32_MIN; hi = INT32_MAX; break;
        default: lo = 0; hi = UINT32_MAX; break;
        }
        // hex is accepted for the bit pattern of a signed type as well (0xff in an i8 tensor = -1)
        if (hex && (type == ElemType::I8 || type == ElemType::I16 || type == ElemType::I32))
            hi = hi * 2 + 1;
        if (*v < lo || *v > hi)
            throw TestError(at_line(line) + std::string(tok) + " is out of range for " + t.type->name);
        const uint32_t u = static_cast<uint32_t>(*v);
        std::memcpy(raw, &u, 4); // little endian host
    }
    t.bytes.insert(t.bytes.end(), raw, raw + t.type->size);
    ++t.count;
}

// "[1, 2*3, [4, 5]*2]": values with an optional repeat count; nested lists only group the values (and are checked
// against the shape). Elements are separated by ',' or white space (also line breaks).
class ValueParser {
public:
    ValueParser(Tensor &t, std::string_view text, int line) : t_(t), text_(text), line_(line) {}

    void parse() {
        skip_space();
        if (pos_ >= text_.size() || text_[pos_] != '[')
            throw TestError(at_line(line_) + "expected a value list in [..]");
        parse_list(0);
        skip_space();
        if (pos_ != text_.size())
            throw TestError(at_line(line_) + "unexpected text after ']'");
        if (t_.count == 0)
            throw TestError(at_line(line_) + "empty tensor");
        check_shape();
    }

private:
    struct List {
        size_t level;
        size_t elements; // values (with their repeat counts) and sub-lists at this level
        bool has_values, has_lists;
    };

    void skip_space() {
        while (pos_ < text_.size() && (std::isspace(static_cast<unsigned char>(text_[pos_])) || text_[pos_] == ','))
            ++pos_;
    }

    // "*n" after a value or a list, 1 without
    size_t repeat_count() {
        if (pos_ >= text_.size() || text_[pos_] != '*')
            return 1;
        ++pos_;
        const size_t begin = pos_;
        while (pos_ < text_.size() && std::isalnum(static_cast<unsigned char>(text_[pos_])))
            ++pos_;
        auto n = parse_integer(text_.substr(begin, pos_ - begin));
        if (!n || *n < 1)
            throw TestError(at_line(line_) + "bad repeat count '" + std::string(text_.substr(begin, pos_ - begin)) + "'");
        return static_cast<size_t>(*n);
    }

    void repeat_tail(size_t from_byte, size_t from_count, size_t times) {
        const std::vector<uint8_t> once(t_.bytes.begin() + from_byte, t_.bytes.end());
        const size_t count = t_.count - from_count;
        for (size_t i = 1; i < times; ++i) {
            t_.bytes.insert(t_.bytes.end(), once.begin(), once.end());
            t_.count += count;
        }
    }

    void parse_list(size_t level) {
        ++pos_; // '['
        const size_t index = lists_.size();
        lists_.push_back({level, 0, false, false});
        for (;;) {
            skip_space();
            if (pos_ >= text_.size())
                throw TestError(at_line(line_) + "missing ']'");
            const char c = text_[pos_];
            const size_t bytes = t_.bytes.size(), count = t_.count;
            if (c == ']') {
                ++pos_;
                return;
            }
            size_t times;
            if (c == '[') {
                parse_list(level + 1);
                times = repeat_count();
                lists_[index].has_lists = true;
            } else {
                const size_t begin = pos_;
                while (pos_ < text_.size() && !std::isspace(static_cast<unsigned char>(text_[pos_])) &&
                       std::strchr(",[]*", text_[pos_]) == nullptr)
                    ++pos_;
                append_value(t_, text_.substr(begin, pos_ - begin), line_);
                times = repeat_count();
                lists_[index].has_values = true;
            }
            repeat_tail(bytes, count, times);
            lists_[index].elements += times;
        }
    }

    // A flat list only has to have as many values as the shape; a nested one has to match it level by level.
    void check_shape() {
        const auto &shape = t_.shape;
        if (shape.empty())
            return;
        size_t total = 1;
        for (size_t d : shape)
            total *= d;
        if (lists_.size() == 1) {
            if (t_.count != total)
                throw TestError(at_line(line_) + "the shape has " + std::to_string(total) + " elements, the list " +
                                std::to_string(t_.count));
            return;
        }
        for (const auto &l : lists_) {
            const bool leaf = l.level + 1 == shape.size();
            if (l.level >= shape.size() || (leaf ? l.has_lists : l.has_values) || l.elements != shape[l.level])
                throw TestError(at_line(line_) + "the nested lists do not match the shape (a list at depth " +
                                std::to_string(l.level + 1) + " has " + std::to_string(l.elements) + " elements, the shape " +
                                (l.level < shape.size() ? std::to_string(shape[l.level]) : std::string("none")) + ")");
        }
    }

    Tensor &t_;
    std::string_view text_;
    int line_;
    size_t pos_ = 0;
    std::vector<List> lists_;
};

// "[@addr:] [type[shape]] [values]" (the key before the ':' was already removed)
Tensor parse_tensor(std::string_view text, int line) {
    text = trim(text);
    int64_t addr = -1;
    if (!text.empty() && text[0] == '@') {
        const size_t colon = text.find(':');
        if (colon == std::string_view::npos)
            throw TestError(at_line(line) + "expected ':' after the address");
        auto v = parse_integer(trim(text.substr(1, colon - 1)));
        if (!v || *v < 0)
            throw TestError(at_line(line) + "bad address");
        addr = *v;
        text = trim(text.substr(colon + 1));
    }

    size_t name_end = 0;
    while (name_end < text.size() && std::isalnum(static_cast<unsigned char>(text[name_end])))
        ++name_end;
    const TypeInfo *type = &kTypes[4];
    if (name_end > 0) {
        const auto type_name = text.substr(0, name_end);
        auto it = std::find_if(std::begin(kTypes), std::end(kTypes), [&](const TypeInfo &t) { return type_name == t.name; });
        if (it == std::end(kTypes))
            throw TestError(at_line(line) + "unknown element type '" + std::string(type_name) + "'");
        type = &*it;
    }
    Tensor t{type, 0, {}, line, addr};
    text.remove_prefix(name_end);

    // shape: written right after the type, "u8[2,3]"
    if (name_end > 0 && !text.empty() && text[0] == '[') {
        const size_t close = text.find(']');
        if (close == std::string_view::npos)
            throw TestError(at_line(line) + "missing ']' after the shape");
        std::string dims(text.substr(1, close - 1));
        std::replace(dims.begin(), dims.end(), ',', ' ');
        std::istringstream is(dims);
        for (std::string d; is >> d;) {
            auto v = parse_integer(d);
            if (!v || *v < 1)
                throw TestError(at_line(line) + "bad dimension '" + d + "'");
            t.shape.push_back(static_cast<size_t>(*v));
        }
        if (t.shape.empty())
            throw TestError(at_line(line) + "empty shape");
        text.remove_prefix(close + 1);
    }

    ValueParser(t, text, line).parse();
    return t;
}

} // namespace

TestCase parse_test_case(const std::string &source) {
    TestCase tc;
    tc.source = source;
    std::vector<std::string> lines;
    {
        std::istringstream is(source);
        for (std::string line; std::getline(is, line);)
            lines.push_back(line);
    }
    for (size_t i = 0; i < lines.size(); ++i) {
        const int no = static_cast<int>(i + 1);
        auto text = trim(lines[i]);
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
        std::string_view key = trim(head.substr(0, head.find('@')));
        if (key != "IN" && key != "OUT" && key != "DDR_IN" && key != "DDR_OUT")
            continue;
        std::string body = head.find('@') == std::string_view::npos
                               ? std::string(strip_comment(text.substr(colon + 1)))
                               : std::string(head.substr(head.find('@'))) + ":" + std::string(strip_comment(text.substr(colon + 1)));
        // a list that is not closed continues on the following comment lines
        int depth = bracket_depth(body);
        while (depth > 0) {
            if (++i >= lines.size())
                throw TestError(at_line(no) + "missing ']' (the list continues on '#' lines)");
            auto next = trim(lines[i]);
            if (next.empty() || next[0] != '#')
                throw TestError(at_line(no) + "missing ']' before line " + std::to_string(i + 1) +
                                " (the list continues on '#' lines)");
            const auto part = strip_comment(next.substr(1));
            depth += bracket_depth(part);
            body += '\n';
            body += part;
        }
        Tensor t = parse_tensor(body, no);
        if (t.addr >= 0 && key != "IN" && key != "OUT")
            throw TestError(at_line(no) + "DDR tensors cannot have an explicit address (the program gets their address through the DDR_IN / DDR_OUT symbols)");
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
