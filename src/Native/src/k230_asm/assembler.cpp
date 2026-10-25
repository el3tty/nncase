#include "assembler.h"
#include "string_switch.h"

#include <algorithm>
#include <cctype>
#include <charconv>

namespace k230::assembler {
namespace {

// ---------------------------------------------------------------------------------------------------------------------
// Instruction descriptions
// ---------------------------------------------------------------------------------------------------------------------

enum class Kind {
    Reg,    // general purpose register x0..x31 (5 bit)
    Shape,  // shape / stride register s0..s7 (also ss0..ss7, the compiler dump spelling; 3 bit)
    Uimm,   // unsigned immediate
    Simm,   // signed immediate (two's complement in the field)
    Branch, // label or byte offset relative to the instruction, signed
};

// Names of the values of an enumerated field, as the compiler dumps them.
using NameTable = std::vector<std::pair<std::string_view, uint32_t>>;

struct Operand {
    Kind kind;
    unsigned lo;
    unsigned width;
    bool optional;
    const NameTable *names = nullptr; // Uimm only: accepted names besides numbers
};

// The enums of the compiler (modules/Nncase.Modules.K230/Nncase.TIR.Instructions/<NAME>.cs), in declaration order.
const NameTable act0_channel{{"pu", 0}, {"pdp0", 1}};                                      // ACT0_CHANNEL
const NameTable act0_output_dest{{"psum", 0}, {"dm", 1}, {"psum_dm", 2}};                  // ACT0_OUTPUT_DEST
const NameTable act_output_type{{"u8", 0}, {"i8", 1}, {"fp16", 2}, {"i16", 3}};            // ACT_OUTPUT_TYPE
const NameTable act1_source_type{{"l2", 0}, {"psum", 1}};                                  // ACT1_SOURCE_TYPE
const NameTable ddr_datatype{{"i8", 0}, {"fp16", 1}, {"fp32", 2}, {"i4", 3}, {"i6", 4}, {"i16", 5}}; // DDR_DATATYPE
const NameTable dm_load_w_dest{{"pu", 0}, {"pdp0", 1}};                                    // DM_LOAD_W_DEST
const NameTable l1_type{{"if_", 0}, {"w", 1}, {"psum", 2}};                                // L1_TYPE
const NameTable l2_datatype{{"i8", 0}, {"fp16", 1}, {"i16", 2}};                           // L2_DATATYPE
const NameTable mfu_act1_function{{"add", 0}, {"mul", 1}};                                 // MFU_ACT1_FUNCTION
const NameTable mfu_trans_permute{                                                         // MFU_TRANS_PERMUTE
    {"NCHW", 0}, {"NCWH", 1}, {"NHCW", 2}, {"NHWC", 3}, {"NWCH", 4}, {"NWHC", 5}, {"CNHW", 6}, {"CNWH", 7},
    {"CHNW", 8}, {"CHWN", 9}, {"CWNH", 10}, {"CWHN", 11}, {"HNCW", 12}, {"HNWC", 13}, {"HCNW", 14}, {"HCWN", 15},
    {"HWNC", 16}, {"HWCN", 17}, {"WNCH", 18}, {"WNHC", 19}, {"WCNH", 20}, {"WCHN", 21}, {"WHNC", 22}, {"WHCN", 23}};
const NameTable pdp_function{{"min", 0}, {"max", 1}, {"average", 2}, {"sum", 3}};          // PDP_FUNCTION
const NameTable pu_compute_mode{{"pu_mode_normal", 0}, {"pu_mode_deconv", 1}};             // PU_COMPUTE_MODE
const NameTable pu_of_shift_mode{{"none", 0}, {"left_shift_4", 1}, {"right_shift_4", 2}};  // PU_OF_SHIFT_MODE
const NameTable pu_output_dest{{"psum", 0}, {"act0", 1}};                                  // PU_OUTPUT_DEST
const NameTable quant_type{{"disable", 0}, {"u8", 1}, {"i8", 2}, {"i16", 3}};              // QUANT_TYPE
const NameTable slice_location{{"l1", 0}, {"l2", 1}};                                      // SLICE_LOCATION

struct Fixed {
    unsigned lo;
    unsigned width;
    uint32_t value;
};

struct InsnSpec {
    unsigned bits;      // 16 or 32
    uint32_t opcode;    // bits [6:0]
    std::vector<Fixed> fixed;
    std::vector<Operand> ops; // in assembly order
};

Operand R(unsigned lo) { return {Kind::Reg, lo, 5, false}; }
Operand S(unsigned lo) { return {Kind::Shape, lo, 3, false}; }
Operand U(unsigned lo, unsigned width) { return {Kind::Uimm, lo, width, false}; }
Operand I(unsigned lo, unsigned width) { return {Kind::Simm, lo, width, false}; }
Operand B(unsigned lo, unsigned width) { return {Kind::Branch, lo, width, false}; }
Operand opt(Operand o) {
    o.optional = true;
    return o;
}
Operand N(Operand o, const NameTable &names) {
    o.names = &names;
    return o;
}

InsnSpec insn32(uint32_t opcode, std::vector<Fixed> fixed, std::vector<Operand> ops) {
    return {32, opcode, std::move(fixed), std::move(ops)};
}

InsnSpec insn16(uint32_t opcode, std::vector<Fixed> fixed, std::vector<Operand> ops) {
    return {16, opcode, std::move(fixed), std::move(ops)};
}

// dm_conf / pu_conf / pu_pdp0_conf: the operands start with tcu_id[9:7], pu_id[12:10]; the selector is conf_id[16:13].
InsnSpec unit_conf(uint32_t opcode, uint32_t conf_id, std::vector<Operand> ops) {
    ops.insert(ops.begin(), {U(7, 3), U(10, 3)});
    return insn32(opcode, {{13, 4, conf_id}}, std::move(ops));
}

// mfu_conf: the selector is mfu_conf_id[11:7].
InsnSpec mfu_conf(uint32_t conf_id, std::vector<Operand> ops) {
    return insn32(0x62, {{7, 5, conf_id}}, std::move(ops));
}

// All unit opcodes with tcu_id[9:7], pu_id[12:10] as the first two operands.
InsnSpec unit_insn(uint32_t opcode, std::vector<Operand> ops) {
    ops.insert(ops.begin(), {U(7, 3), U(10, 3)});
    return insn32(opcode, {}, std::move(ops));
}

std::optional<InsnSpec> lookup(std::string_view mnemonic) {
    auto alu = [](uint32_t funct) { return insn32(0x0C, {{17, 5, funct}}, {R(7), R(12), R(22)}); };
    auto load = [](uint32_t funct) { return insn32(0x06, {{17, 3, funct}}, {R(7), R(12), I(20, 12)}); };
    // stores: the base register sits in the rd field [11:7], the data register in [16:12]
    auto store = [](uint32_t funct) { return insn32(0x08, {{17, 3, funct}}, {R(7), R(12), I(20, 12)}); };
    auto branch = [](uint32_t funct) { return insn32(0x10, {{17, 3, funct}}, {R(7), R(12), B(20, 12)}); };
    auto l2_conf = [](uint32_t opcode) {
        return insn32(opcode, {}, {S(7), S(10), N(U(13, 2), l2_datatype), N(U(15, 3), ddr_datatype)});
    };

    return StringSwitch<InsnSpec>(mnemonic)
        // ---- scalar core ----
        .Case("add", alu(0))
        .Case("sub", alu(1))
        .Case("mul", alu(2))
        .Case("div", alu(3))
        .Case("divu", alu(4))
        .Case("rem", alu(5))
        .Case("remu", alu(6))
        .Case("addi", insn32(0x0E, {{17, 3, 0}}, {R(7), R(12), I(20, 12)}))
        .Case("lui", insn32(0x02, {}, {R(7), U(12, 20)}))
        .Case("auipc", insn32(0x04, {}, {R(7), U(12, 20)}))
        .Case("lw", load(0))
        .Case("lh", load(1))
        .Case("lhu", load(2))
        .Case("lb", load(3))
        .Case("lbu", load(4))
        .Case("sw", store(0))
        .Case("sh", store(1))
        .Case("sb", store(2))
        .Case("beq", branch(0))
        .Case("bne", branch(1))
        .Case("blt", branch(2))
        .Case("bltu", branch(3))
        .Case("bge", branch(4))
        .Case("bgeu", branch(5))
        .Case("jal", insn32(0x12, {}, {R(7), B(12, 20)}))
        .Case("jalr", insn32(0x14, {{17, 3, 0}}, {R(7), R(12), I(20, 12)}))
        .Case("extrw", insn32(0x16, {}, {U(7, 10), R(17), opt(U(22, 10))}))
        .Case("extraw", insn32(0x18, {}, {U(7, 10), R(17), opt(U(22, 10))}))
        // ---- control (16 bit) ----
        .Case("intr", insn16(0x01, {}, {R(7)}))
        .Case("end", insn16(0x03, {}, {opt(R(7))}))
        .Case("fence", insn16(0x05, {}, {}))
        .Case("fence_i", insn16(0x07, {}, {}))
        .Case("ccr_decl", insn16(0x41, {}, {R(7)}))
        .Case("ccr_set", insn16(0x43, {}, {U(7, 5), U(12, 4)}))
        .Case("ccr_clr", insn16(0x45, {}, {U(7, 5)}))
        .Case("mmu_setid", insn16(0x49, {}, {R(7), U(12, 4)}))
        .Case("dm_conf_broadcast", insn16(0x4B, {}, {U(7, 3), U(10, 1), U(11, 1), U(12, 1)}))
        .Case("pu_compute", insn16(0x4D, {}, {U(7, 3), N(U(10, 2), pu_of_shift_mode)}))
        .Case("pu_pdp0_compute", insn16(0x4F, {}, {U(7, 3), R(10)}))
        .Case("ai2d_compute", insn16(0x51, {}, {}))
        // ---- MMU, shape registers, AI2D ----
        .Case("mmu_conf", insn32(0x44, {}, {R(7), R(12), U(17, 4)}))
        .Case("ss_pack_shape", insn32(0x40, {}, {R(7), R(12), R(17), R(22), S(27)}))
        .Case("ss_pack_stride", insn32(0x42, {}, {R(7), R(12), R(17), S(27)}))
        // ---- L2 ----
        .Case("l2_load_conf", l2_conf(0x46))
        .Case("l2_store_conf", l2_conf(0x4A))
        .Case("l2_load_w_conf",
              insn32(0x48, {}, {R(7), R(12), N(U(17, 2), l2_datatype), N(U(19, 3), ddr_datatype), U(22, 1)}))
        .Case("l2_load", insn32(0x4C, {}, {R(7), R(12), S(17)}))
        .Case("l2_store", insn32(0x4E, {}, {R(7), R(12), S(17)}))
        .Case("l2_load_w", insn32(0x57, {}, {R(7), R(12), R(17)}))
        // ---- DM ----
        .Case("dm_load_l1_conf", unit_conf(0x50, 0, {S(17), N(U(20, 2), l2_datatype), N(U(22, 2), l1_type)}))
        .Case("dm_load_w_conf", unit_conf(0x50, 1, {U(17, 5), U(22, 5), R(27)}))
        .Case("dm_load_w_conf_deq", unit_conf(0x50, 2, {N(U(17, 2), quant_type)}))
        .Case("dm_store_of_conf", unit_conf(0x50, 4, {S(17), N(U(20, 2), l2_datatype)}))
        .Case("dm_load_w_conf2", unit_conf(0x50, 5, {R(17), R(22)}))
        .Case("dm_load_l1", unit_insn(0x52, {R(13), R(18), S(23), N(U(26, 2), l1_type)}))
        .Case("dm_load_w", unit_insn(0x54, {R(13), R(18), S(23), N(U(26, 2), dm_load_w_dest)}))
        .Case("dm_load_act0", unit_insn(0x56, {R(13), R(18), N(U(23, 1), act0_channel), U(24, 1)}))
        .Case("dm_store_of", unit_insn(0x58, {R(13), S(18), N(U(21, 1), act0_channel)}))
        // ---- PU ----
        .Case("pu_fetchif_conf1", unit_conf(0x5A, 0, {U(17, 5), U(22, 5), S(27)}))
        .Case("pu_fetchif_conf2", unit_conf(0x5A, 1, {R(17), R(22)}))
        .Case("pu_fetchif_conf3", unit_conf(0x5A, 2, {R(17), R(22), S(27)}))
        .Case("pu_fetchif_conf4", unit_conf(0x5A, 3, {R(17), S(27)}))
        .Case("pu_fetchif_conf_deq", unit_conf(0x5A, 4, {R(17), R(22), N(U(27, 2), quant_type)}))
        .Case("pu_w_conf", unit_conf(0x5A, 5, {U(17, 5), U(22, 5)}))
        .Case("pu_of_conf1", unit_conf(0x5A, 6, {R(17), R(22), S(27)}))
        .Case("pu_of_conf2", unit_conf(0x5A, 7, {R(17), S(27)}))
        .Case("pu_compute_conf", unit_conf(0x5A, 8, {U(17, 1), U(18, 1), N(U(19, 1), pu_output_dest), U(20, 1),
                                                      N(U(21, 1), pu_compute_mode)}))
        .Case("pu_forward_psum", unit_insn(0x5C, {R(13), R(18)}))
        // ---- PU-PDP0 ----
        .Case("pu_pdp0_mode_conf", unit_conf(0x5E, 0, {U(17, 3)}))
        .Case("pu_pdp0_fetchif_conf1", unit_conf(0x5E, 1, {U(17, 5), U(22, 5)}))
        .Case("pu_pdp0_fetchif_conf2", unit_conf(0x5E, 2, {R(17), R(22)}))
        .Case("pu_pdp0_fetchif_conf3", unit_conf(0x5E, 3, {S(27)}))
        .Case("pu_pdp0_fetchif_conf4", unit_conf(0x5E, 4, {R(17), S(22)}))
        .Case("pu_pdp0_conf_deq", unit_conf(0x5E, 5, {R(17), U(22, 2)}))
        .Case("pu_pdp0_w_conf", unit_conf(0x5E, 6, {U(17, 5), U(22, 5)}))
        .Case("pu_pdp0_of_conf", unit_conf(0x5E, 7, {S(17), S(20)}))
        // ---- ACT0 ----
        // act0_src1_conf tcu_id, pu_id, channel, rshape [, shift]
        .Case("act0_src1_conf", insn32(0x60, {}, {U(7, 3), U(10, 3), N(U(16, 1), act0_channel), S(17), opt(U(20, 5))}))
        .Case("act0_compute", insn32(0x74, {}, {R(12), U(17, 3), N(U(20, 1), act0_channel), N(U(21, 2), act0_output_dest),
                                                 N(U(23, 2), act_output_type), U(25, 1)}))
        // ---- MFU ----
        .Case("mfu_transpose_conf", mfu_conf(0, {S(12), S(15), N(U(18, 2), l2_datatype), N(U(20, 5), mfu_trans_permute)}))
        .Case("mfu_pdp1_conf1", mfu_conf(1, {U(12, 5), U(17, 5), S(22), N(U(25, 2), pdp_function), S(27)}))
        .Case("mfu_pdp1_conf2", mfu_conf(2, {R(12), R(17), R(22), R(27)}))
        .Case("mfu_pdp1_conf3", mfu_conf(3, {R(12), R(17), R(22), S(27)}))
        .Case("mfu_pdp1_conf4", mfu_conf(4, {R(12), R(17), R(22), U(27, 1), U(28, 1)}))
        .Case("mfu_pdp1_conf_deq", mfu_conf(6, {R(12), R(17), N(U(22, 2), quant_type), opt(U(24, 5))}))
        .Case("mfu_pdp1_conf_quant", mfu_conf(7, {R(12), R(17), N(U(22, 2), quant_type), opt(U(24, 5))}))
        .Case("mfu_act1_conf_stride", mfu_conf(8, {S(12), S(15), S(18)}))
        .Case("mfu_act1_conf_src1", mfu_conf(9, {R(12), R(17), R(22), U(27, 1), N(U(28, 1), slice_location)}))
        .Case("mfu_act1_conf_src2", mfu_conf(10, {R(12), S(17), U(20, 1), N(U(21, 1), act1_source_type)}))
        .Case("mfu_act1_conf_dest", mfu_conf(11, {R(12), S(17)}))
        .Case("mfu_act1_conf_deq", mfu_conf(12, {R(12), R(17), N(U(22, 2), quant_type), U(24, 1), opt(U(25, 5))}))
        .Case("mfu_act1_conf_quant", mfu_conf(13, {N(U(12, 2), quant_type), opt(U(14, 5))}))
        .Case("mfu_act1_conf", mfu_conf(14, {N(U(12, 4), mfu_act1_function), U(16, 1), U(17, 1)}))
        .Case("mfu_memcpy", insn32(0x64, {}, {R(7), R(12), S(17), S(20), S(23)}))
        .Case("mfu_memset", insn32(0x66, {}, {R(7), R(12), S(17), S(20), N(U(23, 2), l2_datatype)}))
        .Case("mfu_transpose", insn32(0x68, {}, {R(7), R(12), S(17)}))
        .Case("mfu_pdp1_compute", insn32(0x6A, {}, {R(7), R(12), S(17)}))
        .Case("mfu_act1_compute", insn32(0x72, {}, {R(7), R(12), R(17), R(22)}))
        .Optional();
}

// ---------------------------------------------------------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------------------------------------------------------

std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
        s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
        s.remove_suffix(1);
    return s;
}

std::string lower(std::string_view s) {
    std::string r(s);
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return std::tolower(c); });
    return r;
}

bool is_identifier(std::string_view s) {
    if (s.empty() || !(std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_' || s[0] == '.'))
        return false;
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '.' || c == '$'; });
}

// "x12" -> 12 (limit 32), "s3" -> 3 (limit 8)
std::optional<unsigned> parse_register(std::string_view text, char prefix, unsigned limit) {
    if (text.size() < 2 || std::tolower(static_cast<unsigned char>(text[0])) != prefix)
        return std::nullopt;
    unsigned n = 0;
    auto [end, ec] = std::from_chars(text.data() + 1, text.data() + text.size(), n, 10);
    if (ec != std::errc() || end != text.data() + text.size() || n >= limit)
        return std::nullopt;
    return n;
}

uint64_t mask(unsigned width) { return width >= 64 ? ~0ull : ((1ull << width) - 1); }

// ---------------------------------------------------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------------------------------------------------

using Labels = std::map<std::string, uint32_t>;

uint64_t encode_operand(const Operand &op, std::string_view text, uint32_t pc, const Labels &labels, int line) {
    switch (op.kind) {
    case Kind::Reg: {
        if (lower(text) == "zero")
            return 0;
        auto r = parse_register(text, 'x', 32);
        if (!r)
            throw AsmError(line, "expected a register x0..x31, got '" + std::string(text) + "'");
        return *r;
    }
    case Kind::Shape: {
        // ss0..ss7 is how the compiler dumps them (SHAPE_REGISTER)
        const bool dump_spelling = text.size() > 2 && std::tolower(static_cast<unsigned char>(text[1])) == 's';
        auto r = parse_register(dump_spelling ? text.substr(1) : text, 's', 8);
        if (!r)
            throw AsmError(line, "expected a shape register s0..s7 or ss0..ss7, got '" + std::string(text) + "'");
        return *r;
    }
    case Kind::Uimm: {
        std::optional<int64_t> v;
        if (op.names) {
            for (const auto &[name, value] : *op.names)
                if (name == text)
                    v = value;
        }
        if (!v)
            v = parse_integer(text);
        if (!v)
            throw AsmError(line, "expected an integer" + std::string(op.names ? " or a value name" : "") + ", got '" + std::string(text) + "'");
        if (*v < 0 || static_cast<uint64_t>(*v) > mask(op.width))
            throw AsmError(line, "immediate " + std::string(text) + " does not fit in " + std::to_string(op.width) + " unsigned bits");
        return static_cast<uint64_t>(*v);
    }
    case Kind::Simm:
    case Kind::Branch: {
        int64_t value;
        if (auto v = parse_integer(text)) {
            value = *v;
        } else if (op.kind == Kind::Branch && is_identifier(text)) {
            auto it = labels.find(std::string(text));
            if (it == labels.end())
                throw AsmError(line, "undefined label '" + std::string(text) + "'");
            value = static_cast<int64_t>(it->second) - static_cast<int64_t>(pc);
        } else {
            throw AsmError(line, "expected an integer" + std::string(op.kind == Kind::Branch ? " or a label" : "") + ", got '" + std::string(text) + "'");
        }
        const int64_t lo = -(int64_t(1) << (op.width - 1));
        // a signed immediate may also be written as the raw field value (0..2^width-1): the compiler dumps addi that way
        const int64_t hi = op.kind == Kind::Simm ? int64_t(mask(op.width)) : (int64_t(1) << (op.width - 1)) - 1;
        if (value < lo || value > hi)
            throw AsmError(line, "value " + std::to_string(value) + " does not fit in " + std::to_string(op.width) + " signed bits");
        return static_cast<uint64_t>(value) & mask(op.width);
    }
    }
    throw AsmError(line, "internal error");
}

uint32_t encode(const InsnSpec &spec, const std::vector<std::string> &args, uint32_t pc, const Labels &labels, int line, const std::string &mnemonic) {
    size_t required = 0;
    for (const auto &op : spec.ops)
        if (!op.optional)
            ++required;
    if (args.size() < required || args.size() > spec.ops.size())
        throw AsmError(line, mnemonic + " expects " + std::to_string(required) + (required == spec.ops.size() ? "" : ".." + std::to_string(spec.ops.size())) + " operand(s), got " + std::to_string(args.size()));

    uint64_t word = spec.opcode;
    for (const auto &f : spec.fixed)
        word |= (static_cast<uint64_t>(f.value) & mask(f.width)) << f.lo;
    for (size_t i = 0; i < args.size(); ++i) {
        const auto &op = spec.ops[i];
        word |= encode_operand(op, args[i], pc, labels, line) << op.lo;
    }
    return static_cast<uint32_t>(word);
}

// ---------------------------------------------------------------------------------------------------------------------
// Assembler driver
// ---------------------------------------------------------------------------------------------------------------------

struct Item {
    int line;
    std::string mnemonic;
    std::vector<std::string> args;
};

std::vector<std::string> split_operands(std::string_view text, int line) {
    std::vector<std::string> args;
    text = trim(text);
    if (text.empty())
        return args;
    size_t pos = 0;
    for (;;) {
        size_t comma = text.find(',', pos);
        auto piece = trim(text.substr(pos, comma == std::string_view::npos ? comma : comma - pos));
        if (piece.empty())
            throw AsmError(line, "empty operand");
        args.emplace_back(piece);
        if (comma == std::string_view::npos)
            break;
        pos = comma + 1;
    }
    return args;
}

// Expands pseudo instructions into real ones.
std::vector<Item> expand(const Item &it, const SymbolTable &symbols) {
    auto need = [&](size_t n) {
        if (it.args.size() != n)
            throw AsmError(it.line, it.mnemonic + " expects " + std::to_string(n) + " operand(s), got " + std::to_string(it.args.size()));
    };
    if (it.mnemonic == "nop") {
        need(0);
        return {{it.line, "addi", {"x0", "x0", "0"}}};
    }
    if (it.mnemonic == "mv") {
        need(2);
        return {{it.line, "addi", {it.args[0], it.args[1], "0"}}};
    }
    if (it.mnemonic == "j") {
        need(1);
        return {{it.line, "jal", {"x0", it.args[0]}}};
    }
    if (it.mnemonic == "li") {
        need(2);
        auto v = evaluate(it.args[1], symbols);
        if (!v)
            throw AsmError(it.line, "li expects an integer or a symbol, got '" + it.args[1] + "'");
        if (*v < -(int64_t(1) << 31) || *v > int64_t(0xFFFFFFFFll))
            throw AsmError(it.line, "li value " + it.args[1] + " does not fit in 32 bits");
        if (*v >= -2048 && *v <= 2047)
            return {{it.line, "addi", {it.args[0], "x0", std::to_string(*v)}}};
        const uint32_t u = static_cast<uint32_t>(*v);
        int32_t lo = static_cast<int32_t>(u & 0xFFF);
        if (lo >= 2048)
            lo -= 4096;
        const uint32_t hi = (u - static_cast<uint32_t>(lo)) >> 12;
        std::vector<Item> r{{it.line, "lui", {it.args[0], std::to_string(hi)}}};
        if (lo != 0)
            r.push_back({it.line, "addi", {it.args[0], it.args[0], std::to_string(lo)}});
        return r;
    }
    return {it};
}

} // namespace

std::optional<int64_t> parse_integer(std::string_view text) {
    text = trim(text);
    bool negative = false;
    if (!text.empty() && (text[0] == '-' || text[0] == '+')) {
        negative = text[0] == '-';
        text.remove_prefix(1);
    }
    int base = 10;
    if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        base = 16;
        text.remove_prefix(2);
    } else if (text.size() > 2 && text[0] == '0' && (text[1] == 'b' || text[1] == 'B')) {
        base = 2;
        text.remove_prefix(2);
    }
    if (text.empty())
        return std::nullopt;
    uint64_t value = 0;
    auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (ec != std::errc() || end != text.data() + text.size() || value > uint64_t(INT64_MAX))
        return std::nullopt;
    return negative ? -static_cast<int64_t>(value) : static_cast<int64_t>(value);
}

std::optional<int64_t> evaluate(std::string_view text, const SymbolTable &symbols) {
    text = trim(text);
    if (auto v = parse_integer(text))
        return v;
    size_t end = 0;
    while (end < text.size() && (std::isalnum(static_cast<unsigned char>(text[end])) || text[end] == '_'))
        ++end;
    if (end == 0 || std::isdigit(static_cast<unsigned char>(text[0])))
        return std::nullopt;
    auto it = symbols.find(std::string(text.substr(0, end)));
    if (it == symbols.end())
        return std::nullopt;
    auto rest = trim(text.substr(end));
    if (rest.empty())
        return it->second;
    if (rest[0] != '+' && rest[0] != '-')
        return std::nullopt;
    std::string digits;   // "+ 16" is accepted as well
    for (char ch : rest)
        if (!std::isspace(static_cast<unsigned char>(ch)))
            digits += ch;
    auto offset = parse_integer(digits);   // "+16" / "-16"
    if (!offset)
        return std::nullopt;
    return it->second + *offset;
}

AsmProgram assemble(std::string_view source, const SymbolTable &symbols) {
    AsmProgram prog;

    // Pass 1: parse, expand pseudo instructions, assign addresses, collect labels.
    struct Placed {
        Item item;
        InsnSpec spec;
        uint32_t pc;
    };
    std::vector<Placed> placed;
    uint32_t pc = 0;
    int line_no = 0;
    size_t pos = 0;
    while (pos <= source.size()) {
        const size_t eol = source.find('\n', pos);
        std::string_view raw = source.substr(pos, eol == std::string_view::npos ? std::string_view::npos : eol - pos);
        pos = eol == std::string_view::npos ? source.size() + 1 : eol + 1;
        ++line_no;

        std::string_view text = trim(raw.substr(0, raw.find('#')));
        for (;;) { // leading labels
            const size_t colon = text.find(':');
            if (colon == std::string_view::npos)
                break;
            auto name = trim(text.substr(0, colon));
            if (!is_identifier(name))
                break;
            if (!prog.labels.emplace(std::string(name), pc).second)
                throw AsmError(line_no, "label '" + std::string(name) + "' is defined twice");
            text = trim(text.substr(colon + 1));
        }
        if (text.empty())
            continue;

        size_t ws = 0;
        while (ws < text.size() && !std::isspace(static_cast<unsigned char>(text[ws])))
            ++ws;
        Item item{line_no, lower(text.substr(0, ws)), split_operands(text.substr(ws), line_no)};

        for (auto &real : expand(item, symbols)) {
            auto spec = lookup(real.mnemonic);
            if (!spec)
                throw AsmError(line_no, "unknown instruction '" + real.mnemonic + "'");
            const uint32_t size = spec->bits / 8;
            placed.push_back({std::move(real), std::move(*spec), pc});
            pc += size;
        }
    }

    // Pass 2: encode.
    for (const auto &p : placed) {
        const uint32_t word = encode(p.spec, p.item.args, p.pc, prog.labels, p.item.line, p.item.mnemonic);
        prog.offsets.push_back(p.pc);
        prog.lines.push_back(p.item.line);
        for (unsigned b = 0; b < p.spec.bits / 8; ++b)
            prog.text.push_back(static_cast<uint8_t>(word >> (8 * b)));
    }
    return prog;
}

} // namespace k230::assembler
