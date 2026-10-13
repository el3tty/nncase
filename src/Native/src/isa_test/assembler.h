#pragma once
// A small assembler for the K230 NPU instruction set (see ../simulator/README.md for the encodings).
//
// Syntax (one instruction per line):
//
//   label:                      ; labels may share a line with an instruction
//       add   x7, x5, x6        ; registers: x0..x31 (zero = x0), shape registers: s0..s7
//       lw    x5, x0, 4         ; "lw rd, rs, offset" - the operand order of the compiler dumps
//       blt   x1, x9, label     ; branch / jal targets: label or byte offset relative to the instruction
//       li    x8, 0x1000        ; pseudo: lui + addi as needed (value in [-2^31, 2^32))
//       li    x9, DDR_IN+16     ; symbols (see below) and 'symbol +/- number' are accepted by li
//       end   x0
//
//   Comments start with '#'. Numbers: decimal, 0x.. hex, 0b.. binary, optional leading '-'.
//   Pseudo instructions: nop, mv rd, rs, li rd, imm, j label.
//
//   Symbols: the caller of assemble() can define named values (the test app defines DDR_IN and DDR_OUT, the addresses of
//   the DDR tensors, which are not known in advance on the device). Only `li` accepts them.
//
// Instruction operands are listed in assembly (compiler dump) order; fields that are not assembly operands in the
// compiler dumps (reserved bits, selectors) are encoded as zero.
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace k230::isa_test {

class AsmError : public std::runtime_error {
public:
    AsmError(int line, const std::string &message)
        : std::runtime_error("line " + std::to_string(line) + ": " + message), line_(line) {}

    int line() const noexcept { return line_; }

private:
    int line_;
};

struct AsmProgram {
    std::vector<uint8_t> text;                // little-endian instruction stream
    std::map<std::string, uint32_t> labels;   // label -> byte offset in `text`
    std::vector<uint32_t> offsets;            // byte offset of every instruction
};

using SymbolTable = std::map<std::string, int64_t>;

// Assembles `source`; throws AsmError.
AsmProgram assemble(std::string_view source, const SymbolTable &symbols = {});

// Evaluates `number`, `SYMBOL`, `SYMBOL+number` or `SYMBOL-number`. Returns nullopt if `text` is none of these.
std::optional<int64_t> evaluate(std::string_view text, const SymbolTable &symbols);

// Parses an integer literal (decimal, 0x, 0b, optional sign). Returns nullopt if `text` is not an integer.
std::optional<int64_t> parse_integer(std::string_view text);

} // namespace k230::isa_test
