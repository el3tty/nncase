#pragma once
// Top-level state of the K230 NPU instruction-set simulator.
// Lifted from IDA/Hex-Rays output (Simulator1 = destructor; Simulator2.. = per-ISA-class InstParser
// specialisations that live in the *instruction.cpp files).
#include <cstdint>
#include <string>
#include <vector>

class Simulator {
public:
    virtual ~Simulator();   // @0x410a70 (Simulator1)

    // Decode the instruction word at *pc into an instruction object of class T and advance *pc by N/8 bytes.
    // Only the primary template is declared here; every ISA class provides its own specialisation in
    // <class>instruction.cpp.
    template <class T, unsigned long N>
    T InstParser(uint8_t ** pc);

    // TODO(layout): the parsers read these two members through raw names, so the original
    // fNN names (= byte offset in the binary object) are kept.
    //
    // insn_name: address of the std::string holding the mnemonic of the instruction being decoded
    //       (parsers do  inst.name = *reinterpret_cast<const std::string *>(insn_name)).
    //       In the original binary this was an owned (COW) std::string member at +144; the lifted
    //       destructor therefore no longer releases anything for it.
    uint64_t insn_name_ = 0;      // +144
    // start_pc: offset of the code image inside DDR; parsers compute  pc_rel = pc - start_pc.
    uint32_t start_pc_ = 0;      // +440

    // Heap blocks released by the destructor (std::vector buffers in the original binary).
    // UNCERTAIN (asm @0x410a70 leaves it open): the destructor only calls operator delete(begin, cap_end - begin) on the
    // blocks at +160/+176 and +504/+520 (+144 is an owned COW std::string released with _M_dispose); the element type
    // of the vectors cannot be recovered from the destructor alone.
    std::vector<uint32_t> trace_log_;    // +160 (begin) / +168 (end) / +176 (capacity end)
    std::vector<uint32_t> aux_buffer_;   // +504 (begin) / +512 (end) / +520 (capacity end)

    // ---- state initialised inline by main() (verified against main @0x40b8fc..0x40ba8f; the offsets are those of the
    // binary's object, which has no vptr -- here the vptr of the lifted class shifts them by 8) ----
    uint8_t * ddr_ = nullptr;            // +0    DDR image (shared_memory data of argv[1])
    uint8_t * glb_banks_[16] = {};       // +8    copy of the 16 bank pointers handed to SimulatorInit
    uint8_t * pc_ptr_ = nullptr;         // +136  current instruction pointer (main also keeps a stack copy)
    uint32_t bit_offset_ = 0;            // +152  instructions executed so far, in bits (+0x20 per 32-bit, +0x10 per 16-bit insn)
    uint8_t has_base_ = 0;               // +156  set to 1 by FenceI (main starts it at 0)
};
