// Lifted from IDA/Hex-Rays output (K230 NPU C-model simulator).
// Member names describe the decoded instruction fields; byte offsets are given in the header.
#include <iomanip>
#include "isa/insn_mfu_misc.h"
#include "globals.h"
#include "isa/kinstruction.h"
#include "engines/mfu.h"
#include "engines/simulator.h"
#include "iomanip"

// ---- MfuMemcpyInstruction ----
namespace {
// Returns `width` bits of `raw` starting at bit `lo`.
inline uint32_t field(uint64_t raw, unsigned lo, unsigned width)
{
    return static_cast<uint32_t>((raw >> lo) & ((1ull << width) - 1));
}
// Configuration vector logged by the MFU (209 bits).
using ConfigBits = std::bitset<209>;

// Stores the low `width` bits of `value` at bit position `lo` of `cfg`.
void put_bits(ConfigBits &cfg, unsigned lo, unsigned width, uint64_t value)
{
    for (unsigned i = 0; i < width; ++i)
        cfg[lo + i] = (value >> i) & 1;
}

// Writes the vector to the MFU log as 52 hex digits (MSB first; bit 0 is not printed), then a newline.
void log_config_bits(std::ofstream &log, const ConfigBits &cfg)
{
    const std::string bits = cfg.to_string();  // 209 characters, MSB first
    for (size_t pos = 0; pos < 208; pos += 4) {
        const uint64_t nibble = std::bitset<4>(bits, pos, 4).to_ulong();
        log << std::hex << std::setw(1) << std::setfill('0') << nibble;
    }
    log << std::endl;
}

// TODO(layout): the MFU singleton is accessed by raw byte offset (storage is MFU_MFUInst).

// Splits a 64-bit shape register into the four halfwords the MFU expects at byte offsets 32..38.
void store_shape_halfwords(uint8_t *mfu, uint64_t shape)
{
    uint16_t *hw = reinterpret_cast<uint16_t *>(mfu);
    hw[16] = static_cast<uint16_t>(shape >> 48);
    hw[17] = static_cast<uint16_t>(shape >> 32);
    hw[18] = static_cast<uint16_t>(shape >> 16);
    hw[19] = static_cast<uint16_t>(shape);
}
}  // namespace

// Simulator57.cpp  @0x418bc0
template <>
MfuMemcpyInstruction Simulator::InstParser<MfuMemcpyInstruction, 32>(uint8_t ** pc)
{
    MfuMemcpyInstruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(word)) -
                            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(g_DDR));

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.raddr_d_ = field(raw, 7, 5);
    inst.raddr_s_ = field(raw, 12, 5);
    inst.rstride_d_ = field(raw, 17, 3);
    inst.rstride_s_ = field(raw, 20, 3);
    inst.rshape_ = field(raw, 23, 3);
    inst.reserved_26_ = field(raw, 26, 6);
    inst.raddr_d_val_ = g_gp_reg[inst.raddr_d_];
    inst.raddr_s_val_ = g_gp_reg[inst.raddr_s_];
    inst.shape_a_ = g_shape_reg[inst.rstride_d_];
    inst.shape_b_ = g_shape_reg[inst.rstride_s_];
    inst.shape_c_ = g_shape_reg[inst.rshape_];
    // (IDA also computed MMU-translated rs1_val here, but the result was discarded.)
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    inst.name_ = *reinterpret_cast<const std::string *>(static_cast<uintptr_t>(insn_name_));  // insn_name: pointer to mnemonic
    *pc += 4;
    return inst;
}

// MfuMemcpyInstruction1.cpp  @0x41deb0
void MfuMemcpyInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuMemcpyInstruction2.cpp  @0x423420
void MfuMemcpyInstruction::operation()
{
    uint8_t *mfu = static_cast<uint8_t *>(MFU::GetMFU());
    uint32_t *mfu_words = reinterpret_cast<uint32_t *>(mfu);

    // Hand the operands to the MFU (TODO(layout): MFU object offsets 4, 8, 32..38, 2208).
    mfu_words[1] = raddr_s_val_;
    mfu_words[2] = raddr_d_val_;
    reinterpret_cast<MFU *>(mfu)->busy_ = 1;
    store_shape_halfwords(mfu, shape_c_);
    reinterpret_cast<MFU *>(mfu)->Memcpy();
    reinterpret_cast<MFU *>(mfu)->busy_ = 0;

    // Checkpoint log of the configuration that was used.
    // verified against asm @0x423420: the 208-bit bitset is zero-initialised (vpxor + 2x vmovdqu to the stack), so unset bits 193..207 are 0
    ConfigBits cfg;
    put_bits(cfg, 0, 7, opcode_ & 0x7F);
    put_bits(cfg, 7, 21, raddr_d_val_);
    put_bits(cfg, 28, 21, raddr_s_val_);
    put_bits(cfg, 49, 48, shape_a_);
    put_bits(cfg, 97, 48, shape_b_);
    put_bits(cfg, 145, 48, shape_c_);
    log_config_bits(reinterpret_cast<MFU *>(mfu)->log_[3], cfg);
}

// MfuMemcpyInstruction3.cpp  @0x4254f0
MfuMemcpyInstruction::~MfuMemcpyInstruction()
{
}

// ---- MfuMemsetInstruction ----
// Simulator58.cpp  @0x418e90
template <>
MfuMemsetInstruction Simulator::InstParser<MfuMemsetInstruction, 32>(uint8_t ** pc)
{
    MfuMemsetInstruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(word)) -
                            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(g_DDR));

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.raddr_d_ = field(raw, 7, 5);
    inst.rv_ = field(raw, 12, 5);
    inst.rstride_ = field(raw, 17, 3);
    inst.rshape_ = field(raw, 20, 3);
    inst.l2_datatype_ = field(raw, 23, 2);
    inst.reserved_25_ = field(raw, 25, 7);
    inst.raddr_d_val_ = g_gp_reg[inst.raddr_d_];
    inst.rv_val_ = g_gp_reg[inst.rv_];
    inst.shape_a_ = g_shape_reg[inst.rstride_];
    inst.shape_b_ = g_shape_reg[inst.rshape_];
    // MMU translation: 28-bit offset + 32 * table[addr >> 28].
    inst.rd_addr_ = (inst.raddr_d_val_ & 0xFFFFFFF) + 32 * MMU_MMUItem[2 * (inst.raddr_d_val_ >> 28)];
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    inst.name_ = *reinterpret_cast<const std::string *>(static_cast<uintptr_t>(insn_name_));  // insn_name: pointer to mnemonic
    *pc += 4;
    return inst;
}

// MfuMemsetInstruction1.cpp  @0x41dec0
void MfuMemsetInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuMemsetInstruction2.cpp  @0x423b60
void MfuMemsetInstruction::operation()
{
    uint8_t *mfu = static_cast<uint8_t *>(MFU::GetMFU());

    // Hand the operands to the MFU (TODO(layout): MFU object offsets 8, 64, 72, 98, 32..38, 2208).
    *reinterpret_cast<uint32_t *>(mfu + 8) = raddr_d_val_;
    *reinterpret_cast<uint16_t *>(mfu + 98) = static_cast<uint16_t>(rv_val_);
    mfu[72] = l2_datatype_;
    *reinterpret_cast<uint64_t *>(mfu + 64) = shape_a_;
    reinterpret_cast<MFU *>(mfu)->busy_ = 1;
    store_shape_halfwords(mfu, shape_b_);
    reinterpret_cast<MFU *>(mfu)->Memset();
    reinterpret_cast<MFU *>(mfu)->busy_ = 0;

    // Checkpoint log of the configuration that was used.
    // verified against asm @0x423b60: the 208-bit bitset is zero-initialised (vpxor + 2x vmovdqu to the stack), so unset bits 157..207 are 0
    ConfigBits cfg;
    put_bits(cfg, 0, 7, opcode_ & 0x7F);
    put_bits(cfg, 7, 21, raddr_d_val_);
    put_bits(cfg, 28, 8, rv_);      // low 8 bits of the rs1 field
    put_bits(cfg, 43, 48, shape_a_);
    put_bits(cfg, 91, 64, shape_b_);
    put_bits(cfg, 155, 2, l2_datatype_);
    log_config_bits(reinterpret_cast<MFU *>(mfu)->log_[3], cfg);
}

// MfuMemsetInstruction3.cpp  @0x4254a0
MfuMemsetInstruction::~MfuMemsetInstruction()
{
}

// ---- MfuTransposeInstruction ----
// Simulator60.cpp  @0x419380
template <>
MfuTransposeInstruction Simulator::InstParser<MfuTransposeInstruction, 32>(uint8_t ** pc)
{
    MfuTransposeInstruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(word)) -
                            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(g_DDR));

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.raddr_d_ = field(raw, 7, 5);
    inst.raddr_s_ = field(raw, 12, 5);
    inst.rshape_ = field(raw, 17, 3);
    inst.reserved_20_ = field(raw, 20, 12);  // verified against asm: wider immediate (unused by operation)
    inst.raddr_d_val_ = g_gp_reg[inst.raddr_d_];
    inst.raddr_s_val_ = g_gp_reg[inst.raddr_s_];
    inst.rshape_val_ = g_shape_reg[inst.rshape_];
    // (IDA also computed MMU-translated raddr_s_val here, but the result was discarded.)
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    inst.name_ = *reinterpret_cast<const std::string *>(static_cast<uintptr_t>(insn_name_));  // insn_name: pointer to mnemonic
    *pc += 4;
    return inst;
}

// MfuTransposeInstruction1.cpp  @0x41dee0
void MfuTransposeInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuTransposeInstruction2.cpp  @0x4213d0
void MfuTransposeInstruction::operation()
{
    // verified against asm @0x4213d0: the "globals" 0x54BB04..0x54BB26 are fields of the MFU singleton (0x54BB00):
    // +8 dst_addr <- raddr_d_val (32-bit store), +4 src_addr <- raddr_s_val, dim[3..0] (+0x26, +0x24, +0x22, +0x20) <- rshape_val
    // halfwords 0..3; then a tail call to MFU::Trans.
    MFU * const mfu = static_cast<MFU *>(MFU::GetMFU());
    mfu->dst_addr_ = raddr_d_val_;
    mfu->src_addr_ = raddr_s_val_;
    mfu->dim_[3] = static_cast<uint16_t>(rshape_val_);
    mfu->dim_[0] = static_cast<uint16_t>(rshape_val_ >> 48);
    mfu->dim_[1] = static_cast<uint16_t>(rshape_val_ >> 32);
    mfu->dim_[2] = static_cast<uint16_t>(rshape_val_ >> 16);
    mfu->Trans();
}

// MfuTransposeInstruction3.cpp  @0x425400
MfuTransposeInstruction::~MfuTransposeInstruction()
{
}

// ---- MfuTransposeConfInstruction ----
// Simulator59.cpp  @0x419120
template <>
MfuTransposeConfInstruction Simulator::InstParser<MfuTransposeConfInstruction, 32>(uint8_t ** pc)
{
    MfuTransposeConfInstruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(word)) -
                            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(g_DDR));

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.funct5_ = field(raw, 7, 5);
    inst.rstride_d_ = field(raw, 12, 3);
    inst.rstride_s_ = field(raw, 15, 3);
    inst.cfg_ = static_cast<uint16_t>(field(raw, 18, 2) | (field(raw, 20, 5) << 8));
    inst.reserved_25_ = field(raw, 25, 7);
    inst.rstride_d_val_ = g_shape_reg[inst.rstride_d_];
    inst.rstride_s_val_ = g_shape_reg[inst.rstride_s_];
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    inst.name_ = *reinterpret_cast<const std::string *>(static_cast<uintptr_t>(insn_name_));  // insn_name: pointer to mnemonic
    *pc += 4;
    return inst;
}

// MfuTransposeConfInstruction1.cpp  @0x41ded0
void MfuTransposeConfInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuTransposeConfInstruction2.cpp  @0x421230
void MfuTransposeConfInstruction::operation()
{
    // verified against asm @0x421230: MFU+0x50 / +0x58 (trans_shape_src / trans_shape_dst) <- {rstride_s_val, rstride_d_val}
    // (a 16-byte load of rstride_d_val:rstride_s_val with the halves swapped), and the 16-bit cfg word is stored at MFU+0x60
    // (trans_elem16 = low byte, trans_type = high byte).
    MFU * const mfu = static_cast<MFU *>(MFU::GetMFU());
    mfu->trans_shape_src_ = rstride_s_val_;
    mfu->trans_shape_dst_ = rstride_d_val_;
    mfu->trans_elem16_ = static_cast<uint8_t>(cfg_ & 0xFF);
    mfu->trans_type_ = static_cast<uint8_t>(cfg_ >> 8);
}

// MfuTransposeConfInstruction3.cpp  @0x425450
MfuTransposeConfInstruction::~MfuTransposeConfInstruction()
{
}
