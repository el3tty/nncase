// Reconstructed main() of the nncase K230 NPU instruction-set simulator (C-model, nncase.simulator.k230.sc).
//
// Rewritten against the lifted class files and verified against the disassembly of `main` (sim.s,
// main @0x40b7f0 .. 0x40fc57, cold paths in main.cold @0x4058f0).  The previous version (work/main_old.cpp) was
// written from decompiler output before the classes existed and contained invented instruction names.
//
// Usage (argc must be 5, otherwise std::invalid_argument("The Argument Count != 5") is thrown):
//   simulator <ddr_shm> <glb_shm> <pc_offset> <argv4>
//     argv[1] : shared-memory object mapped as the 2 GiB DDR image            (shared_memory(path, 0x80000000, open))
//     argv[2] : shared-memory object mapped as the 4 MiB GLB bank 0 region     (shared_memory(path, 0x400000,  open))
//     argv[3] : start pc, byte offset into DDR (strtol, base 10) -> Simulator::start_pc_ and the initial pc
//     argv[4] : NOT referenced anywhere in main (verified: argv is only indexed with +8/+0x10/+0x18)
//
// Fetch/decode/execute loop (main @0x40baa8):
//   * every iteration first pushes Simulator::bit_offset_ onto Simulator::trace_log (vector<uint32_t>)
//   * the 32-bit word at pc is read; opcode = raw & 0x7F, funct3 = raw[19:17], funct5 = raw[21:17],
//     sub13 = raw[16:13] (DM / PU selector), sub7 = raw[11:7] (MFU selector)
//   * the matching Simulator::InstParser<I, 16|32>(&pc) decodes (and, for the scalar ALU/LSU insns, executes via
//     parser_operation()) and advances pc by 2 or 4; for most NPU insns main then calls I::operation()
//   * control-flow insns (Beq..Bgeu, Jal, Jalr) retarget pc to _G.DDR + next_pc when `taken` is set
//   * bit_offset += 32 (or 16 for the 16-bit encodings) after each instruction
//   * End and FenceI leave the loop and main returns 0; Intr throws runtime_error("INTR!");
//     an unknown opcode prints the "unsupported instruction in Cmodel, Skip!" message and throws
//     runtime_error("Invaild Opcode") (sic).
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <filesystem>

#include <nncase/runtime/k230/shared_memory.h>
#include "engines/checkpoint.h"
#include "engines/conv2d.h"
#include "engines/pdp0.h"
#include "engines/meshnet.h"
#include "engines/mfu.h"
#include "globals.h"
#include "engines/simulator.h"
#include "isa/kinstruction.h"
#include "isa/insn_act0.h"
#include "isa/insn_alu.h"
#include "isa/insn_ai2d.h"
#include "isa/insn_branch.h"
#include "isa/insn_system.h"
#include "isa/insn_dm.h"
#include "isa/insn_l2.h"
#include "isa/insn_mem.h"
#include "isa/insn_mfu_act1.h"
#include "isa/insn_mfu_misc.h"
#include "isa/insn_mfu_pdp1.h"
#include "isa/insn_pu.h"
#include "isa/insn_pupdp0.h"


using nncase::runtime::k230::shared_memory;
using nncase::runtime::k230::shared_memory_openmode;

// Opcode = raw & 0x7F (verified against the compare chain at main @0x40bb02..0x40f8ed).
// 32-bit encodings have an even opcode >= 2 with bit0 == 0 ... except where noted; 16-bit encodings are listed as such.
enum Opcode : uint32_t {
    // 16-bit encodings (pc += 2)
    OP_INTR        = 0x01,  // IntrInstruction          @0x40e47b  -> throws "INTR!"
    OP_END         = 0x03,  // EndInstruction           @0x40db5e  -> leaves the loop
    OP_FENCE       = 0x05,  // FenceInstruction         @0x40e1d2
    OP_FENCEI      = 0x07,  // FenceIInstruction        @0x40dd3f  -> sets has_base, leaves the loop
    OP_CCR_DECL    = 0x41,  // CcrDeclInstruction       @0x40d7d0
    OP_CCR_SET     = 0x43,  // CcrSetInstruction        @0x40e341
    OP_CCR_CLR     = 0x45,  // CcrClrInstruction        @0x40de1e
    OP_MMU_SETID   = 0x49,  // MmuSetidInstruction      @0x40d97b
    OP_DM_CONF_BCAST = 0x4B,// DmConf_broadcastInstruction (16-bit)
    OP_PU_COMPUTE  = 0x4D,  // PuComputeInstruction     @0x40ec45
    OP_PU_PDP0_COMPUTE = 0x4F, // PuPdp0ComputeInstruction @0x40f0fa
    OP_AI2D_COMPUTE= 0x51,  // Ai2dComputeInstruction   @0x40fb5a
    // 32-bit scalar encodings (pc += 4)
    OP_LUI   = 0x02, OP_AUIPC = 0x04, OP_LOAD = 0x06, OP_STORE = 0x08,
    OP_ALU   = 0x0C,        // funct5 selects Add/Sub/Mul/Div/Divu/Rem/Remu
    OP_ADDI  = 0x0E,        // only funct3 == 0
    OP_BRANCH= 0x10,        // funct3 selects Beq/Bne/Blt/Bltu/Bge/Bgeu
    OP_JAL   = 0x12, OP_JALR = 0x14, OP_EXTRW = 0x16, OP_EXTRAW = 0x18,
    // 32-bit NPU encodings
    OP_SS_PACK_SHAPE = 0x40, OP_SS_PACK_STRIDE = 0x42, OP_MMU_CONF = 0x44,
    OP_L2_LOAD_CONF = 0x46, OP_L2_LOADW_CONF = 0x48, OP_L2_STORE_CONF = 0x4A,
    OP_L2_LOAD = 0x4C, OP_L2_STORE = 0x4E,
    OP_DM_CONF = 0x50,      // sub13: 0 LoadL1Conf, 1 LoadWConf, 2 LoadWConf_deq, 4 StoreOfConf, 5 LoadWConf2
    OP_DM_LOAD_L1 = 0x52, OP_DM_LOAD_W = 0x54, OP_DM_LOAD_ACT0 = 0x56, OP_DM_STORE_OF = 0x58,
    OP_L2_LOADW = 0x57,
    OP_PU_CONF = 0x5A,      // sub13: 0..8
    OP_PU_FORWARD_PSUM = 0x5C,
    OP_PU_PDP0_CONF = 0x5E, // sub13: 0..7
    OP_ACT0_SRC1_CONF = 0x60,
    OP_MFU_CONF = 0x62,     // sub7: 0..14 (5 is unused)
    OP_MFU_MEMCPY = 0x64, OP_MFU_MEMSET = 0x66, OP_MFU_TRANSPOSE = 0x68, OP_MFU_PDP1_COMPUTE = 0x6A,
    OP_MFU_ACT1_COMPUTE = 0x72, OP_ACT0_COMPUTE = 0x74,
};

// Parse one instruction of class I (N = 16 or 32 bits).  When OP is set the asm calls I::operation() right after the
// parser; when BR is set the (already advanced) pc is replaced by _G.DDR + next_pc if the instruction was taken.
// bit_offset is advanced by N afterwards in every case.
template <class I, unsigned N, bool OP = false, bool BR = false>
static inline void step(Simulator & sim, uint8_t *& pc)
{
    I inst = sim.InstParser<I, N>(&pc);
    if constexpr (OP) inst.operation();
    if constexpr (BR) { if (inst.taken_) pc = _G.DDR + inst.next_pc_; }
    sim.bit_offset_ += N;
}
// parse only                       parse + operation()              parse + operation() + branch retarget
#define P32(I)   step<I, 32>(sim, pc)
#define PO32(I)  step<I, 32, true>(sim, pc)
#define PB32(I)  step<I, 32, true, true>(sim, pc)
#define PO16(I)  step<I, 16, true>(sim, pc)
#define P16(I)   step<I, 16>(sim, pc)

static int simulator_run(int argc, const char ** argv)
{
    // The DLL build stays loaded between invocations: start every run from a clean global state.
    initialize_globals();

    // verified against asm @0x40b801 / main.cold @0x4062ee: std::invalid_argument("The Argument Count != 5")
    if (argc != 5)
        throw std::invalid_argument("The Argument Count != 5");

    srand(0x14);   // verified @0x40b817: srand(20)

    // argv[1] -> DDR image (2 GiB), argv[2] -> GLB region (4 MiB); openmode == 1 (attach to the existing object)
    shared_memory ddr_mem(std::filesystem::path(std::string(argv[1])), 0x80000000ULL, shared_memory_openmode::open);  // @0x40b85d
    shared_memory glb_mem(std::filesystem::path(std::string(argv[2])), 0x400000, shared_memory_openmode::open);       // @0x40b8b0

    uint8_t * const ddr_base = static_cast<uint8_t *>(ddr_mem.data());

    // Bank table: bank 0 = the mapped 4 MiB region, banks 1..15 = zero-filled new[0x400000] (@0x40b8d5..0x40b8fa).
    // The original leaks them until process exit; here they are released when the run ends (the simulator may be
    // started repeatedly from one process when built as a DLL).
    uint8_t * banks[16];
    std::vector<std::unique_ptr<uint8_t[]>> bank_storage;
    banks[0] = static_cast<uint8_t *>(glb_mem.data());
    for (int i = 1; i < 16; ++i)
    {
        bank_storage.emplace_back(new uint8_t[0x400000]());
        banks[i] = bank_storage.back().get();
    }

    SimulatorInit(ddr_base, banks);     // @0x40b907, publishes _G.DDR / _G.GLB

    // Simulator object is built inline in main (@0x40b90c..0x40ba1d): ddr pointer, copy of the bank table,
    // bit_offset = 0, has_base = 0, empty trace vectors, start_pc = 0.
    Simulator sim;
    sim.ddr_ = ddr_base;
    memcpy(sim.glb_banks_, banks, sizeof(banks));
    sim.bit_offset_ = 0;
    sim.has_base_ = 0;

    std::vector<std::shared_ptr<KInstruction>> insts;   // @0x40(%rsp): never filled, only destroyed (@0x40dca2)

    // argv[3]: start offset (strtol base 10), stored in Simulator::start_pc (@0x40ba1d, sign-extended to 64 bit)
    int64_t start = strtol(argv[3], nullptr, 10);
    sim.start_pc_ = (uint32_t)start;
    uint8_t * pc = sim.ddr_ + start;
    if (sim.has_base_)                         // @0x40ba26/0x40ba4d; has_base is 0 here, so this adds nothing
        pc += sim.bit_offset_ >> 3;
    sim.pc_ptr_ = pc;
    sim.bit_offset_ = 0;                       // @0x40ba72
    sim.trace_log_.clear();                    // @0x40ba7d: end = begin

    for (;;)
    {
        sim.trace_log_.push_back(sim.bit_offset_);                        // @0x40bab8 (vector<uint>::push_back)

        const uint32_t raw    = *reinterpret_cast<uint32_t *>(pc);      // @0x40bae1 movl (%rax),%ecx
        const uint32_t opcode = raw & 0x7F;
        const uint32_t funct3 = (raw >> 17) & 0x7;
        const uint32_t funct5 = (raw >> 17) & 0x1F;
        const uint32_t sub13  = (raw >> 13) & 0xF;
        const uint32_t sub7   = (raw >> 7) & 0x1F;

        switch (opcode)
        {
        // ---------------- scalar core (parser_operation() executes them, pc += 4) ----------------
        case OP_ADDI:
            if (funct3 != 0) goto unsupported;                          // verified @0x40bd6e
            P32(AddiInstruction); break;
        case OP_LUI:   P32(LuiInstruction);   break;
        case OP_AUIPC: P32(AuipcInstruction); break;
        case OP_ALU:                                                    // funct5 = raw[21:17]
            switch (funct5) {
            case 0: P32(AddInstruction);  break;
            case 1: P32(SubInstruction);  break;
            case 2: P32(MulInstruction);  break;
            case 3: P32(DivInstruction);  break;
            case 4: P32(DivuInstruction); break;
            case 5: P32(RemInstruction);  break;
            case 6: P32(RemuInstruction); break;   // out-of-line InstParser<Remu,32> @0x410d50
            default: goto unsupported;
            }
            break;
        case OP_LOAD:                                                   // funct3 = raw[19:17]
            switch (funct3) {
            case 0: P32(LwInstruction);  break;
            case 1: P32(LhInstruction);  break;
            case 2: P32(LhuInstruction); break;
            case 3: P32(LbInstruction);  break;
            case 4: P32(LbuInstruction); break;
            default: goto unsupported;
            }
            break;
        case OP_STORE:
            switch (funct3) {
            case 0: P32(SwInstruction); break;
            case 1: P32(ShInstruction); break;
            case 2: P32(SbInstruction); break;
            default: goto unsupported;
            }
            break;
        case OP_BRANCH:                                                 // parse, operation(), retarget pc if taken
            switch (funct3) {
            case 0: PB32(BeqInstruction);  break;
            case 1: PB32(BneInstruction);  break;
            case 2: PB32(BltInstruction);  break;
            case 3: PB32(BltuInstruction); break;
            case 4: PB32(BgeInstruction);  break;
            case 5: PB32(BgeuInstruction); break;
            default: goto unsupported;
            }
            break;
        case OP_JAL:   PB32(JalInstruction);  break;                    // @0x40e628
        case OP_JALR:  PB32(JalrInstruction); break;                    // @0x40df58
        case OP_EXTRW:  PO32(ExtrwInstruction);  break;                 // @0x40e057
        case OP_EXTRAW: PO32(ExtrawInstruction); break;                 // @0x40f8f7

        // ---------------- 16-bit control encodings ----------------
        case OP_INTR:                                                   // @0x40e47b, then jmp main.cold+0x1394
            sim.InstParser<IntrInstruction, 16>(&pc);
            throw std::runtime_error("INTR!");
        case OP_END:                                                    // @0x40db5e: decode, then leave the loop
            sim.InstParser<EndInstruction, 16>(&pc);
            sim.bit_offset_ += 16;
            return 0;
        case OP_FENCEI:                                                 // @0x40dd3f
            sim.InstParser<FenceIInstruction, 16>(&pc);
            sim.bit_offset_ += 16;
            sim.has_base_ = 1;                                           // movb $1,0x21c(%rsp), then jmp to the exit path
            return 0;
        case OP_FENCE:    P16(FenceInstruction);   break;               // @0x40e1d2
        case OP_CCR_DECL: P16(CcrDeclInstruction); break;               // @0x40d7d0 (no operation() call)
        case OP_CCR_SET:  PO16(CcrSetInstruction); break;               // @0x40e341 (operation() is the empty stub 0x41d820)
        case OP_CCR_CLR:  PO16(CcrClrInstruction); break;               // @0x40de1e (same)
        case OP_MMU_SETID: PO16(MmuSetidInstruction); break;            // parser_operation() + operation()
        case OP_AI2D_COMPUTE: PO16(Ai2dComputeInstruction); break;      // @0x40fb5a (empty operation stub)
        case OP_DM_CONF_BCAST: PO16(DmConf_broadcastInstruction); break;
        case OP_PU_COMPUTE:      PO16(PuComputeInstruction);     break;
        case OP_PU_PDP0_COMPUTE: PO16(PuPdp0ComputeInstruction); break;

        // ---------------- SS / MMU / L2 ----------------
        case OP_SS_PACK_SHAPE:  P32(SsPackShapeInstruction);  break;    // no operation() call
        case OP_SS_PACK_STRIDE: P32(SsPackStrideInstruction); break;    // no operation() call
        case OP_MMU_CONF:       PO32(MmuConfInstruction);     break;
        case OP_L2_LOAD_CONF:   PO32(L2LoadConfInstruction);  break;
        case OP_L2_LOADW_CONF:  PO32(L2LoadWConfInstruction); break;
        case OP_L2_STORE_CONF:  PO32(L2StoreConfInstruction); break;
        case OP_L2_LOAD:        PO32(L2LoadInstruction);      break;
        case OP_L2_STORE:       PO32(L2StoreInstruction);     break;
        case OP_L2_LOADW:       PO32(L2LoadWInstruction);     break;

        // ---------------- DM ----------------
        case OP_DM_CONF:
            switch (sub13) {
            case 0: PO32(DmLoadL1ConfInstruction);     break;
            case 1: PO32(DmLoadWConfInstruction);      break;
            case 2: PO32(DmLoadWConf_deqInstruction);  break;
            case 4: PO32(DmStoreOfConfInstruction);    break;
            case 5: PO32(DmLoadWConf2Instruction);     break;   // operation() = empty stub
            default: goto unsupported;
            }
            break;
        case OP_DM_LOAD_L1:   PO32(DmLoadL1Instruction);   break;
        case OP_DM_LOAD_W:    PO32(DmLoadWInstruction);    break;
        case OP_DM_LOAD_ACT0: PO32(DmLoadAct0Instruction); break;
        case OP_DM_STORE_OF:  PO32(DmStoreOfInstruction);  break;

        // ---------------- PU ----------------
        case OP_PU_CONF:
            switch (sub13) {
            case 0: PO32(PuFetchifConf1Instruction);    break;
            case 1: PO32(PuFetchifConf2Instruction);    break;      // operation() = empty stub
            case 2: PO32(PuFetchifConf3Instruction);    break;
            case 3: PO32(PuFetchifConf4Instruction);    break;
            case 4: PO32(PuFetchifConf_deqInstruction); break;
            case 5: PO32(PuWConfInstruction);           break;
            case 6: PO32(PuOfConf1Instruction);         break;
            case 7: PO32(PuOfConf2Instruction);         break;
            case 8: PO32(PuComputeConfInstruction);     break;
            default: goto unsupported;
            }
            break;
        case OP_PU_FORWARD_PSUM: PO32(PuForward_psumInstruction); break;
        case OP_PU_PDP0_CONF:
            switch (sub13) {
            case 0: PO32(PuPdp0ModeConfInstruction);     break;
            case 1: PO32(PuPdp0FetchifConf1Instruction); break;
            case 2: PO32(PuPdp0FetchifConf2Instruction); break;
            case 3: PO32(PuPdp0FetchifConf3Instruction); break;
            case 4: PO32(PuPdp0FetchifConf4Instruction); break;
            case 5: PO32(PuPdp0Conf_deqInstruction);     break;
            case 6: PO32(PuPdp0WConfInstruction);        break;
            case 7: PO32(PuPdp0OfConfInstruction);       break;
            default: goto unsupported;
            }
            break;

        // ---------------- ACT0 / MFU ----------------
        case OP_ACT0_SRC1_CONF: PO32(Act0Src1ConfInstruction);   break;
        case OP_ACT0_COMPUTE:   PO32(Act0ComputeInstruction);    break;
        case OP_MFU_MEMCPY:     PO32(MfuMemcpyInstruction);      break;
        case OP_MFU_MEMSET:     PO32(MfuMemsetInstruction);      break;
        case OP_MFU_TRANSPOSE:  PO32(MfuTransposeInstruction);   break;
        case OP_MFU_PDP1_COMPUTE: PO32(MfuPdp1ComputeInstruction); break;
        case OP_MFU_ACT1_COMPUTE: PO32(MfuAct1ComputeInstruction); break;
        case OP_MFU_CONF:                                               // sub7 = raw[11:7]
            switch (sub7) {
            case 0:  PO32(MfuTransposeConfInstruction);   break;
            case 1:  PO32(MfuPdp1Conf1Instruction);       break;
            case 2:  PO32(MfuPdp1Conf2Instruction);       break;
            case 3:  PO32(MfuPdp1Conf3Instruction);       break;
            case 4:  PO32(MfuPdp1Conf4Instruction);       break;
            case 6:  PO32(MfuPdp1Conf_deqInstruction);    break;
            case 7:  PO32(MfuPdp1Conf_quantInstruction);  break;
            case 8:  PO32(MfuAct1ConfStrideInstruction);  break;
            case 9:  PO32(MfuAct1ConfSrc1Instruction);    break;
            case 10: PO32(MfuAct1ConfSrc2Instruction);    break;
            case 11: PO32(MfuAct1ConfDestInstruction);    break;
            case 12: PO32(MfuAct1Conf_deqInstruction);   break;
            case 13: PO32(MfuAct1Conf_quantInstruction);  break;
            case 14: PO32(MfuAct1ConfInstruction);        break;
            default: goto unsupported;
            }
            break;

        default:
        unsupported:
            // verified against asm @0x40facb..0x40fb55 (strings at .rodata 0x4800c8 / 0x480046 / 0x48004a)
            std::cout << "unsupported instruction in Cmodel, Skip!" << std::endl;
            std::cout << "pc " << (size_t)((pc - _G.DDR) - (int64_t)sim.start_pc_) << std::endl;
            std::cout << "opcode " << (size_t)*pc << std::endl;
            throw std::runtime_error("Invaild Opcode");                 // main.cold @0x4058f0 (string @0x480052)
        }
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------------------------------------

// Runs the simulator and turns exceptions into an error code (they must not cross the DLL boundary, and a plain
// terminate() is of little help when debugging).
static int simulator_guarded_run(int argc, const char ** argv)
{
    int result;
    try
    {
        result = simulator_run(argc, argv);
    }
    catch (const std::exception & e)
    {
        std::cerr << "simulator: " << e.what() << std::endl;
        result = 1;
    }
    CheckPoint::GetCheckPoint()->Reset();
    Conv2D::GetConv2D()->Reset();
    static_cast<PDP0 *>(PDP0::GetPDP0())->Reset();
    MeshNet::GetMeshNet()->Reset();
    static_cast<MFU *>(MFU::GetMFU())->Reset();
    return result;
}

#ifdef K230_SIMULATOR_DLL

#ifdef _WIN32
#define K230_SIMULATOR_API __declspec(dllexport)
#else
#define K230_SIMULATOR_API __attribute__((visibility("default")))
#endif

// Splits a command line into arguments: separated by white space, double quotes group (and are removed),
// no escape characters (backslashes are kept, so Windows paths work).
static std::vector<std::string> split_commandline(const char * cmdline)
{
    std::vector<std::string> args;
    std::string cur;
    bool in_quotes = false, have = false;
    for (const char * p = cmdline ? cmdline : ""; *p; ++p)
    {
        const char c = *p;
        if (c == '"')
        {
            in_quotes = !in_quotes;
            have = true;
        }
        else if (!in_quotes && (c == ' ' || c == '\t' || c == '\r' || c == '\n'))
        {
            if (have)
                args.push_back(std::move(cur)), cur.clear(), have = false;
        }
        else
        {
            cur.push_back(c);
            have = true;
        }
    }
    if (have)
        args.push_back(std::move(cur));
    return args;
}

// The only function exported by the DLL build. `commandline` holds the arguments of the executable
// (without the program name):  <ddr_shm> <glb_shm> <pc_offset> <argv4>
// Returns 0 on success and non-zero on failure (the reason is printed to stderr).
extern "C" K230_SIMULATOR_API int SimulatorMain(const char * commandline)
{
    std::vector<std::string> args = split_commandline(commandline);
    args.insert(args.begin(), "SimulatorMain");
    std::vector<const char *> argv;
    for (const auto & a : args)
        argv.push_back(a.c_str());
    return simulator_guarded_run(static_cast<int>(argv.size()), argv.data());
}

#else

int main(int argc, const char ** argv)
{
    return simulator_guarded_run(argc, argv);
}

#endif
