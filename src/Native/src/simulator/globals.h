#pragma once
// Free functions and global state of the K230 NPU C-model simulator (lifted from IDA/Hex-Rays).
//
//  * Section 1: C++ helpers (_global1.._global26 in sources/).
//  * Section 2: C fixed-point helpers (shr_rnd_int.c ... linear_quant.c).
//  * Section 3: named global state shared by all instruction classes and NPU units.
//  * Section 4: anonymous IDA placeholders (unk_/dword_/word_/byte_ at addresses of the original image).
//
// All symbols have C++ linkage (the original binary mangles them as C++ names, e.g.
// _Z12set_g_gp_reghj; the fixed-point helpers are the only plain-C symbols, but nothing links
// against them from C so no extern "C" is used).
#include <cstdint>
#include <stdexcept>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <bitset>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "math/numeric_types.h"
#include "engines/memaccessor.h"

enum SPU_TYPE_ID : int;   // UNCERTAIN: enumerators are not visible in the dumps; printed as a number

// ===================================================================================================
// 1. C++ helpers
// ===================================================================================================

// @0x41fc50 (_global1)  g_gp_reg[reg] = value; x0 is hard-wired to zero.
void set_g_gp_reg(uint8_t reg, uint32_t value);

// @0x41fcc0 / @0x420020 / @0x420390 (_global2..4)  One trace line "AAAAAAAA T NN DDDD" (hex, zero filled;
// the data field is 16 / 8 / 4 digits wide for the uint64_t / uint32_t / uint16_t overload).
// UNCERTAIN: meaning of the first three fields (address, SPU data type, element tag).
void print_spu_data(uint32_t addr, SPU_TYPE_ID type, uint32_t tag, uint64_t data, std::ofstream & out);
void print_spu_data(uint32_t addr, SPU_TYPE_ID type, uint32_t tag, uint32_t data, std::ofstream & out);
void print_spu_data(uint32_t addr, SPU_TYPE_ID type, uint32_t tag, uint16_t data, std::ofstream & out);

// @0x42d3e0 (_global5)  Publishes the DDR image and the 16 GLB bank pointers (g_DDR / g_GLB).
void SimulatorInit(uint8_t * ddr, uint8_t ** glb_banks);

// @0x42d4f0 (_global6)  Copies the low `width` bits of `value` into `bits` starting at bit `lsb`.
void set_bit(std::bitset<32> & bits, int width, int lsb, int value);

// @0x42d560 (_global7)  Dumps DDR read data: one line per entry of `byte_masks`, `ddr_burst_len` hex bytes
// per line (most significant byte first).
void print_r_data(std::vector<uint16_t> & byte_masks, std::vector<uint8_t> & data, std::ofstream & out);

// @0x42d910 (_global8)  One 8-digit hex address per line.
void print_addr(std::vector<uint32_t> & addrs, std::vector<uint8_t> & unused, std::ofstream & out);

// @0x42da80 (_global9) / @0x42de80 (_global10)  Byte <-> bit vector conversion (bit 0 first).
std::vector<bool> ByteToBits(uint8_t value);
std::vector<uint8_t> BitsToByte(std::vector<bool> const & bits);

// @0x42e340 (_global11)  Splits a DDR transfer of `elem_bytes * num_elems` bytes starting at `ddr_addr` into
// bursts: aligned burst addresses, burst count minus one per DDR group and 16-bit byte-enable masks.
void calc_ddr_param(std::vector<uint32_t> & burst_addr, std::vector<uint8_t> & burst_count_m1,
                    std::vector<uint16_t> & byte_mask, uint8_t elem_bytes, uint32_t num_elems, uint32_t ddr_addr);

// @0x42ec80 (_global12)  Writes the debug dumps requested through debug_map (L3 / L2 / registers).
void dump_data_proc(std::string prefix);

// @0x4310a0 (_global13)  Parses the debug command line (see debug_function) and stores the result in `out`
// under the keys "pc", "dump_L3_start", "dump_L3_len", "dump_L2", "dump_reg".
void Stringsplit(std::string const & text, std::string const & delim,
                 std::map<std::string, std::vector<uint32_t>> & out);

// @0x4329b0 (_global14)  Interactive debugger hook, called once per instruction with the current pc.
void debug_function(uint32_t pc, std::string prefix);

// @0x4385a0 (_global15)  Hex dump of `size` bytes to stdout, 16 per line.
void memory_dump(void const * data, int size);

// @0x439930 (_global16) / @0x4399c0 (_global17)  Same algorithms as TileHelper::BF16ToQint / GetAddress.
int64_t BF16ToQint(BF16::bfloat16 value);
int64_t GetAddress(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,
                   uint32_t stride0, uint32_t stride1, uint32_t stride2, uint8_t layout);

// @0x441de0 / @0x4421c0 (_global18/19)  GLB access trace of the MFU: collects bytes into a 32-byte line and
// writes it out when full. `byte_valid` has one bit per entry of `line` (both are filled front-to-back in
// reverse address order).
void print_glb_read_core(std::vector<uint8_t> & line, std::vector<bool> & byte_valid, uint32_t glb_addr,
                         std::ofstream & addr_log, std::ofstream & data_log, uint32_t & valid_so_far,
                         uint32_t total_bytes, bool last);
void print_glb_write_core(std::vector<uint8_t> & line, std::vector<bool> & byte_valid, uint32_t glb_addr,
                          std::ofstream & log, uint32_t & valid_so_far, uint32_t total_bytes, bool last);

// @0x443460 / @0x443b00 (_global20/21)  Walk a 4-D tile of a GLB bank and emit the access trace.
void mfu_glb_read(uint64_t shape, uint16_t cnt3, uint16_t cnt2, uint16_t cnt1, uint16_t cnt0, uint32_t glb_addr,
                  uint8_t elem_log2, MemAccessor mem, std::ofstream & addr_log, std::ofstream & data_log,
                  uint32_t total_bytes);
void mfu_glb_write(uint64_t shape, uint16_t cnt3, uint16_t cnt2, uint16_t cnt1, uint16_t cnt0, uint32_t glb_addr,
                   uint8_t elem_log2, MemAccessor mem, std::ofstream & log, uint32_t total_bytes, uint8_t row_mode);

// @0x464c30 (_global22)  Splits `text` at every `delim` (std::getline semantics, empty fields kept).
void Stringsplit(std::string text, char delim, std::vector<std::string> & out);

// @0x4672d0..@0x467330 (_global23..26)
int fp_to_int32(float value);                                   // round to nearest even, then truncate
int64_t xy_climp(int64_t value, int64_t lo, int64_t hi);        // clamp
int64_t mirror_climp(int64_t value, int64_t lo, int64_t hi);    // reflect at the borders
void Mat_Inv(double * m);                                       // in-place inverse of a 2x3 affine matrix

// ===================================================================================================
// 2. Fixed-point helpers (plain C functions in the original)
// ===================================================================================================
int64_t shl_2_xbit(int64_t value, int shift, int xbits);   // @0x46b0f0
int32_t shr_rnd_int(int32_t value, int shift, int round_mode);
int64_t shr_rnd_lint(int64_t value, int shift, int round_mode);
int32_t sat_int_xbits(int32_t value, int xbits);
int64_t sat_lint_xbits(int64_t value, int xbits);
int64_t mul_int(int32_t a, int32_t b, int bits_a, int bits_b, int bits_out, uint32_t * shift_out);
int64_t mul_lint(int64_t a, int64_t b, int bits_a, int bits_b, int bits_out, uint32_t * shift_out);
int norm_int(int32_t value);
int norm_lint(int64_t value);
int norm_uint(int32_t value);
int64_t div_int_with_shift(int32_t num, int32_t den, uint32_t * shift_out);
uint64_t div_uint_with_shift(uint32_t num, uint32_t den, uint32_t * shift_out);
int16_t div_int16_with_shift(int32_t num, int16_t den, uint32_t * shift_out);
int64_t div_uint16_with_shift(int32_t num, uint16_t den, uint32_t * shift_out);
int linear_quant(int add_first, int qmax, int qmin, float x, float scale, float offset);

// ===================================================================================================
// 3. Named global state
// ===================================================================================================
extern uint32_t g_gp_reg[32];             // RISC-V style general purpose registers (x0 stays 0)
extern uint64_t g_shape_reg[8];           // shape registers: four 16-bit dimensions each (3-bit selector)
extern uint8_t * g_DDR;                   // base of the simulated DDR image

// Offset of `ptr` inside the DDR image (ptr - g_DDR) as a 32-bit instruction address ("pc").
// Throws std::out_of_range when ptr lies below g_DDR or the offset does not fit in uint32_t.
inline uint32_t kpu_pc(const void * ptr)
{
    const uintptr_t base = reinterpret_cast<uintptr_t>(g_DDR);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if (addr < base || addr - base > UINT32_MAX)
        throw std::out_of_range("KPU_PC: pointer is outside of the 32-bit DDR address range");
    return static_cast<uint32_t>(addr - base);
}
#define KPU_PC(ptr) kpu_pc(ptr)
extern uint8_t * g_GLB[16];               // base of each of the 16 GLB banks
extern uint32_t g_glb_start[16];          // MmuConf: segment start per bank (in 32-byte lines)
extern uint32_t g_glb_depth[16];          // MmuConf: segment depth per bank (in 32-byte lines)
extern uint8_t g_GLB_DATA[0x400000];      // flat 4 MiB copy of all GLB segments used by dump_data_proc
extern uint32_t MMU_MMUItem[32];          // per bank: [2*bank] = segment start, [2*bank+1] = segment depth

// Storage of the unit singletons that the instruction classes cast to the unit class
// (reinterpret_cast<Unit *>(X_Y)).  Sized to the lifted class, see globals.cpp.
extern uint32_t AI2D_Ai2dInst[];          // AI2D       (sizeof(AI2D))
extern uint32_t Act0_act0[0x10010];       // Act0       (0x40000 bytes of PSUM + configuration words)
extern uint32_t Dm_dm[0x4000];            // Dm
extern uint32_t L2Load_L2LoadInst[0x4000];// L2Load
extern uint32_t L2Store_L2StoreInst[0x4000]; // L2Store

// On-chip buffers
extern uint32_t PSUM_L1[0x8000];          // L1 partial-sum buffer (0x20000 bytes)
extern uint8_t IF_L1[0x6000];             // L1 input-feature buffer
extern char pdp0_out_buffer[0x20000];          // PDP0 output buffer (pdp0.cpp, 0x20000 bytes)

// Debug / dump control
extern uint8_t debug_flag;                // set when the next weight / feature load should be dumped
extern std::string debug_file;            // path of that dump file
extern uint8_t debug_tcu_sel;             // 1 = dump TCU weights, otherwise Act0 data
extern int debug_dmw_h;                   // kernel row whose weights are dumped
extern uint8_t step_debug_flag;           // debug_function prompts for a command on the next call
extern uint8_t dump_data_flag;            // dump_data_proc is running
extern uint32_t debug_pc;                 // pc at which debug_function stops (-1: never)
extern std::vector<uint32_t> debug_dump_L3_start;  // DDR ranges to dump
extern std::vector<uint32_t> debug_dump_L3_len;
extern uint32_t debug_dump_L2;            // non-zero: dump all GLB banks
extern uint32_t debug_dump_reg;           // non-zero: dump registers
extern std::map<std::string, std::vector<uint32_t>> debug_map;   // parsed debug command
extern uint32_t ddr_burst_len;            // bytes per DDR burst
extern uint32_t ddr_burst_num;            // bursts per DDR group


// ===================================================================================================
// 4. Anonymous placeholders (addresses in the original image).  The former loose fields of the PDP1, L2Load and Act0
//    singletons are now real struct members (Pdp1Config, L2Load, Act0).
//    Declared in address order; the definitions in globals.cpp use the same order.
// ===================================================================================================
