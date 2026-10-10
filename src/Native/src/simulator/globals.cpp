// Free functions and global state of the K230 NPU C-model simulator.
//
// Lifted from the IDA/Hex-Rays dumps _global1..26.cpp and the plain-C fixed-point helpers
// (shr_rnd_int.c ... linear_quant.c).
#include "globals.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sstream>

#include "engines/ai2d.h"

// ===================================================================================================
// Global state
// ===================================================================================================


Globals _G;

void initialize_globals()
{
  std::memset(&_G, 0, sizeof _G);
  // verified against ELF .data @0x53a48c/0x53a490: initial ddr_burst_num = 256, ddr_burst_len = 16.
  _G.ddr_burst_len = 16;
  _G.ddr_burst_num = 256;
}

// Not defined any more: CheckPoint_checkpoint, Conv2D_conv2d, MFU_MFUInst, MeshNet_MeshNetInst, PDP0_pdp0 and
// PDP1_Pdp1Inst (the lifted classes keep their instances as function-local statics).


std::string debug_file;
std::vector<uint32_t> debug_dump_L3_start;
std::vector<uint32_t> debug_dump_L3_len;
std::map<std::string, std::vector<uint32_t>> debug_map;

// ---- Anonymous placeholders ----------------------------------------------------------------------
// Addresses in the original image; mostly fields of NPU singletons, see class headers.





// ===================================================================================================
// C++ helpers
// ===================================================================================================

namespace {

// x86 masks the count of a 32/64-bit shift; the original relies on that in a few corner cases.
inline uint32_t Shl32(uint32_t value, uint32_t count) { return value << (count & 31); }
inline int32_t Sar32(int32_t value, uint32_t count) { return value >> (count & 31); }
inline uint32_t Shr32(uint32_t value, uint32_t count) { return value >> (count & 31); }
inline int64_t Sar64(int64_t value, uint32_t count) { return value >> (count & 63); }
inline uint64_t Shr64(uint64_t value, uint32_t count) { return value >> (count & 63); }

// Hex field of the given width, zero filled (the original sets flags, width and fill inline before every number).
inline std::ostream & Hex(std::ostream & os, int width)
{
  return os << std::hex << std::setw(width) << std::setfill('0');
}

}  // namespace

// @0x41fc50 (_global1)
void set_g_gp_reg(uint8_t reg, uint32_t value)
{
  if (reg)
    _G.gp_reg[reg] = value;
  else
    _G.gp_reg[0] = 0;      // x0 is hard-wired to zero
}

// @0x41fcc0 (_global2)
void print_spu_data(uint32_t addr, SPU_TYPE_ID type, uint32_t tag, uint64_t data, std::ofstream & out)
{
  Hex(out, 8) << addr << ' ';
  Hex(out, 1) << static_cast<int>(type) << ' ';
  Hex(out, 2) << tag << ' ';
  Hex(out, 16) << data << std::endl;
}

// @0x420020 (_global3)
void print_spu_data(uint32_t addr, SPU_TYPE_ID type, uint32_t tag, uint32_t data, std::ofstream & out)
{
  Hex(out, 8) << addr << ' ';
  Hex(out, 1) << static_cast<int>(type) << ' ';
  Hex(out, 2) << tag << ' ';
  Hex(out, 8) << data << std::endl;
}

// @0x420390 (_global4)
void print_spu_data(uint32_t addr, SPU_TYPE_ID type, uint32_t tag, uint16_t data, std::ofstream & out)
{
  Hex(out, 8) << addr << ' ';
  Hex(out, 1) << static_cast<int>(type) << ' ';
  Hex(out, 2) << tag << ' ';
  Hex(out, 4) << static_cast<unsigned>(data) << std::endl;
}

// @0x42d3e0 (_global5)
void SimulatorInit(uint8_t * ddr, uint8_t ** glb_banks)
{
  _G.DDR = ddr;
  for (int bank = 0; bank < 16; ++bank)
    _G.GLB[bank] = glb_banks[bank];
}

// @0x42d4f0 (_global6)
void set_bit(std::bitset<32> & bits, int width, int lsb, int value)
{
  for (int i = 0; i < width; ++i) {
    const int pos = lsb + i;
    if (pos < 0 || pos >= 32)
      continue;                       // the original writes past the bitset's 32 bits, which is invisible
    bits[pos] = i < 32 && ((static_cast<uint32_t>(value) >> i) & 1u);
  }
}

// @0x42d560 (_global7)
// One line per entry of `byte_masks`; every line has ddr_burst_len bytes, highest byte first.
void print_r_data(std::vector<uint16_t> & byte_masks, std::vector<uint8_t> & data, std::ofstream & out)
{
  // Position of the first set bit of the first mask (burst length if none): number of leading unused bytes.
  uint32_t first_bit = 0;
  if (_G.ddr_burst_len && !byte_masks.empty()) {
    const uint32_t mask0 = byte_masks[0];
    if (!(mask0 & 1u)) {
      do
        ++first_bit;
      while (_G.ddr_burst_len > static_cast<uint8_t>(first_bit) && !((mask0 >> static_cast<uint8_t>(first_bit)) & 1u));
    }
  }

  for (size_t row = 0; row < byte_masks.size(); ++row) {
    const uint32_t burst = _G.ddr_burst_len;
    for (uint32_t col = 0; col < burst; ++col) {
      const bool enabled = (byte_masks[row] >> (burst - 1 - col)) & 1u;
      const uint64_t index = static_cast<uint64_t>(static_cast<uint32_t>(row * burst)) + burst - 1 - first_bit - col;
      // Row 0 prints zero for disabled bytes; later rows always print the data byte.
      // verified against asm @0x42d560: disabled byte prints 0 only in row 0 (rbx==0 test) (row 0 is the unaligned first beat).
      const unsigned value = (!enabled && row == 0) ? 0u : data[index];
      Hex(out, 2) << value;
    }
    out << std::endl;
  }
}

// @0x42d910 (_global8)
void print_addr(std::vector<uint32_t> & addrs, std::vector<uint8_t> & /*unused*/, std::ofstream & out)
{
  for (uint32_t addr : addrs)
    Hex(out, 8) << addr << std::endl;
}

// @0x42da80 (_global9)
std::vector<bool> ByteToBits(uint8_t value)
{
  std::vector<bool> bits;
  for (int i = 0; i < 8; ++i)
    bits.push_back((value >> i) & 1);        // bit 0 first
  return bits;
}

// @0x42de80 (_global10)
std::vector<uint8_t> BitsToByte(std::vector<bool> const & bits)
{
  // verified against asm @0x42de80: byte count = ceil(bit_count * 0.125) (.rodata 0x481398 = 0.125, vroundsd imm 0xa).
  const uint64_t byte_count = static_cast<uint64_t>(std::ceil(static_cast<double>(bits.size()) * 0.125));
  std::vector<uint8_t> bytes;
  for (uint64_t i = 0; i < byte_count; ++i) {
    uint8_t value = 0;
    for (unsigned bit = 0; bit < 8; ++bit) {
      const uint64_t index = 8 * i + bit;
      if (index < bits.size() && bits[index])
        value |= static_cast<uint8_t>(1u << bit);
    }
    bytes.push_back(value);
  }
  return bytes;
}

// @0x42e340 (_global11)
void calc_ddr_param(std::vector<uint32_t> & burst_addr, std::vector<uint8_t> & burst_count_m1,
                    std::vector<uint16_t> & byte_mask, uint8_t elem_bytes, uint32_t num_elems, uint32_t ddr_addr)
{
  const uint32_t total = num_elems * elem_bytes;
  if (total) {
    const uint32_t end_addr = total + ddr_addr;
    uint32_t done = 0;                               // bytes of the transfer handled so far
    while (done < total) {
      const uint32_t start = done + ddr_addr;
      const uint32_t burst = _G.ddr_burst_len;
      const uint32_t group = _G.ddr_burst_len * _G.ddr_burst_num;   // bytes per DDR group

      // Burst-aligned address of the first burst of this piece.
      burst_addr.push_back(burst * (start / burst));
      const uint32_t aligned = burst_addr.back();
      const uint32_t group_off = aligned % group;                 // position of that burst inside its group
      const uint32_t group_end = (start / group + 1) * group;     // end address of the group
      // verified against asm @0x42e340: ceil((double)end_addr / (double)burst) via vdivsd + vroundsd 0xa.
      const uint32_t end_rounded =
          burst * static_cast<uint32_t>(static_cast<int64_t>(std::ceil(static_cast<double>(end_addr) / static_cast<double>(burst))));
      const uint32_t unused_bursts = group_end <= end_rounded ? 0 : group_end - end_rounded;   // whole bursts after the end
      const uint32_t group_bytes = group - group_off - unused_bursts;                            // bytes read from this group
      burst_count_m1.push_back(static_cast<uint8_t>(group_bytes / burst - 1));

      // Byte offset of the start inside its first burst.
      uint32_t lead = start;
      if (aligned)
        lead = start % aligned;

      // Bytes of the last burst that lie behind the end of the transfer.
      uint32_t tail = 0;
      uint32_t tail_mask = 0;
      bool partial_tail = false;
      if (group_end > end_addr) {
        tail = group_end - end_addr - unused_bursts;
        tail_mask = Shl32(1, tail) - 1;
        partial_tail = (group_end - end_addr) != unused_bursts;
      }

      // Byte-enable mask per burst: the first burst skips `lead` bytes, the last one drops `tail` bytes.
      const uint8_t last_beat = burst_count_m1.back();
      for (uint32_t beat = 0; beat <= last_beat; ++beat) {
        uint32_t mask = 0xFFFFFFFFu;
        if (beat == 0 && lead != 0)
          mask = 0u - Shl32(1, lead);
        if (beat == last_beat && partial_tail)
          mask = ~Shl32(tail_mask, burst - tail) & mask;
        byte_mask.push_back(static_cast<uint16_t>(mask));
      }

      done += group_bytes - lead - tail;
    }
  }
  if (!burst_addr.empty())
    burst_addr[0] = ddr_addr;                        // the first burst keeps the unaligned start address
}

// @0x42ec80 (_global12)
// Dumps what the debug command in debug_map asked for.  File names are <prefix>L3_pc_<pc>_start_<s>_len_<n>.txt,
// <prefix>L2_pc_<pc>.txt, <prefix>GPR_pc_<pc>.txt, <prefix>MMU_pc_<pc>.txt and <prefix>SSR_pc_<pc>.txt.
// verified against asm @0x42ec80: the std::to_string() arguments are entry [i] of the L3 vectors and entry [0] of
// the one-element vectors ("pc", "dump_L2", "dump_reg").
void dump_data_proc(std::string prefix)
{
  auto value_of = [](const char * key, size_t index) -> uint32_t {
    const std::vector<uint32_t> & v = debug_map[key];
    return index < v.size() ? v[index] : 0u;
  };
  const std::string pc = std::to_string(value_of("pc", 0));

  // --- L3 (DDR) ranges ---
  for (uint32_t i = 0; i < debug_map["dump_L3_start"].size(); ++i) {
    const uint32_t start = value_of("dump_L3_start", i);
    const uint32_t len = value_of("dump_L3_len", i);
    std::cout << "dump_L3_start: " << start << std::endl;
    std::cout << "dump_L3_len: " << len << std::endl;
    if (len) {
      const std::string name = prefix + "L3_pc_" + pc + "_start_" + std::to_string(start) + "_len_" +
                               std::to_string(len) + ".txt";
      std::ofstream out(name.c_str(), std::ios::out);
      for (uint32_t addr = start; addr < start + len; ++addr)
        Hex(out, 2) << static_cast<unsigned>(_G.DDR[addr]) << '\n';
    }
  }

  // --- L2 (all GLB banks, flattened into _G.GLB_DATA) ---
  std::cout << "dump_L2:" << value_of("dump_L2", 0) << std::endl;
  if (value_of("dump_L2", 0)) {
    const std::string name = prefix + "L2_pc_" + pc + ".txt";
    std::ofstream out(name.c_str(), std::ios::out);
    std::memset(_G.GLB_DATA, 0, sizeof _G.GLB_DATA);
    for (int bank = 0; bank < 16; ++bank) {
      // start / depth are counted in 32-byte lines.
      const uint64_t byte_start = 32ull * _G.glb_start[bank];
      uint64_t byte_len = 32ull * _G.glb_depth[bank];
      if (byte_start >= sizeof _G.GLB_DATA)
        continue;                                    // guard added: the original has no bounds check
      if (byte_start + byte_len > sizeof _G.GLB_DATA)
        byte_len = sizeof _G.GLB_DATA - byte_start;
      std::memcpy(_G.GLB_DATA + byte_start, _G.GLB[bank], byte_len);
    }
    for (size_t i = 0; i < sizeof _G.GLB_DATA; ++i)
      Hex(out, 2) << static_cast<unsigned>(_G.GLB_DATA[i]) << '\n';
  }

  // --- registers ---
  std::cout << "dump_reg: " << value_of("dump_reg", 0) << std::endl;
  if (value_of("dump_reg", 0)) {
    {
      const std::string name = prefix + "GPR_pc_" + pc + ".txt";
      std::ofstream out(name.c_str(), std::ios::out);
      for (uint32_t reg : _G.gp_reg)
        Hex(out, 8) << reg << '\n';
    }
    {
      // One line per GLB bank: index, segment start, segment depth (decimal, default fill).
      const std::string name = prefix + "MMU_pc_" + pc + ".txt";
      std::ofstream out(name.c_str(), std::ios::out);
      for (unsigned bank = 0; bank != 16; ++bank) {
        out << std::dec << std::setw(2) << bank << ' ' << std::setw(8) << _G.glb_start[bank] << ' ' << std::setw(8)
            << _G.glb_depth[bank] << '\n';
      }
    }
    {
      // verified against asm/symbols: _G.shape_reg is 64 bytes (8 registers) and is directly followed by _G.GLB.
      const std::string name = prefix + "SSR_pc_" + pc + ".txt";
      std::ofstream out(name.c_str(), std::ios::out);
      for (uint64_t shape : _G.shape_reg)
        Hex(out, 16) << shape << '\n';
    }
  }
}

// @0x4310a0 (_global13)
// Command syntax (tokens separated by `delim`):
//   +debug_pc_<n> | +debug_pc_end   pc to stop at (end -> -1)
//   +dump_L3_start_<n> +dump_L3_len_<n>   DDR range (may repeat)
//   +dump_L2_data   +dump_reg
void Stringsplit(std::string const & text, std::string const & delim,
                 std::map<std::string, std::vector<uint32_t>> & out)
{
  _G.debug_pc = 0;
  debug_dump_L3_start.clear();
  debug_dump_L3_len.clear();
  _G.debug_dump_L2 = 0;
  _G.debug_dump_reg = 0;
  if (text.empty())
    return;

  std::vector<std::string> tokens;
  if (delim.empty()) {
    tokens.push_back(text);        // UNCERTAIN: asm @0x4311b1 handles an empty delimiter with a separate loop (step 0, no
                                   // find); only reached when delim is empty, which no caller does
  } else {
    std::string rest = text + delim;
    size_t pos;
    while ((pos = rest.find(delim)) != std::string::npos) {
      tokens.push_back(rest.substr(0, pos));
      rest = rest.substr(pos + delim.size());
    }
  }

  for (const std::string & token : tokens) {
    if (token.find("+debug_pc_") != std::string::npos) {
      if (token.find("end") == std::string::npos)
        _G.debug_pc = static_cast<uint32_t>(std::strtol(token.substr(10).c_str(), nullptr, 10));
      else
        _G.debug_pc = static_cast<uint32_t>(-1);
    } else if (token.find("+dump_L3_start_") != std::string::npos) {
      debug_dump_L3_start.push_back(static_cast<uint32_t>(std::strtol(token.substr(15).c_str(), nullptr, 10)));
    } else if (token.find("+dump_L3_len_") != std::string::npos) {
      debug_dump_L3_len.push_back(static_cast<uint32_t>(std::strtol(token.substr(13).c_str(), nullptr, 10)));
    } else if (token.find("+dump_L2_data") != std::string::npos) {
      _G.debug_dump_L2 = 1;
    } else if (token.find("+dump_reg") != std::string::npos) {
      _G.debug_dump_reg = 1;
    }
  }

  if (debug_dump_L3_start.empty()) {     // default range: nothing
    debug_dump_L3_start.push_back(0);
    debug_dump_L3_len.push_back(0);
  }

  out["pc"] = std::vector<uint32_t>{_G.debug_pc};
  out["dump_L3_start"] = debug_dump_L3_start;
  out["dump_L3_len"] = debug_dump_L3_len;
  out["dump_L2"] = std::vector<uint32_t>{_G.debug_dump_L2};
  out["dump_reg"] = std::vector<uint32_t>{_G.debug_dump_reg};
}

// @0x4329b0 (_global14)
void debug_function(uint32_t pc, std::string prefix)
{
  std::string line;
  _G.dump_data_flag = 0;
  if (_G.step_debug_flag) {
    std::cout << "debug_cmd: +debug_pc_x +dump_L3_start_x +dump_L3_len_x +dump_L2_data +dump_reg [input other for end]"
              << std::endl;
    std::getline(std::cin, line);
    Stringsplit(line, " ", debug_map);
    const std::vector<uint32_t> & stop_pc = debug_map["pc"];
    std::cout << "set pc: " << (stop_pc.empty() ? 0u : stop_pc[0]) << std::endl;
    _G.step_debug_flag = 0;
  }

  const std::vector<uint32_t> & stop_pc = debug_map["pc"];
  if ((stop_pc.empty() ? 0u : stop_pc[0]) == pc) {
    _G.step_debug_flag = 1;                 // prompt again on the next call
    _G.dump_data_flag = 1;
  } else if (!_G.dump_data_flag) {
    return;
  }
  dump_data_proc(prefix);
  _G.dump_data_flag = 0;
  std::cout << "current pc: " << pc << std::endl;
}

// @0x4385a0 (_global15)
void memory_dump(void const * data, int size)
{
  const uint8_t * bytes = static_cast<const uint8_t *>(data);
  for (int i = 0; i < size; ++i) {
    if (i > 0 && i % 16 == 0)
      std::putchar('\n');
    std::printf("%02x ", bytes[i]);
    if (i + 1 == size)
      break;
    if ((i + 1) % 8 == 0)
      std::putchar(' ');
  }
  std::putchar('\n');
}

// @0x439930 (_global16)
// Converts a bfloat16 to a signed integer (the mantissa with its implicit one, shifted by the exponent, rounded);
// same algorithm as TileHelper::BF16ToQint.
int64_t BF16ToQint(BF16::bfloat16 value)
{
  const uint32_t bits = value.bits_;
  const uint32_t sign = bits >> 15;
  const uint8_t mantissa = static_cast<uint8_t>(bits | 0x80);   // 7 mantissa bits + implicit one
  const uint8_t exponent = static_cast<uint8_t>(bits >> 7);      // biased exponent
  const int16_t shift = static_cast<int16_t>(134 - exponent);
  const int32_t sign_mul = 1 - 2 * static_cast<int32_t>(sign);
  if (shift <= 0)   // left shift by exponent - 134 (x86 masks the count)
    return static_cast<int32_t>(Shl32(mantissa, (bits >> 7) + 122) * static_cast<uint32_t>(sign_mul));
  if (shift <= 8) {
    // round to nearest (ties follow the bit that will be shifted out), then shift right
    const uint8_t rounding = static_cast<uint8_t>((((int)mantissa >> shift) & 1) + (127 >> (8 - shift)));
    return sign_mul * ((int)static_cast<uint16_t>(mantissa + rounding) >> shift);
  }
  return 0;
}

// @0x4399c0 (_global17)
// Linear element address of the 4-D index (d0, d1, d2, d3); layout 1 = (d0, d1, d2) major, d3 minor.
int64_t GetAddress(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,
                   uint32_t stride0, uint32_t stride1, uint32_t stride2, uint8_t layout)
{
  const uint32_t base = stride0 * d0;
  if (layout == 1)
    return stride2 * (stride1 * (d1 + base) + d2) + d3;
  return base + d3 + stride2 * d2 + d1 * stride1;
}

// ---------------------------------------------------------------------------------------------------
// GLB access traces
// ---------------------------------------------------------------------------------------------------

namespace {

constexpr size_t kTraceLineBytes = 32;

uint32_t CountValid(std::vector<bool> const & valid)
{
  uint32_t count = 0;
  for (bool bit : valid)
    count += bit;
  return count;
}

// The first entry becomes bit 31 of the mask.
uint32_t PackValid(std::vector<bool> const & valid)
{
  uint32_t mask = 0;
  for (size_t i = 0; i < valid.size() && i < 32; ++i)
    if (valid[i])
      mask |= 1u << (31 - i);
  return mask;
}

// When this call completes the transfer (all bytes seen so far add up to total_bytes) and the line is not yet
// full, the line is padded at the front with zero bytes that are marked invalid.
void PadCompletedLine(std::vector<uint8_t> & line, std::vector<bool> & valid, uint32_t valid_in_line,
                      uint32_t valid_carry, uint32_t total_bytes)
{
  if (valid_in_line + valid_carry == total_bytes && !line.empty() && line.size() < kTraceLineBytes) {
    do {
      line.insert(line.begin(), 0);
      valid.insert(valid.begin(), false);
    } while (!line.empty() && line.size() < kTraceLineBytes);
  }
}

// New value of the carried valid-byte count after a full line was written: the count is only carried for
// transfers shorter than a line that are still incomplete.
uint32_t NextValidCarry(uint32_t valid_carry, uint32_t valid_in_line, uint32_t total_bytes, bool last)
{
  if (!(valid_carry == 0 && total_bytes < kTraceLineBytes && !last) || total_bytes == valid_in_line)
    return 0;
  return valid_in_line;
}

}  // namespace

// @0x441de0 (_global18)
// Header file: "<line address>(5 digits) <valid mask>(8 digits)"; data file: the 32 bytes of the line.
void print_glb_read_core(std::vector<uint8_t> & line, std::vector<bool> & byte_valid, uint32_t glb_addr,
                         std::ofstream & addr_log, std::ofstream & data_log, uint32_t & valid_so_far,
                         uint32_t total_bytes, bool last)
{
  uint32_t valid_in_line = CountValid(byte_valid);
  PadCompletedLine(line, byte_valid, valid_in_line, valid_so_far, total_bytes);
  if (line.size() == kTraceLineBytes) {
    valid_in_line = NextValidCarry(valid_so_far, valid_in_line, total_bytes, last);
    valid_so_far = valid_in_line;
    const uint32_t mask = PackValid(byte_valid);
    Hex(addr_log, 5) << (glb_addr >> 5) << ' ';
    Hex(addr_log, 8) << mask << std::endl;
    for (uint8_t byte : line)
      Hex(data_log, 2) << static_cast<unsigned>(byte);
    data_log << std::endl;
    line.clear();
    byte_valid.clear();
  }
}

// @0x4421c0 (_global19)
// One line per 32 bytes: "<line address>(5) <valid mask>(8) <32 data bytes>".
void print_glb_write_core(std::vector<uint8_t> & line, std::vector<bool> & byte_valid, uint32_t glb_addr,
                          std::ofstream & log, uint32_t & valid_so_far, uint32_t total_bytes, bool last)
{
  uint32_t valid_in_line = CountValid(byte_valid);
  PadCompletedLine(line, byte_valid, valid_in_line, valid_so_far, total_bytes);
  if (line.size() == kTraceLineBytes) {
    valid_in_line = NextValidCarry(valid_so_far, valid_in_line, total_bytes, last);
    valid_so_far = valid_in_line;
    const uint32_t mask = PackValid(byte_valid);
    Hex(log, 5) << (glb_addr >> 5) << ' ';
    Hex(log, 8) << mask << ' ';
    for (uint8_t byte : line)
      Hex(log, 2) << static_cast<unsigned>(byte);
    log << std::endl;
    line.clear();
    byte_valid.clear();
  }
}

namespace {

// Trace line under construction.  Bytes are inserted at the front, so index 0 is the highest address.
struct TraceLine {
  std::vector<uint8_t> bytes_;
  std::vector<bool> valid_;
  uint32_t valid_carry_ = 0;

  void PushBack(uint8_t byte, bool is_valid) { bytes_.push_back(byte); valid_.push_back(is_valid); }
  void PushFront(uint8_t byte, bool is_valid)
  {
    bytes_.insert(bytes_.begin(), byte);
    valid_.insert(valid_.begin(), is_valid);
  }
  bool Empty() const { return bytes_.empty(); }
  size_t Size() const { return bytes_.size(); }
};

// Byte address of the GLB segment of `glb_addr` after MMU translation: offset in the bank plus 32 * segment start.
uint32_t GlbBase(uint32_t glb_addr)
{
  return (glb_addr & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * (glb_addr >> 28)];
}

// Shared tile walk of mfu_glb_read and mfu_glb_write (flat mode): visits every element of the 4-D tile
// (cnt3 x cnt2 x cnt1 x cnt0, strides from the first three shape dimensions), inserts its bytes into `line`
// and calls emit(byte address, last) after every byte and once more at the end of every element.
template <typename Emit>
void WalkTile(uint64_t shape, uint16_t cnt3, uint16_t cnt2, uint16_t cnt1, uint16_t cnt0, uint32_t glb_addr,
              uint8_t elem_log2, MemAccessor const & mem, TraceLine & line, Emit emit)
{
  const uint32_t elem = static_cast<uint8_t>(Shl32(1, elem_log2));   // element size in bytes
  const uint32_t dim0 = shape & 0xFFFF;
  const uint32_t dim1 = (shape >> 16) & 0xFFFF;
  const uint32_t dim2 = (shape >> 32) & 0xFFFF;
  const uint32_t stride_k = elem * dim0;
  const uint32_t stride_j = elem * dim0 * dim1;
  const uint32_t stride_i = elem * dim0 * dim1 * dim2;
  const uint32_t base = GlbBase(glb_addr);

  bool last_state = false;          // "last byte of the tile" flag carried between elements
  uint32_t addr = 0;                // GLB byte address of the next byte
  for (uint32_t i = 0; i < cnt3; ++i)
    for (uint32_t j = 0; j < cnt2; ++j)
      for (uint32_t k = 0; k < cnt1; ++k)
        for (uint32_t l = 0; l < cnt0; ++l) {
          const uint32_t offset = i * stride_i + j * stride_j + k * stride_k + l * elem;   // byte offset in `mem`
          if (line.Empty()) {
            // A new line starts: continue from the translated address and skip the bytes before it in the line.
            addr = offset + base;
            for (const size_t lead = addr & 0x1F; line.Size() < lead;)
              line.PushBack(0, false);
          }
          const bool last_l = (l == static_cast<uint32_t>(cnt0) - 1);
          uint32_t next;                // address behind the element
          if (elem) {
            for (uint32_t b = 0; b < elem; ++b) {
              line.PushFront(mem.MemAt<uint8_t>(static_cast<int>(offset + b)), true);
              last_state = last_l && b == elem - 1;
              emit(addr, last_state);
              if (b == elem - 1)
                break;
              ++addr;
            }
            next = addr + 1;
          } else {
            next = addr;
          }

          uint32_t emit_addr;
          bool emit_last;
          if (last_l && line.Size() < kTraceLineBytes) {
            // End of the innermost row: fill the line up with invalid zero bytes.
            if (line.Empty()) {
              emit_addr = next - 1;
              addr = next;
            } else {
              uint32_t pad = next;
              for (;;) {
                line.PushFront(0, false);
                addr = pad + 1;
                if (line.Size() >= kTraceLineBytes)
                  break;
                ++pad;
              }
              emit_addr = pad;
            }
            last_state = true;
            emit_last = true;
          } else {
            emit_addr = next - 1;
            addr = next;
            emit_last = last_state;
          }
          emit(emit_addr, emit_last);
        }
}

}  // namespace

// @0x443460 (_global20)
// `shape`: three 16-bit dimensions (dim0 = bits 15:0, dim1 = 31:16, dim2 = 47:32) giving the strides of the
// tile in elements; cnt3..cnt0 are the loop counts (outermost first); the element size is 1 << elem_log2 bytes.
// total_bytes is the number of valid bytes of the whole access, used to detect the last line.
void mfu_glb_read(uint64_t shape, uint16_t cnt3, uint16_t cnt2, uint16_t cnt1, uint16_t cnt0, uint32_t glb_addr,
                  uint8_t elem_log2, MemAccessor mem, std::ofstream & addr_log, std::ofstream & data_log,
                  uint32_t total_bytes)
{
  TraceLine line;
  if (cnt3 == 0)
    return;
  WalkTile(shape, cnt3, cnt2, cnt1, cnt0, glb_addr, elem_log2, mem, line, [&](uint32_t addr, bool last) {
    print_glb_read_core(line.bytes_, line.valid_, addr, addr_log, data_log, line.valid_carry_, total_bytes, last);
  });
}

// @0x443b00 (_global21)
// Same tile walk as mfu_glb_read.  row_mode != 0 (stack argument, tested at @0x443b45) selects a second traversal that only uses cnt3 and
// cnt2 (one element at index i*dim0*dim1*dim2 + j*dim0*dim1 per step); the semantic name of the flag is not recoverable from asm.
void mfu_glb_write(uint64_t shape, uint16_t cnt3, uint16_t cnt2, uint16_t cnt1, uint16_t cnt0, uint32_t glb_addr,
                   uint8_t elem_log2, MemAccessor mem, std::ofstream & log, uint32_t total_bytes, uint8_t row_mode)
{
  TraceLine line;
  auto emit = [&](uint32_t addr, bool last) {
    print_glb_write_core(line.bytes_, line.valid_, addr, log, line.valid_carry_, total_bytes, last);
  };

  if (!row_mode) {
    if (cnt3)
      WalkTile(shape, cnt3, cnt2, cnt1, cnt0, glb_addr, elem_log2, mem, line, emit);
    return;
  }

  if (cnt3 == 0)
    return;
  const uint32_t elem = static_cast<uint8_t>(Shl32(1, elem_log2));
  const uint32_t dim0 = shape & 0xFFFF;
  const uint32_t dim1 = (shape >> 16) & 0xFFFF;
  const uint32_t dim2 = (shape >> 32) & 0xFFFF;
  const uint32_t base = GlbBase(glb_addr);
  bool last_state = false;
  for (uint32_t i = 0; i < cnt3; ++i)
    for (uint32_t j = 0; j < cnt2; ++j) {
      const uint32_t row_offset = elem * (i * dim0 * dim1 * dim2 + j * dim0 * dim1);   // byte offset in `mem`
      uint32_t addr = base + row_offset;
      if (line.Empty() && (addr & 0x1F) != 0)
        for (const size_t lead = addr & 0x1F; line.Size() < lead;)
          line.PushBack(0, false);

      const bool last_j = (j == static_cast<uint32_t>(cnt2) - 1);
      uint32_t next;
      if (elem) {
        next = addr + elem;
        for (uint32_t b = 0; b < elem; ++b, ++addr) {
          line.PushFront(mem.MemAt<uint8_t>(static_cast<int>(row_offset + b)), true);
          if (last_j) {
            last_state = (b == elem - 1);
            emit(addr, last_state);
          } else {
            emit(addr, false);
          }
        }
        if (!last_j)
          last_state = false;
      } else {
        next = base;
      }

      bool final_last = last_state;
      if (last_j && line.Size() < kTraceLineBytes) {
        while (line.Size() < kTraceLineBytes && !line.Empty()) {
          ++next;
          line.PushFront(0, false);
        }
        last_state = true;
        final_last = true;
      }
      emit(next - 1, final_last);
    }
}

// @0x464c30 (_global22)
void Stringsplit(std::string text, char delim, std::vector<std::string> & out)
{
  std::istringstream stream(text);
  std::string item;
  while (std::getline(stream, item, delim))
    out.push_back(item);
}

// @0x4672d0 (_global23)
int fp_to_int32(float value)
{
  // vroundss with imm 0x0C (current rounding mode, nearest-even by default), then vcvttss2si.
  const float rounded = std::nearbyintf(value);
  if (!(rounded >= -2147483648.0f && rounded < 2147483648.0f))
    return INT_MIN;                  // x86 "integer indefinite"
  return static_cast<int>(rounded);
}

// @0x4672e0 (_global24)
int64_t xy_climp(int64_t value, int64_t lo, int64_t hi)
{
  if (value < lo)
    return lo;
  if (value > hi)
    return hi;
  return value;
}

// @0x467300 (_global25)
// Mirrors an out-of-range coordinate back into the range.
// verified against asm @0x467300: below the range the value is negated (reflection about 0), not about `lo`.
int64_t mirror_climp(int64_t value, int64_t lo, int64_t hi)
{
  if (value < lo)
    return -value;
  if (value <= hi)
    return value;
  return hi - (value - hi);          // reflection about hi
}

// @0x467330 (_global26)
// In-place inverse of the 2x3 affine matrix {a, b, c, d, e, f}:  x' = a*x + b*y + c,  y' = d*x + e*y + f.
// A singular matrix (determinant 0) becomes the zero matrix.
// verified against asm @0x467330: .rodata 0x5220D8 = 1.0, .rodata 0x522D10 = -0.0 (so a singular matrix gives +0.0 / -0.0
// as inv / neg_inv), sign mask .rodata 0x522D20 = 0x8000000000000000.
void Mat_Inv(double * m)
{
  const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5];
  const double det = a * e - b * d;
  double inv = 0.0;
  double neg_inv = -0.0;
  if (det != 0.0 || std::isnan(det)) {
    inv = 1.0 / det;
    neg_inv = -inv;
  }
  m[0] = e * inv;
  m[1] = b * neg_inv;
  m[3] = d * neg_inv;
  m[4] = a * inv;
  m[2] = -(e * inv) * c - (b * neg_inv) * f;
  m[5] = -(d * neg_inv) * c - f * (a * inv);
}

// ===================================================================================================
// Fixed-point helpers (plain C functions in the original)
// ===================================================================================================
// Rounding modes of the shr_rnd_* functions: 1 = round half up, 2 = round half to even,
// anything else = truncate (arithmetic shift, rounds towards minus infinity).

// @0x46b0f0 (shl_2_xbit)  verified against asm: saturating shift of a signed value to `xbits` bits.  Each of the
// `shift` doublings first checks value > hi / value < lo (returning the limit); shift <= 0 tail-calls
// shr_rnd_lint(value, -shift, 0).  The last doubling is not re-checked.
int64_t shl_2_xbit(int64_t value, int shift, int xbits)
{
  int64_t hi = INT64_MAX, lo = INT64_MIN;
  if (xbits != 64) {
    const unsigned c = static_cast<unsigned>(xbits - 1) & 63;
    hi = static_cast<int64_t>((uint64_t{1} << c) - 1);
    lo = static_cast<int64_t>(~uint64_t{0} << c);
  }
  if (shift <= 0)
    return shr_rnd_lint(value, -shift, 0);
  for (; shift != 0; --shift) {
    if (hi < value)
      return hi;
    if (lo > value)
      return lo;
    value = static_cast<int64_t>(static_cast<uint64_t>(value) << 1);
  }
  return value;
}

// @0x46b170 (shr_rnd_int.c)
// Arithmetic right shift of a 32-bit value with rounding.  Negative shift counts shift left with saturation to 32
// bits; shifts of 33 or more give 0.  (The dump shows a 64-bit rax with the 32-bit result zero-extended.)
int32_t shr_rnd_int(int32_t value, int shift, int round_mode)
{
  if (shift < 0)
    return static_cast<int32_t>(shl_2_xbit(value, -shift, 32));
  if (shift == 0)
    return value;
  const int64_t half = int64_t{1} << ((shift - 1) & 63);
  int64_t biased = value;
  if (round_mode == 1) {
    biased = value + half;                                     // round half up
  } else if (round_mode == 2) {
    // round half to even: at an exact tie the value is only rounded up when the result would be odd
    const int32_t low_mask = static_cast<int32_t>(Shl32(1, shift)) - 1;
    const bool tie = (value & low_mask) == static_cast<int32_t>(Shl32(1, shift - 1));
    if (!tie || (Sar32(value, shift) & 1))
      biased = value + half;
  }
  if (shift >= 33)
    return 0;
  return static_cast<int32_t>(static_cast<uint32_t>(biased >> shift));   // low 32 bits (the dump zero-extends)
}

// @0x46b200 (shr_rnd_lint.c)
// Same for 64-bit values; shifts above 63 abort the program.
int64_t shr_rnd_lint(int64_t value, int shift, int round_mode)
{
  if (shift > 63) {
    std::printf("the func shr_rnd_lint parameter shift should not greater than 63!!!\n ");
    std::exit(0);
  }
  if (shift < 0)
    return shl_2_xbit(value, -shift, 64);
  if (shift == 0)
    return value;
  const int64_t half = int64_t{1} << (shift - 1);
  if (round_mode == 1)
    return (value + half) >> shift;
  if (round_mode != 2)
    return value >> shift;
  const int64_t low = value & ((int64_t{1} << shift) - 1);
  if (low == half && !((value >> shift) & 1))
    return value >> shift;                    // exact tie, even result: round down
  return (value + half) >> shift;
}

// @0x46b2a0 (sat_int_xbits.c)
// Clamps a 32-bit value to the signed range of `xbits` bits (xbits <= 32, otherwise the program aborts).
int32_t sat_int_xbits(int32_t value, int xbits)
{
  if (xbits > 32) {
    std::printf("func sat_int_xbits x should not great than 32!!!\n ");
    std::exit(0);
  }
  const uint32_t count = static_cast<uint8_t>(xbits - 1);
  const int64_t hi = int64_t{1} << (count & 63);
  if (hi <= value)
    return static_cast<int32_t>(hi - 1);
  const int64_t lo = static_cast<int64_t>(~uint64_t{0} << (count & 63));
  if (value >= lo)
    return value;
  return static_cast<int32_t>(lo);
}

// @0x46b300 (sat_lint_xbits.c)
// Clamps a 64-bit value to the signed range of `xbits` bits (xbits <= 64, otherwise the program aborts).
// verified against asm @0x46b300: for xbits == 64 the upper limit 1 << 63 wraps to INT64_MIN, so every input
// saturates to INT64_MAX (quirk of the original).
int64_t sat_lint_xbits(int64_t value, int xbits)
{
  if (xbits > 64) {
    std::puts("func sat_int_xbits x should not great than 64!!!");
    std::exit(0);
  }
  const uint32_t count = static_cast<uint8_t>(xbits - 1) & 63;
  const int64_t hi = static_cast<int64_t>(uint64_t{1} << count);
  if (hi <= value)
    return static_cast<int64_t>(static_cast<uint64_t>(hi) - 1);
  const int64_t lo = static_cast<int64_t>(~uint64_t{0} << count);
  if (lo < value)
    return value;
  return lo;
}

// @0x46b350 (mul_int.c)
// Product of two 32-bit fixed-point numbers, rounded half-to-even.  *shift_out receives the right shift that was
// applied.  verified against asm @0x46b350: shift = bits_a + bits_b - 1 - bits_out.
int64_t mul_int(int32_t a, int32_t b, int bits_a, int bits_b, int bits_out, uint32_t * shift_out)
{
  const uint32_t shift = static_cast<uint32_t>(bits_a + bits_b - 1 - bits_out);
  *shift_out = shift;
  return shr_rnd_lint(static_cast<int64_t>(b) * a, static_cast<int>(shift), 2);
}

// @0x46b370 (mul_lint.c)
int64_t mul_lint(int64_t a, int64_t b, int bits_a, int bits_b, int bits_out, uint32_t * shift_out)
{
  const uint32_t shift = static_cast<uint32_t>(bits_a + bits_b - 1 - bits_out);
  *shift_out = shift;
  return shr_rnd_lint(b * a, static_cast<int>(shift), 2);
}

// @0x46b390 (norm_int.c)
// Number of left shifts that bring a signed 32-bit value into [2^30, 2^31) (redundant sign bits);
// 0 for 0, 31 for -1.
int norm_int(int32_t value)
{
  if (value == 0)
    return 0;
  if (value == -1)
    return 31;
  int32_t magnitude = ~value;
  if (value >= magnitude)
    magnitude = value;               // max(value, ~value)
  int count = 0;
  while (magnitude <= 0x3FFFFFFF) {
    magnitude *= 2;
    ++count;
  }
  return count;
}

// @0x46b3d0 (norm_lint.c)
// 64-bit version of norm_int; 63 for 0 and -1.
int norm_lint(int64_t value)
{
  if (static_cast<uint64_t>(value + 1) <= 1)
    return 63;
  int64_t magnitude = (value >> 63) ^ value;
  int count = 0;
  while (magnitude <= 0x3FFFFFFFFFFFFFFFLL) {
    magnitude *= 2;
    ++count;
  }
  return count;
}

// @0x46b420 (norm_uint.c)
// Number of leading zero bits of the 32-bit pattern (32 for 0).
int norm_uint(int32_t value)
{
  uint32_t bits = static_cast<uint32_t>(value);
  if (bits == 0)
    return 32;
  int count = 0;
  while (!(bits & 0x80000000u)) {
    bits <<= 1;
    ++count;
  }
  return count;
}

// @0x46b450 (div_int_with_shift.c)
// Normalised quotient num / den: the numerator is shifted left by its norm, the quotient is brought back to a
// 32-bit mantissa and *shift_out reports the total left shift.  Division by zero saturates (0 for num == 0).
int64_t div_int_with_shift(int32_t num, int32_t den, uint32_t * shift_out)
{
  if (den == 0) {
    int64_t result = 0;
    if (num)
      result = static_cast<uint32_t>(num <= 0) + 0x7FFFFFFFu;     // 0x7FFFFFFF, or 0x80000000 for negative num
    *shift_out = 0;
    return result;
  }
  const int num_norm = norm_int(num);
  const int64_t quotient = (static_cast<int64_t>(num) << ((static_cast<uint8_t>(num_norm) + 32) & 63)) / den;
  const int quotient_norm = norm_lint(quotient);
  *shift_out = static_cast<uint32_t>(quotient_norm + num_norm);
  return Sar64(quotient, 32 - static_cast<uint8_t>(quotient_norm));
}

// @0x46b4e0 (div_uint_with_shift.c)
// Unsigned version; division by zero gives 0xFFFFFFFF (0 for num == 0).
uint64_t div_uint_with_shift(uint32_t num, uint32_t den, uint32_t * shift_out)
{
  if (den == 0) {
    *shift_out = 0;
    return static_cast<uint32_t>(-(num != 0));
  }
  const int num_norm = norm_uint(static_cast<int32_t>(num));
  const uint64_t quotient = (static_cast<uint64_t>(num) << ((static_cast<uint8_t>(num_norm) + 32) & 63)) / den;
  const int quotient_norm = norm_uint(static_cast<int32_t>(quotient >> 32));
  *shift_out = static_cast<uint32_t>(quotient_norm + num_norm);
  return Shr64(quotient, 32 - static_cast<uint8_t>(quotient_norm));
}

// @0x46b560 (div_int16_with_shift.c)
// 16-bit version (only the low 16 bits of num are used); division by zero gives 0x7FFF / 0x8000 (0 for num == 0).
int16_t div_int16_with_shift(int32_t num, int16_t den, uint32_t * shift_out)
{
  if (den == 0) {
    int16_t result = 0;
    if (static_cast<int16_t>(num) != 0)
      result = static_cast<int16_t>((static_cast<int16_t>(num) <= 0) + 0x7FFF);
    *shift_out = 0;
    return result;
  }
  const int num_norm = norm_int(static_cast<int32_t>(static_cast<uint32_t>(num) << 16));
  const int64_t quotient =
      static_cast<int64_t>(static_cast<int32_t>(Shl32(static_cast<uint32_t>(static_cast<int16_t>(num)), num_norm + 16))) / den;
  const int quotient_norm = norm_int(static_cast<int32_t>(quotient));
  *shift_out = static_cast<uint32_t>(quotient_norm + num_norm);
  return static_cast<int16_t>(Sar64(quotient, 16 - static_cast<uint8_t>(quotient_norm)));
}

// @0x46b5f0 (div_uint16_with_shift.c)
// Unsigned 16-bit version; division by zero gives 0xFFFFFFFF (0 for num == 0).
int64_t div_uint16_with_shift(int32_t num, uint16_t den, uint32_t * shift_out)
{
  if (den == 0) {
    *shift_out = 0;
    return static_cast<uint32_t>(-(static_cast<uint16_t>(num) != 0));
  }
  const int num_norm = norm_uint(static_cast<int32_t>(static_cast<uint32_t>(num) << 16));
  const uint32_t quotient = Shl32(static_cast<uint16_t>(num), num_norm + 16) / den;
  const int quotient_norm = norm_uint(static_cast<int32_t>(quotient));
  *shift_out = static_cast<uint32_t>(quotient_norm + num_norm);
  return Shr32(quotient, 16 - quotient_norm);
}

// @0x46b670 (linear_quant.c)
// Affine quantisation of `x`: (x + offset) * scale when add_first is non-zero, otherwise x * scale + offset;
// rounded half away from zero and clamped to [qmin, qmax].
int linear_quant(int add_first, int qmax, int qmin, float x, float scale, float offset)
{
  const float scaled = add_first ? (x + offset) * scale : x * scale + offset;
  const float rounded = std::roundf(scaled);
  int result = (rounded >= -2147483648.0f && rounded < 2147483648.0f) ? static_cast<int>(rounded) : INT_MIN;
  if (result > qmax)
    result = qmax;
  if (result < qmin)
    return qmin;
  return result;
}
