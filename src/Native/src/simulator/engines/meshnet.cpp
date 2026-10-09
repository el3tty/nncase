// MeshNet: lifted from IDA/Hex-Rays output (MeshNet1..MeshNet16 in sources/).  The AVX instructions the
// auto-converted results/ copy had stripped (vcomiss / vroundss / vcvttss2si / ...) were recovered from sources/.
//
// Conventions used by the lifted code
//  * Tensors are addressed as   element(i, j, k, l) = l + w * (k + h * (j + c * i))   where (c, h, w) are the three
//    halfwords of a 64-bit "shape" register (MFU::GetCHW).  "Dims" registers hold four halfwords (w, h, c, n).
//  * The inlined fp16 <-> binary32 conversions (norm_uint chains) are FP16::fp16::f_value / round_to_fp16, the
//    inlined fp24 adder is FP24::operator+, and the inlined fp16 mul / add are FP16::operator* / operator+.
//  * vcomiss sets CF when "below or unordered" (CfBelow) and ZF when "equal or unordered" (CfZfSet).
#include "engines/meshnet.h"
#include "globals.h"
#include "engines/mfu.h"
#include "engines/mne.h"
#include "math/numeric_types.h"
#include "engines/memaccessor.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <set>

// TODO(layout): defined by the globals owner (act0.cpp / conv2d.cpp / dm.cpp declare the same objects locally).
extern uint32_t PSUM_L1[];       // L1 partial-sum buffer
extern uint8_t debug_flag;       // dump request flag (cleared after the dump)
extern std::string debug_file;   // dump file path

// Connection code (cfg_code, 1..14) -> producer node id stored in an input byte.  Out of range: 0xFF (unconnected).
// Expansion of the compiler lookup table CSWTCH.874 (ELF .rodata @0x481628).
static uint8_t ProducerNodeFromCode(uint8_t cfg_code)
{
  switch (cfg_code) {
    case 1:  return 31;
    case 2:  return 32;
    case 3:  return 27;
    case 4:  return 28;
    case 5:  return 29;
    case 6:  return 30;
    case 7:  return 19;
    case 8:  return 20;
    case 9:  return 21;
    case 10: return 22;
    case 11: return 23;
    case 12: return 24;
    case 13: return 25;
    case 14: return 26;
    default: return 0xFF;   // the original stores an undefined leftover register here
  }
}

// Target selector (cfg_target, 0..8) -> consumer node index.  Out of range: -1 (nothing is written).
// Expansion of the compiler lookup table CSWTCH.876 (ELF .rodata @0x481618).
static int ConsumerNodeFromTarget(uint8_t cfg_target)
{
  switch (cfg_target) {
    case 0: return 33;
    case 1: return 19;
    case 2: return 20;
    case 3: return 21;
    case 4: return 22;
    case 5: return 23;
    case 6: return 24;
    case 7: return 25;
    case 8: return 26;
    default: return -1;
  }
}


namespace {

using FP16::fp16;
using FP24::fp24;

// CF after `vcomiss a, b`: set when a < b or the operands are unordered.
inline bool CfBelow(float a, float b) { return !(a >= b); }
// CF | ZF after `vcomiss a, b`: a <= b or unordered.
inline bool CfZfSet(float a, float b) { return !(a > b); }

// vcvttss2si: truncation, 0x80000000 ("integer indefinite") for NaN / out of range.
inline int32_t TruncToInt32(float f)
{
  if (!(f > -2147483904.0f && f < 2147483648.0f))
    return INT32_MIN;
  return static_cast<int32_t>(f);
}

// verified against asm @0x43e3e4..0x43eb7e (MfuAct1 stores) and ELF .rodata: .rodata 0x522060 = 127, 522080 = 255,
// 5220A0 = -127, 5220B0 = 32767, 5220C0 = -32767 (NOT -32768).
constexpr int32_t kInt8Max = 127;         // .rodata 0x522060
constexpr int32_t kInt8Min = -127;        // .rodata 0x5220A0
constexpr int32_t kUint8Max = 255;        // .rodata 0x522080
constexpr int32_t kInt16Max = 32767;      // .rodata 0x5220B0
constexpr int32_t kInt16Min = -32767;     // .rodata 0x5220C0
// verified against asm @0x43e8ae: `vandps abs; vucomiss .rodata 0x481410 (= FLT_MAX 0x7f7fffff); jbe`: a fp16 widened to float is
// saturated to 0x7bff/0xfbff only when it is NaN or +-infinity (for rounded fp16 this equals "|x| > 65504").
constexpr float kHalfInfLimit = 65504.0f;

inline uint16_t Load16(const uint8_t * p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline void Store16(uint8_t * p, uint16_t v) { std::memcpy(p, &v, 2); }

// GLB address -> host pointer: (addr >> 28) selects the segment, the low 28 bits are the offset.
inline uint8_t * GlbPtr(uint64_t addr) { return g_GLB[addr >> 28] + (addr & 0xFFFFFFF); }

// Three strides of a shape register (MFU::GetCHW).
struct Chw {
  uint32_t c_, h_, w_;
  explicit Chw(uint64_t reg) { MFU::GetCHW(reg, c_, h_, w_); }
};

// Four-halfword dims register: (w, h, c, n).
struct Dims {
  uint32_t w_, h_, c_, n_;
  explicit Dims(uint64_t reg)
      : w_(reg & 0xFFFF), h_((reg >> 16) & 0xFFFF), c_((reg >> 32) & 0xFFFF), n_(reg >> 48) {}
  bool Valid() const { return w_ && h_ && c_ && n_; }
};

// Converts a linear element index of a tensor with logical dims `d` into the element offset in memory laid out with
// strides `s`; `channel` (optional) receives the channel (j) coordinate.
uint32_t TensorOffset(uint32_t index, const Dims & d, const Chw & s, uint32_t * channel = nullptr)
{
  const uint16_t i = static_cast<uint16_t>(index / (d.w_ * d.h_ * d.c_));
  const uint16_t j = static_cast<uint16_t>(index / (d.w_ * d.h_) - i * d.c_);
  const uint32_t plane = j + i * d.c_;
  const uint16_t k = static_cast<uint16_t>(index / d.w_ - plane * d.h_);
  const uint16_t l = static_cast<uint16_t>(index - d.w_ * (index / d.w_));
  if (channel)
    *channel = j;
  return s.w_ * (k + s.h_ * (s.c_ * i + j)) + l;
}

// Piece-wise linear tail shared by mfu_linefit / act1_linefit: y = clamp(slope * x * 2^shift + icpt, lo, hi) with the
// compare chain of the original (NaNs fall through the same branches as the vcomiss flags do).
uint16_t FitSegment(fp16 x, fp16 slope, fp16 icpt, fp16 lo, fp16 hi, signed char shift)
{
  fp16 t = x * slope;
  t = t.fp16_2exp_shift(shift);
  t = t + icpt;

  const float t_f = t.f_value(), lo_f = lo.f_value(), hi_f = hi.f_value();
  uint16_t out = lo.bits_;
  if (CfBelow(lo_f, t_f)) {                 // lo < t
    if (CfBelow(t_f, hi_f)) {               // t < hi
      const fp16 chosen = CfZfSet(lo_f, t_f) ? t : lo;
      out = CfZfSet(chosen.f_value(), hi_f) ? chosen.bits_ : hi.bits_;
    } else {
      out = hi.bits_;
    }
  }
  return out;
}

// float of a rounded fp16, with NaN and +-infinity saturated to +-65504 (0x7BFF / 0xFBFF).
float SaturatedHalf(fp16 v)
{
  const float f = v.f_value();
  if (std::isnan(f) || std::fabs(f) > kHalfInfLimit)
    return (v.bits_ & 0x8000) ? -65504.0f : 65504.0f;
  return f;
}

// Integer value stored by the Act1 destination stages: round to nearest even in fp16, saturate, truncate.
int32_t HalfToInteger(fp16 v)
{
  const fp16 rounded = fp16::round_to_fp16(std::nearbyint(v.f_value()));
  return TruncToInt32(SaturatedHalf(rounded));
}

}  // namespace

// ==================================================================================================================
// construction / destruction
// ==================================================================================================================

// Initial state of the singleton (inlined into MeshNet::GetMeshNet @0x445f80): every node has its three input bytes
// at 0xFF (unconnected), everything else is zero.
MeshNet::MeshNet()
{
  std::memset(static_cast<void *>(this), 0, offsetof(MeshNet, order_));          // reduce_fn, nodes, flags
  for (MeshNode & n : node_)
    n.in_[0] = n.in_[1] = n.in_[2] = 0xFF;
  std::memset(reserved1672_, 0, sizeof reserved1672_);
  std::memset(reserved1680_, 0, sizeof reserved1680_);
  // everything behind the four trace streams (string_slot .. a1_fit_addr)
  std::memset(static_cast<void *>(&string_slot_), 0, sizeof(MeshNet) - offsetof(MeshNet, string_slot_));
  // verified against asm @0x445f80: the object lives at 0x54AA60, so its fields sit at 0x54BAB8..0x54BAC8
  // +0x1058.. (a1_s1_scale .. a1_use_mfu_fit); the original's explicit 16-bit zero stores at +0x1058 / +0x105e are
  // covered by the memset above.
}

// @0x426ba0 (MeshNet1): the original body only destroys the four trace streams, the order vector and the (empty) string.
MeshNet::~MeshNet() = default;

// @0x445f80 (MeshNet16)
MeshNet * MeshNet::GetMeshNet()
{
  static MeshNet instance;      // the original instance is MeshNet::GetMeshNet()::MeshNetInst
  return &instance;
}

// ==================================================================================================================
// graph configuration
// ==================================================================================================================

// @0x438620 (MeshNet2)
// Writes the producer node selected by cfg_code into input byte `cfg_slot` (0..31) of the graph.
// Slots 0..21 are in0 / in1 of nodes 0..10, slots 22..28 are in0 of nodes 11..17, 29..31 are in0..in2 of node 18.
void MeshNet::MeshNetOpRoutConfig()
{
  // verified against asm @0x438620: tables/ranges match; for an out-of-range code the original stores the caller's leftover
  // ecx (never written in this function), so the value is undefined; 0xFF (unconnected) is the chosen deterministic default.
  const uint8_t source = ProducerNodeFromCode(cfg_code_);

  static const uint8_t kSlotNode[32] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                                        8, 8, 9, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18, 18};
  static const uint8_t kSlotInput[32] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                         0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2};
  if (cfg_slot_ < 32)
    node_[kSlotNode[cfg_slot_]].in_[kSlotInput[cfg_slot_]] = source;
}

// @0x438830 (MeshNet3)
// node[ConsumerNodeFromTarget(cfg_target)].in[0] = cfg_source (0..18, otherwise 0)
void MeshNet::MeshNetRoutOpConfig()
{
  const uint8_t source = cfg_source_ <= 0x12 ? cfg_source_ : 0;
  // verified against asm @0x438830: for cfg_target > 8 the table load is skipped and the leftover rax is used as node index
  // (undefined in the original); nothing is written here.
  const int target = ConsumerNodeFromTarget(cfg_target_);
  if (target >= 0)
    node_[target].in_[0] = source;
}

// @0x438870 (MeshNet4)
// node[ConsumerNodeFromTarget(cfg_target)].in[0] = ProducerNodeFromCode(cfg_code)
void MeshNet::MeshNetRoutRoutConfig()
{
  const uint8_t source = ProducerNodeFromCode(cfg_code_);   // verified against asm @0x438870: leftover ecx (undefined) for an out-of-range code
  const int target = ConsumerNodeFromTarget(cfg_target_);   // leftover edx (undefined) for the node index otherwise
  if (target >= 0)
    node_[target].in_[0] = source;
}

// @0x4388c0 (MeshNet5)
// Maps the linear element `index` of a broadcast operand to the index inside its source slice: every group of
// slice_len * rpt_a * rpt_b elements reads one slice (each element repeated rpt_a times, the slice rpt_b times).
uint32_t MeshNet::GetBroadAddress(uint32_t len, uint32_t rpt_a, uint32_t slice_len, uint32_t rpt_b, uint32_t index,
                                  uint8_t k0, uint8_t k1, uint8_t k2)
{
  const uint32_t group = slice_len * rpt_a * rpt_b;
  const uint32_t pos = index % (group * (len / slice_len));
  return pos / group * (k2 + 1) * ((k0 + 1) * slice_len + k1) + (k0 + 1) * (pos % group / rpt_a % slice_len);
}

// ==================================================================================================================
// reduction
// ==================================================================================================================

// @0x438930 (MeshNet6)
// Folds `values` into an fp24 accumulator (initial value reduce_init) with reduce_fn and returns it as bfloat16.
BF16::bfloat16 MeshNet::MnReduceProc(const std::vector<BF16::bfloat16> & values) const
{
  fp24 acc = fp24::round_to_fp24(BF16::bfloat16::from_bits(reduce_init_).f_value());
  for (uint32_t i = 0; i < reduce_len_; ++i) {
    fp24 x = fp24::round_to_fp24(values[i].f_value());
    reduce_fn_(&acc, &x, &acc, nullptr);
  }
  return BF16::bfloat16::from_bits(static_cast<uint16_t>(BF16::bfloat16::round_to_bfloat16(acc.f_value())));
}

// @0x43a8a0 (MeshNet11)
// verified against asm @0x43a8a0: reduce_op (+0xf20) is read from the singleton 0x54AA60, reduce_fn is stored through `this`
// (same object); 1 min, 2 add, 3 sub (0x444a30), 4 mul, otherwise max.
void MeshNet::MeshNetReduce()
{
  switch (GetMeshNet()->reduce_op_) {
    case 1:  reduce_fn_ = REDUCE_ELEMENT::re_min; break;
    case 2:  reduce_fn_ = REDUCE_ELEMENT::re_add; break;
    case 3:  reduce_fn_ = REDUCE_ELEMENT::re_sub; break;
    case 4:  reduce_fn_ = REDUCE_ELEMENT::re_mul; break;
    default: reduce_fn_ = REDUCE_ELEMENT::re_max; break;
  }
}

// ==================================================================================================================
// line fit
// ==================================================================================================================

// @0x438a20 (MeshNet7)
// Parameter set `param_set` of the table at GLB address `table_addr` holds (3 * segments + 1) fp16 values:
//   knee[segments - 1], slope[segments], intercept[segments], lower clamp, upper clamp.
// The segment is the first one whose knee is above x (the last segment when none is).
void MeshNet::mfu_linefit(FP16::fp16 * x, FP16::fp16 * y, uint32_t table_addr, uint8_t ** glb, uint16_t param_set,
                          uint8_t segments)
{
  const MemAccessor table(&glb[table_addr >> 28][table_addr & 0xFFFFFFF]);
  const int32_t set_base = 2 * (3 * segments + 1) * param_set;      // byte offset of the parameter set
  const int32_t knees = segments - 1;

  const float x_f = x->f_value();
  int32_t seg = knees;
  for (int32_t i = 0; i < knees; ++i) {
    const fp16 knee = table.MemAt<FP16::fp16>(set_base + 2 * i);
    if (knee.f_value() > x_f) {
      seg = i;
      break;
    }
  }
  const fp16 slope = table.MemAt<FP16::fp16>(set_base + 2 * knees + 2 * seg);
  const fp16 icpt = table.MemAt<FP16::fp16>(set_base + 4 * knees + 2 + 2 * seg);
  const fp16 lo = table.MemAt<FP16::fp16>(set_base + 6 * segments - 2);
  const fp16 hi = table.MemAt<FP16::fp16>(set_base + 6 * segments);
  y->bits_ = FitSegment(*x, slope, icpt, lo, hi, static_cast<signed char>(a1_fit_shift_));
}

// @0x438c70 (MeshNet8)
// Parameter set `param_set` holds 7 fp16 values: knee, slope[2], intercept[2], lower clamp, upper clamp.
// Segment 0 applies when x < knee (or unordered), otherwise segment 1.
void MeshNet::act1_linefit(FP16::fp16 * x, FP16::fp16 * y, uint32_t table_addr, uint8_t ** glb, uint16_t param_set)
{
  const MemAccessor table(&glb[table_addr >> 28][table_addr & 0xFFFFFFF]);
  const int32_t set_base = 14 * param_set;

  const fp16 knee = table.MemAt<FP16::fp16>(set_base);
  const fp16 lo = table.MemAt<FP16::fp16>(set_base + 10);
  const fp16 hi = table.MemAt<FP16::fp16>(set_base + 12);
  fp16 slope, icpt;
  if (CfBelow(x->f_value(), knee.f_value())) {
    slope = table.MemAt<FP16::fp16>(set_base + 2);
    icpt = table.MemAt<FP16::fp16>(set_base + 6);
  } else {
    slope = table.MemAt<FP16::fp16>(set_base + 4);
    icpt = table.MemAt<FP16::fp16>(set_base + 8);
  }
  y->bits_ = FitSegment(*x, slope, icpt, lo, hi, static_cast<signed char>(a1_fit_shift_));
}

// @0x438ea0 (MeshNet9)
// Walks the rows of the (n, c, h) slices of a tensor whose logical dims are `dims` and whose memory strides are
// `chw_reg`, starting at L1 address `addr`, until `slice_len` elements are collected, and counts the 32-byte L1 lines
// the rows touch.  Exits when the tensor ends before slice_len is reached.
uint32_t MeshNet::L1_slice_rpt_buffer_len(uint32_t addr, uint64_t dims_reg, uint64_t chw_reg, uint32_t slice_len,
                                          uint32_t elem_bytes)
{
  const Dims d(dims_reg);
  const uint32_t sw = chw_reg & 0xFFFF, sh = (chw_reg >> 16) & 0xFFFF, sc = (chw_reg >> 32) & 0xFFFF;
  const uint32_t row_stride = elem_bytes * sw;
  const uint32_t plane_stride = elem_bytes * sw * sh;
  const uint32_t batch_stride = elem_bytes * sw * sh * sc;
  const uint32_t row_bytes = elem_bytes * d.w_;

  int32_t collected = 0;
  uint32_t lines = 0;
  for (uint32_t n = 0; n < d.n_; ++n) {
    for (uint32_t c = 0; c < d.c_; ++c) {
      for (uint32_t h = 0; h < d.h_; ++h) {
        const uint32_t row_addr = addr + n * batch_stride + c * plane_stride + h * row_stride;
        collected += d.w_;
        lines += (row_bytes + 31 + (row_addr & 0x1F)) >> 5;
        if (static_cast<int32_t>(slice_len) == collected)
          return lines;
      }
    }
  }
  std::printf("slice_len is not multiples of ShapeW, slice_len = %d, ShapeW = %d! \n", static_cast<int>(slice_len),
              static_cast<int>(d.w_));
  std::exit(1);
}

// ==================================================================================================================
// mesh graph
// ==================================================================================================================

// @0x43a660 (MeshNet10)
// Selects the operator of every node.  Nodes 0..11, 13 and 15 take their MneProc argument from mne_arg[], the other
// nodes (12, 14 and the non-MNE nodes 16..33) use 0.
void MeshNet::MeshNetOp()
{
  for (uint32_t i = 0; i < 34; ++i) {
    uint8_t arg = 0;
    if (i <= 11)
      arg = mne_arg_[i];
    else if (i == 13)
      arg = mne_arg_[12];
    else if (i == 15)
      arg = mne_arg_[13];
    reinterpret_cast<MNE *>(&node_[i])->MneProc(static_cast<uint8_t>(i), arg);
  }
}

// @0x440020 (MeshNet13)
// Appends the producers of the node at order[index] to `order` (in0, in1, in2) and recurses into them;
// order_count tracks the index of the last appended entry.
void MeshNet::MnConstruct(uint8_t index)
{
  for (;;) {
    uint8_t id = order_[index];
    if (node_[id].in_[0] != 0xFF) {
      ++order_count_;
      order_.push_back(node_[order_[index]].in_[0]);
      MnConstruct(order_count_);
      id = order_[index];
    }
    if (node_[id].in_[1] != 0xFF) {
      ++order_count_;
      order_.push_back(node_[id].in_[1]);
      MnConstruct(order_count_);
      id = order_[index];
    }
    if (node_[id].in_[2] == 0xFF)
      return;
    ++order_count_;
    order_.push_back(node_[id].in_[2]);
    index = order_count_;      // the third producer is handled by the loop (tail call in the original)
  }
}

// @0x440200 (MeshNet14)
// Removes duplicates from `order`, keeping the last occurrence of every node (so producers come after consumers).
void MeshNet::MnPrune()
{
  std::set<uint8_t> seen;
  std::vector<uint8_t> kept;
  for (auto it = order_.rbegin(); it != order_.rend(); ++it)
    if (seen.insert(*it).second)
      kept.push_back(*it);
  order_.assign(kept.rbegin(), kept.rend());
}

// @0x442570 (MeshNet15)
// Runs the graph over elem_count elements.  Per element: load operand a (input 0) and b (input 1) with optional
// broadcast and 8-bit de-quantisation, evaluate the nodes in reverse `order`, and store node 33's result (optionally
// quantised); with reduce_enable every reduce_len results are folded by MnReduceProc into one output element.
// verified against asm @0x442570: order_count is neither reset before MnConstruct nor cleared (it is only incremented
// once after the construct and then overwritten by order.size()), exactly as written below.
void MeshNet::MnCompute()
{
  order_.clear();
  order_.push_back(33);          // the sink
  MnConstruct(0);
  ++order_count_;
  MnPrune();
  order_count_ = static_cast<uint8_t>(order_.size());

  const MemAccessor in0(GlbPtr(src0_addr_)), in1(GlbPtr(src1_addr_));
  uint8_t * const out0 = GlbPtr(dst0_addr_);
  uint8_t * const out1 = GlbPtr(dst1_addr_);

  const Chw chw0(chw_in0_), chw1(chw_in1_), chw_o(chw_out_);
  const Dims d0(in0_dims_), d1(in1_dims_), dout(out_dims_);

  // operand pointers: every node reads the `result` of its producers.  An unconnected input (0xFF) points past the
  // node array, as in the original (the operator never reads it).
  auto result_of = [this](uint8_t id) {
    return reinterpret_cast<BF16::bfloat16 *>(reinterpret_cast<uint8_t *>(this) + 48 * id + 48);
  };
  for (int k = order_count_ - 1; k >= 0; --k) {
    MeshNode & n = node_[order_[k]];
    n.operand_[0] = result_of(n.in_[0]);
    n.operand_[1] = result_of(n.in_[1]);
    n.operand_[2] = result_of(n.in_[2]);
  }
  node_[27].operand_[0] = reinterpret_cast<BF16::bfloat16 *>(&const_val_[0]);
  node_[28].operand_[0] = reinterpret_cast<BF16::bfloat16 *>(&const_val_[1]);
  node_[29].operand_[0] = reinterpret_cast<BF16::bfloat16 *>(&const_val_[2]);
  node_[30].operand_[0] = reinterpret_cast<BF16::bfloat16 *>(&const_val_[3]);

  uint16_t value_a = 0, value_b = 0;      // node 31 / 32 inputs
  // verified against asm @0x442570: `channel` is the 16-bit stack slot rbp-0xda, written by the TensorOffset of operand 0
  // (0x442d3c), operand 1 (0x4430f2) and the output (0x442bfd / 0x443219) and read by node 17; it is never initialised, so
  // 0 for the first element when neither operand is loaded is a deterministic choice.
  uint32_t channel = 0;
  std::vector<BF16::bfloat16> group;      // results waiting for the next reduction
  uint32_t in_group = 0;

  // Quantises `value` with the given parameters (bf16 scale / zero point).
  auto quantize = [](uint16_t value, uint16_t scale, uint16_t zero, uint8_t is_signed) {
    const BF16::bfloat16 x = BF16::bfloat16::from_bits(value);
    const BF16::bfloat16 s = BF16::bfloat16::from_bits(scale);
    const uint32_t zp = zero;
    uint8_t out = 0;
    MFU::quant(x, s, zp, is_signed, out);
    return out;
  };
  // Stores `value` at element `index` of an output tensor (8 bit quantised or raw bf16).
  auto store = [&](uint8_t * dst, uint32_t index, uint16_t value, bool enable, uint16_t scale, uint16_t zero,
                   uint8_t is_signed) {
    if (enable)
      dst[index] = quantize(value, scale, zero, is_signed);
    else
      Store16(dst + 2 * static_cast<int32_t>(index), value);
  };
  // Loads one operand element: 8 bit data is de-quantised, otherwise a raw bf16.
  auto load = [](const MemAccessor & src, uint32_t offset, bool enable, uint16_t scale, uint8_t zero,
                 uint8_t is_signed) {
    BF16::bfloat16 out;
    if (enable) {
      const uint8_t q = src.MemAt<uint8_t>(offset);
      const BF16::bfloat16 s = BF16::bfloat16::from_bits(scale);
      MFU::dequant(q, s, zero, is_signed, out);
    } else {
      out = src.MemAt<BF16::bfloat16>(2 * offset);
    }
    return out.bits_;
  };

  for (int32_t idx = 0; idx < static_cast<int32_t>(elem_count_); ++idx) {
    value_a = 0;
    value_b = 0;
    if (d0.Valid() && in0_len_ != 0) {
      const uint32_t ba = GetBroadAddress(in0_len_, in0_rpt_a_, in0_slice_len_, in0_rpt_b_, idx, 0, 0, 0);
      const uint32_t off = TensorOffset(ba, d0, chw0, &channel);
      value_a = load(in0, off, dq0_enable_, dq0_scale_, dq0_zero_, dq0_signed_);
    }
    if (d1.Valid() && in1_len_ != 0) {
      const uint32_t ba = GetBroadAddress(in1_len_, in1_rpt_a_, in1_slice_len_, in1_rpt_b_, idx, 0, 0, 0);
      const uint32_t off = TensorOffset(ba, d1, chw1, &channel);
      value_b = load(in1, off, dq1_enable_, dq1_scale_, dq1_zero_, dq1_signed_);
    }
    node_[31].operand_[0] = reinterpret_cast<BF16::bfloat16 *>(&value_a);
    node_[32].operand_[0] = reinterpret_cast<BF16::bfloat16 *>(&value_b);

    // evaluate the graph (producers first)
    for (int k = order_count_ - 1; k >= 0; --k) {
      const uint8_t id = order_[k];
      MeshNode & n = node_[id];
      uint32_t cfg = n.cfg_;
      uint16_t function_set = 0;
      if (id == 17) {
        cfg = node17_cfg_;
        if (node17_per_channel_)
          function_set = static_cast<uint16_t>(channel);
      }
      n.op_(n.operand_[0], n.operand_[1], n.operand_[2], &n.result_, cfg, g_GLB, function_set);
    }
    const uint16_t result = node_[33].result_.bits_;

    // output element offset
    const uint32_t out_off = TensorOffset(idx, dout, chw_o, &channel);

    if (!reduce_enable_) {
      store(out0, out_off, result, q0_enable_, q0_scale_, q0_zero_, q0_signed_);
      continue;
    }

    if (write_both_)
      store(out0, out_off, result, q0_enable_, q0_scale_, q0_zero_, q0_signed_);
    if (++in_group == reduce_len_)
      in_group = 0;
    group.push_back(BF16::bfloat16::from_bits(result));
    if (in_group != 0)
      continue;

    const uint16_t reduced = MnReduceProc(group).bits_;
    const uint32_t reduced_index = idx / reduce_len_;
    if (write_both_)
      store(out1, reduced_index, reduced, q1_enable_, q1_scale_, q1_zero_, q1_signed_);
    else
      store(out0, reduced_index, reduced, q0_enable_, q0_scale_, q0_zero_, q0_signed_);
    group.clear();
  }
}

// ==================================================================================================================
// MfuAct1
// ==================================================================================================================

// @0x43da10 (MeshNet12)
// dst = linefit(src1 (+|*) src2) element-wise over 4-D tensors.  src1 comes from GLB or PSUM_L1, src2 from GLB; both can
// be broadcast (GetBroadAddress) and de-quantised, the result is stored as fp16 / uint8 / int8 / int16.
void MeshNet::MfuAct1()
{
  MFU::GetMFU();      // instantiate the MFU singleton (result unused)

  const Dims s1(a1_src1_dims_), s2(a1_src2_dims_), loop(a1_dst_dims_);
  const Chw chw1(a1_src1_chw_), chw2(a1_src2_chw_);
  const bool src1_valid = s1.Valid(), src2_valid = s2.Valid();
  const uint32_t addr1 = a1_src1_addr_ & 0xFFFFFFF, addr2 = a1_src2_addr_ & 0xFFFFFFF;

  // ---- parameter validation (each failure prints a message and exits) ------------------------------------------
  auto fail = [](const char * msg) { std::puts(msg); std::exit(1); };

  if (!a1_src1_psum_ && src1_valid && (a1_s1_type_ == 0 || a1_s1_type_ == 3) && (a1_src1_addr_ & 1)) {
    std::printf("when data type is fp16 or int16, act1 addr_s1 must be align with 2 bytes, addr_src1 = %d! \n", addr1);
    std::exit(1);
  }
  if (!a1_src2_psum_ && src2_valid && (a1_s2_type_ == 0 || a1_s2_type_ == 3) && (a1_src2_addr_ & 1)) {
    std::printf("when data type is fp16 or int16, act1 addr_s2 must be align with 2 bytes, addr_src2 = %d! \n", addr2);
    std::exit(1);
  }
  if ((a1_dst_type_ == 0 || a1_dst_type_ == 3) && (a1_dst_addr_ & 1)) {
    std::printf("when data type is fp16 or int16, act1 addr_d must be align with 2 bytes, addr_dest = %d! \n",
                a1_dst_addr_ & 0xFFFFFFF);
    std::exit(1);
  }
  if (a1_fit_addr_ & 1) {
    std::printf("act1 parameter addr_arg must be align with 2 bytes, addr_arg = %d! \n", a1_fit_addr_ & 0xFFFFFFF);
    std::exit(1);
  }

  // src1 read from GLB with repeated slices: the repeated rows must fit the 128-line RTL L1 buffer
  if (src1_valid && a1_src1_psum_ == 0 && !a1_s1_no_l1_check_ && a1_s1_rpt_b_ > 1) {
    const uint32_t elem = (a1_s1_type_ == 0 || a1_s1_type_ == 3) ? 2 : 1;
    if (L1_slice_rpt_buffer_len(addr1, a1_src1_dims_, a1_src1_chw_, a1_s1_slice_len_, elem) > 0x80)
      fail("Src1 slice len is bigger than RTL L1 buffer 128! ");
  }
  if (src2_valid) {
    // with src1 in PSUM the src2 channel stride (h * w elements) must be a multiple of 32 bytes
    if (a1_src1_psum_ == 1 &&
        (chw2.w_ * chw2.h_) % ((static_cast<uint8_t>(a1_s2_type_ - 1) < 2) ? 32 : 16) != 0) {
      std::printf("conv_fuse_act1 GLB input Stride_C must be align with 32 bytes, stride_h = %d, stride_w = %d! \n",
                  chw2.h_, chw2.w_);
      std::exit(1);
    }
    if (!a1_s2_no_l1_check_ && a1_s2_rpt_b_ > 1) {
      const uint32_t elem = (a1_s2_type_ == 0 || a1_s2_type_ == 3) ? 2 : 1;
      const uint32_t lines = L1_slice_rpt_buffer_len(addr2, a1_src2_dims_, a1_src2_chw_, a1_s2_slice_len_, elem);
      if (a1_src1_psum_) {
        if (lines > 0x40)
          fail("Src2 slice len is bigger than RTL L1 buffer 64! ");
      } else if (lines > 0x80) {
        fail("Src2 slice len is bigger than RTL L1 buffer 128! ");
      }
    }
  }

  const uint32_t src1_len = s1.w_ * s1.h_ * s1.c_ * s1.n_;
  const uint32_t src2_len = s2.w_ * s2.h_ * s2.c_ * s2.n_;
  if (a1_dst_len_ != src1_len * a1_s1_rpt_b_ * a1_s1_rpt_a_ * a1_s1_rpt_c_)
    fail("src1 len doesn't accord with dst len, please check src1 shape size and act1_rleft_repeats_s1! ");
  if (a1_dst_len_ != src2_len * a1_s2_rpt_b_ * a1_s2_rpt_a_ * a1_s2_rpt_c_ && src2_len != 0)
    fail("src2 len doesn't accord with dst len, please check src2 shape size and act1_rleft_repeats_s2! ");

  // ---- operands ---------------------------------------------------------------------------------------------------
  // src1 is either a PSUM_L1 byte pointer or a GLB MemAccessor; src2 and dst are always GLB.
  const uint8_t * const psum = a1_src1_psum_
      ? reinterpret_cast<const uint8_t *>(PSUM_L1) + 4 * (a1_src1_addr_ >> 2) : nullptr;
  const MemAccessor src1_mem(a1_src1_psum_ ? nullptr : g_GLB[a1_src1_addr_ >> 28] + addr1);
  const MemAccessor src2_mem(g_GLB[a1_src2_addr_ >> 28] + addr2);
  uint8_t * const dst = g_GLB[a1_dst_addr_ >> 28] + (a1_dst_addr_ & 0xFFFFFFF);

  // dst pitches (halfwords of a1_dst_chw) and loop extents
  const uint32_t pitch_w = a1_dst_chw_ & 0xFFFF, pitch_h = (a1_dst_chw_ >> 16) & 0xFFFF, pitch_c = (a1_dst_chw_ >> 32) & 0xFFFF;

  fp16 src1_val = fp16::round_to_fp16(0.0f);       // keeps its value when an operand is not loaded
  fp16 src2_val = fp16::round_to_fp16(0.0f);

  for (uint32_t n = 0; n < loop.n_; ++n) {
    for (uint32_t c = 0; c < loop.c_; ++c) {
      const uint32_t dst_plane = pitch_w * pitch_h * (c + n * pitch_c);       // element offset of (n, c) in dst
      const uint32_t dense_plane = loop.w_ * loop.h_ * (c + n * loop.c_);        // linear index of (n, c, 0, 0)
      for (uint32_t h = 0; h < loop.h_; ++h) {
        for (uint32_t w = 0; w < loop.w_; ++w) {
          const uint32_t dst_idx = dst_plane + h * pitch_w + w;
          const uint32_t dense = dense_plane + h * loop.w_ + w;
          const uint32_t psum_idx = c * 2048 + h * loop.w_ + w;                // PSUM_L1 holds 2048 elements per channel

          if (src1_valid) {
            if (a1_src1_psum_) {
              src1_val.bits_ = Load16(psum + 2 * psum_idx);
            } else {
              const uint32_t ba = GetBroadAddress(src1_len, a1_s1_rpt_a_, a1_s1_slice_len_, a1_s1_rpt_b_, dense, 0, 0, 0);
              MFU::dequant_new_MemAt(TensorOffset(ba, s1, chw1), src1_mem, fp16(a1_s1_scale_),
                                     static_cast<short>(a1_s1_zero_), a1_s1_type_, a1_s1_shift_, src1_val);
            }
          }

          fp16 sum;
          bool use_sum = false;
          if (src2_valid) {
            const uint32_t ba = GetBroadAddress(src2_len, a1_s2_rpt_a_, a1_s2_slice_len_, a1_s2_rpt_b_, dense, 0, 0, 0);
            MFU::dequant_new_MemAt(TensorOffset(ba, s2, chw2), src2_mem, fp16(a1_s2_scale_),
                                   static_cast<short>(a1_s2_zero_), a1_s2_type_, a1_s2_shift_, src2_val);
            if (a1_op_mul_)
              sum = src1_val * src2_val;
            else
              use_sum = true;
          } else if (a1_op_mul_) {
            sum = src1_val;                 // nothing to multiply with
          } else {
            use_sum = true;
          }
          if (use_sum) {
            // addition is done in fp24 and rounded back to fp16
            const fp24 a = fp24::round_to_fp24(src1_val.f_value());
            const fp24 b = fp24::round_to_fp24(src2_val.f_value());
            sum = fp16::round_to_fp16((a + b).f_value());
          }

          // piece-wise linear stage
          const uint16_t param_set = a1_per_channel_ ? static_cast<uint16_t>(c) : 0;
          fp16 fit;
          if (a1_use_mfu_fit_)
            mfu_linefit(&sum, &fit, a1_fit_addr_, g_GLB, param_set, 16);
          else
            act1_linefit(&sum, &fit, a1_fit_addr_, g_GLB, param_set);

          // store
          switch (a1_dst_type_) {
            case 1:     // uint8
              dst[dst_idx] = static_cast<uint8_t>(std::max(std::min(HalfToInteger(fit), kUint8Max), 0));
              break;
            case 2:     // int8
              dst[dst_idx] = static_cast<uint8_t>(std::max(std::min(HalfToInteger(fit), kInt8Max), kInt8Min));
              break;
            case 3:     // int16
              Store16(dst + 2 * dst_idx,
                      static_cast<uint16_t>(std::max(std::min(HalfToInteger(fit), kInt16Max), kInt16Min)));
              break;
            default:    // fp16
              Store16(dst + 2 * dst_idx, fit.bits_);
              break;
          }
        }
      }
    }
  }

  // ---- optional debug dump of the destination bytes (one hex byte per line) ------------------------------------
  if (debug_flag) {
    std::ofstream dump(debug_file.c_str(), std::ios::out);
    // verified against asm @0x43ddac: the dump starts at (a1_fit_addr & 1), which is always 0 here (odd addresses exit above).
    uint32_t batch_base = 0;
    for (uint32_t n = 0; n < loop.n_; ++n) {
      uint32_t plane_base = batch_base;
      for (uint32_t c = 0; c < loop.c_; ++c) {
        uint32_t row_base = plane_base;
        for (uint32_t h = 0; h < loop.h_; ++h) {
          for (uint32_t w = 0; w < loop.w_; ++w) {
            dump << std::hex << std::setw(2) << std::setfill('0')
                 << static_cast<uint64_t>(dst[row_base + w]) << std::endl;
          }
          row_base += pitch_w;
        }
        plane_base += pitch_w * pitch_h;
      }
      batch_base += pitch_w * pitch_h * pitch_c;
    }
    dump.close();
    debug_flag = 0;
  }
}
