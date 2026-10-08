// Lifted from IDA/Hex-Rays output (CheckPoint1..11.cpp).
#include "engines/checkpoint.h"
#include "globals.h"
#include <iomanip>

namespace {

// Raw layouts of helper types that no header defines yet (only the fields CheckPoint reads).
// TODO(layout): replace by the real L1Helper / WeightsHelper / Matrix4 once they are defined.
struct L1HelperView {
    uint8_t* data_;        // +0
    uint8_t pad8_[4];      // +8
    int32_t elemStride_;   // +12  byte stride between lanes
    int32_t baseOffset_;   // +16  byte offset of the first element
    int32_t aStride_;      // +20  byte stride of the outer loop (count at +32)
    int32_t bStride_;      // +24  byte stride of the inner loop (count at +36)
    int32_t lanes_;        // +28  number of elements per row
    int32_t aCount_;       // +32
    int32_t bCount_;       // +36
};
struct WeightsHelperView {
    uint8_t wide_;         // +0   0 => rows of 24 bytes, otherwise 32
    uint8_t pad1_[7];
    uint8_t* data_;        // +8
    int32_t dimA_;         // +16
    int32_t dimB_;         // +20
};
struct MatrixU8View {
    uint8_t pad0_[8];      // +0
    uint8_t* data_;        // +8
    int32_t dim0_;         // +16
    int32_t dim1_;         // +20
    int32_t dim2_;         // +24
    int32_t count_;        // +28
};

constexpr int kRowLanes = 32;  // every trace row is a vector<int>(32)

// File names; all verified against the string literals in .rodata referenced by OpenPuCheckPoint @0x451ff0 and
// OpenAi2dCheckPoint @0x454db0 (0x52254b..0x522688, 0x52269c..0x522776).
struct StreamFile {
    CheckPoint::StreamId id_;
    const char* name_;
};

const StreamFile kLoadStoreFiles[] = {
    {CheckPoint::kLoadCmd, "/load_cmd.chk"},
    {CheckPoint::kLoadDdrRaddr, "/load_ddr_raddr.chk"},
    {CheckPoint::kLoadDdrRdata, "/load_ddr_rdata.chk"},
    {CheckPoint::kLoadGlbWrite, "/load_glb_write.chk"},
    {CheckPoint::kStoreCmd, "/store_cmd.chk"},
    {CheckPoint::kStoreGlbRaddr, "/store_glb_raddr.chk"},
    {CheckPoint::kStoreGlbRdata, "/store_glb_rdata.chk"},
    {CheckPoint::kStoreDdrWaddr, "/store_ddr_waddr.chk"},
    {CheckPoint::kStoreDdrWdata, "/store_ddr_wdata.chk"},
};

const StreamFile kPuFiles[] = {
    {CheckPoint::kPu0, "/pu_cmd_in.chk"},
    {CheckPoint::kPu1, "/pu_if_in.chk"},
    {CheckPoint::kPuWeIn, "/pu_we_in.chk"},
    {CheckPoint::kPu3, "/pu_pe_if_in.chk"},
    {CheckPoint::kPuPeWeInC0, "/pu_pe_we_in_c0.chk"},
    {CheckPoint::kPuPePsum, "/pu_pe_psum.chk"},
    {CheckPoint::kPu6, "/pu_psum_acc.chk"},
    {CheckPoint::kPu7, "/pu_psum_l1_wr_data.chk"},
    {CheckPoint::kPu8, "/pu_psum_l1_addr.chk"},
    {CheckPoint::kPu9, "/pu_psum_l1_rd_data.chk"},
    {CheckPoint::kPuOutA, "/pu_psum_out.chk"},
    {CheckPoint::kActDmWrite, "/act_dm_write.chk"},
    {CheckPoint::kActCmd, "/act_cmd.chk"},
    {CheckPoint::kAct0Param, "/Act0_param.chk"},
    {CheckPoint::kActPsumL1Write, "/act_psuml1_write.chk"},
    {CheckPoint::kDmIf, "/dm_if.chk"},
    {CheckPoint::kDmOf, "/dm_of.chk"},
    {CheckPoint::kPdp0CmdIn, "/pdp0_cmd_in.chk"},
    {CheckPoint::kPdp0DmWeIn, "/pdp0_dm_we_in.chk"},
    {CheckPoint::kPdp0Unk19, "/pdp0_pe_if_in.chk"},
    {CheckPoint::kPdp0PeWeIn, "/pdp0_pe_we_in.chk"},
    {CheckPoint::kPdp0PeOut, "/pdp0_pe_out.chk"},
    {CheckPoint::kPuOutB, "/pdp0_pu_out_data.chk"},
};

// Note: the AI2D names have no leading '/' in the binary (the directory string is
// expected to end with a separator).
const StreamFile kAi2dFiles[] = {
    {CheckPoint::kAi2d0, "ai2d_para_check.txt"},
    {CheckPoint::kAi2d1, "ai_2d_cmd.chk"},
    {CheckPoint::kAi2d2, "ai_2d_rx_req.chk"},
    {CheckPoint::kAi2dRxDdrRaddr, "ai_2d_rx_ddr_raddr.chk"},
    {CheckPoint::kAi2d4, "ai_2d_rx_ddr_rdata.chk"},
    {CheckPoint::kAi2d5, "ai_2d_src_position.chk"},
    {CheckPoint::kAi2d6, "ai_2d_dst_calc_out.chk"},
    {CheckPoint::kAi2dFinalCalcOut, "ai_2d_final_calc_out.chk"},
    {CheckPoint::kAi2d8, "ai_2d_csc_out.chk"},
    {CheckPoint::kAi2dTxReq, "ai_2d_tx_req.chk"},
    {CheckPoint::kAi2dTxDdrWaddr, "ai_2d_tx_ddr_waddr.chk"},
    {CheckPoint::kAi2d11, "ai_2d_tx_ddr_wdata.chk"},
    {CheckPoint::kAi2dSrcCalcIn, "ai_2d_src_calc_in.chk"},
    {CheckPoint::kAi2d13, "ai_2d_glb_raddr.chk"},
    {CheckPoint::kAi2d14, "ai_2d_glb_rdata.chk"},
    {CheckPoint::kAi2dGlbWrite, "ai_2d_glb_write.chk"},
};

}  // namespace

// Opens `path` for binary output and selects hexadecimal integer output.
// All Open*CheckPoint functions repeat exactly this sequence for every stream:
//   ofstream tmp(path, out|binary); stream = std::move(tmp); stream.setf(hex, basefield)
void CheckPoint::OpenStream(std::ofstream & os, std::string const & path)
{
  if (os.is_open())
    os.close();
  os.clear();
  os.open(path, std::ios::out | std::ios::binary);  // flags 20; failure sets failbit
  os.setf(std::ios::hex, std::ios::basefield);
}

// CheckPoint1.cpp  @0x44ebf0 (Source 1)
void CheckPoint::PrintActPsumL1Write(L1Helper & helper, ActOutputType outType)
{
  const L1HelperView& h = reinterpret_cast<const L1HelperView&>(helper);  // TODO(layout)
  const uint32_t type = static_cast<uint32_t>(outType);
  const bool is8bit = type <= 1;
  const bool is16bit = (type - 2) <= 1;
  int mask = static_cast<int>((1LL << h.lanes_) - 1);
  int counter = 0;  // number of values collected per lane (8-bit units; doubled for 16-bit when flushed)
  // 32 lanes, 8 words each.
  std::vector<std::vector<int>> lanes(32, std::vector<int>(8, 0));

  // CheckPoint::PrintActPsumL1Write(L1Helper&, ActOutputType)::{lambda(int)#1} @0x44e6e0 (flushes the collected lanes)
  // verified against asm @0x44e6e0 (the lambda body): captures [this, &mask, &counter, &lanes, &type].
  // Writes one line to stream +7680 (act_psuml1_write): mask (%08x), address (%03x), (1<<counter)-1 (%02x), then
  // every lane (highest lane first, lanes separated by "_"): 8-bit types print the low byte of each of the lane's
  // words from the last word down to the first (width 2), other types print the low 16 bits of words 3..0
  // (width 4).  Every printed word is cleared afterwards.
  auto flush = [this, &mask, &counter, &lanes, &type](int address) {
    std::ofstream & os = streams_[kActPsumL1Write];
    os << std::setw(8) << std::setfill('0') << mask << " ";
    os << std::setw(3) << std::setfill('0') << address << " ";
    os << std::setw(2) << std::setfill('0') << ((1 << counter) - 1) << " ";
    for (int lane = static_cast<int>(lanes.size()) - 1; lane >= 0; --lane) {
      std::vector<int> & words = lanes[lane];
      if (type <= 1) {
        for (int i = static_cast<int>(words.size()) - 1; i >= 0; --i) {
          os << std::setw(2) << std::setfill('0') << static_cast<int>(static_cast<uint8_t>(words[i]));
          words[i] = 0;
        }
      } else {
        for (int i = 3; i >= 0; --i) {
          os << std::setw(4) << std::setfill('0') << static_cast<int>(static_cast<uint16_t>(words[i]));
          words[i] = 0;
        }
      }
      if (lane != 0)
        os << "_";
    }
    os << std::endl;
  };

  if (h.aCount_ > 0) {
    for (int a = 0; a < h.aCount_; ++a) {
      for (int b = 0; b < h.bCount_; ++b) {
        for (int k = 0; k < h.lanes_; ++k) {
          uint32_t value;
          if (is8bit) {
            value = h.data_[k * h.elemStride_ + b * h.bStride_ + h.baseOffset_ + a * h.aStride_];
          } else {
            value = 0;
            if (is16bit)
              value = *reinterpret_cast<const uint16_t*>(
                  h.data_ + 2LL * static_cast<int>(a * h.aStride_ + (static_cast<uint64_t>(h.baseOffset_) >> 1) +
                                                  b * h.bStride_ +
                                                  (static_cast<uint64_t>(k * h.elemStride_) >> 1)));
          }
          lanes[k][counter] = value;
        }
        ++counter;
        if (counter > 3 && is16bit) {
          counter *= 2;
          int address = (h.baseOffset_ + 2 * (b * h.bStride_ + a * h.aStride_ - 3)) >> 3;
          flush(address);
          counter = 0;
        } else if (is8bit && counter > 7) {
          int address = static_cast<int>(
              static_cast<uint32_t>(h.baseOffset_ + a * h.aStride_ + b * h.bStride_ - 7) >> 3);
          flush(address);
          counter = 0;
        }
      }
    }
  }
  if (counter > 0) {
    // Flush the remaining partial group.
    int last = h.bStride_ * (h.bCount_ - 1) + h.aStride_ * (h.aCount_ - 1);
    int address;
    if (is16bit) {
      uint32_t pos = static_cast<uint32_t>(last - counter);
      counter *= 2;
      address = (h.baseOffset_ + 2 * static_cast<int>(pos) + 2) >> 3;
    } else {
      uint32_t pos = static_cast<uint32_t>(h.baseOffset_ + last - counter);
      address = (static_cast<int>(pos) + 1) >> 3;
    }
    flush(address);
  }
}

// CheckPoint2.cpp  @0x44ef60 (Source 2)
// Writes one line: every value masked to `width` hex nibbles, zero padded (stream is in hex mode).
void CheckPoint::PrintCheckPoint(std::ofstream & os, std::vector<int> const & values, int width, bool reverse)
{
  uint32_t mask = 0xFFFFFFFFu;
  if (width <= 7)
    mask = static_cast<uint32_t>((1 << (4 * width)) - 1);
  auto emit = [&](size_t i) {
    os << std::setw(width) << std::setfill('0') << static_cast<unsigned long>(static_cast<uint32_t>(values[i]) & mask);
  };
  if (reverse) {
    for (int i = static_cast<int>(values.size()) - 1; i >= 0; --i)
      emit(i);
  } else {
    for (size_t i = 0; i < values.size(); ++i)
      emit(i);
  }
  os << std::endl;
}

// CheckPoint3.cpp  @0x44f1d0 (Source 3)
void CheckPoint::PrintPuOut(L1Helper & helper, int useWide)
{
  const L1HelperView& h = reinterpret_cast<const L1HelperView&>(helper);  // TODO(layout)
  const uint32_t* words = reinterpret_cast<const uint32_t*>(h.data_);
  for (int a = 0; a < h.aCount_; ++a) {
    for (int b = 0; b < h.bCount_; ++b) {
      std::vector<int> row(kRowLanes, 0);
      int base = b * h.bStride_ + a * h.aStride_ + static_cast<int>(static_cast<uint64_t>(h.baseOffset_) >> 2);
      uint64_t laneOffset = 0;
      for (int k = 0; k < h.lanes_; ++k) {
        row[k] = words[static_cast<int>(base + (laneOffset >> 2))];
        laneOffset += h.elemStride_;
      }
      if (useWide)
        PrintCheckPoint(Stream(kPuOutB), row, 6, true);
      else
        PrintCheckPoint(Stream(kPuOutA), row, 8, true);
    }
  }
}

// CheckPoint4.cpp  @0x44f380 (Source 4)
// Dumps PDP0 weights (rawWeights) or weights minus bias into the pdp0_dm_we_in trace.
void CheckPoint::PrintPdp0WeSpadWr(WeightsHelper & weights, Matrix4<uint8_t> & bias, bool rawWeights)
{
  const WeightsHelperView& w = reinterpret_cast<const WeightsHelperView&>(weights);  // TODO(layout)
  const MatrixU8View& m = reinterpret_cast<const MatrixU8View&>(bias);                // TODO(layout)
  const int rowBytes = (w.wide_ == 0) ? 24 : 32;
  for (int a = 0; a < w.dimA_; ++a) {
    for (int i = 0; i < w.dimB_; ++i) {
      std::vector<int> row(kRowLanes, 0);
      for (int k = 0; k < m.count_; ++k) {
        const int idx = rowBytes * (i + a * w.dimB_ + w.dimB_ * w.dimA_ * (k / rowBytes)) + k % rowBytes;
        if (rawWeights) {
          row[k] = static_cast<int8_t>(w.data_[idx]);
        } else {
          if (m.dim0_ <= 0 || m.dim1_ <= 0 || m.dim2_ <= 0)
            std::cout << "[Error: Matrix Exceed]" << std::endl;
          row[k] = static_cast<int>(w.data_[idx]) - static_cast<int>(m.data_[k]);
        }
      }
      PrintCheckPoint(Stream(kPdp0DmWeIn), row, 3, true);
    }
  }
}

// CheckPoint5.cpp  @0x44f680 (Source 5)
// Line format: "<address, 6 hex digits> <bytes, most significant first, 2 hex digits each>"
void CheckPoint::PrintStoreOfCheckPoint(std::ofstream & os, std::vector<uint8_t> & bytes, uint32_t & address)
{
  os << std::setw(6) << std::setfill('0') << static_cast<unsigned long>(address) << " ";
  for (int i = static_cast<int>(bytes.size()) - 1; i >= 0; --i)
    os << std::setw(2) << std::setfill('0') << static_cast<long>(bytes[i]);
  os << std::endl;
}

// CheckPoint6.cpp  @0x44f860 (Source 6)
CheckPoint* CheckPoint::GetCheckPoint()
{
  static CheckPoint checkpoint;  // guarded static, destroyed at exit
  return &checkpoint;
}

// CheckPoint7.cpp  @0x44fa50 (Source 7)
void CheckPoint::OpenLoadStoreCheckPoint(std::string const & dir)
{
  for (const StreamFile& f : kLoadStoreFiles)
    OpenStream(streams_[f.id_], dir + f.name_);
}

// CheckPoint8.cpp  @0x451ff0 (Source 8)
void CheckPoint::OpenPuCheckPoint(std::string const & dir)
{
  for (const StreamFile& f : kPuFiles) {
    std::string name = f.name_;
    OpenStream(streams_[f.id_], dir + name);
  }
}

// CheckPoint9.cpp  @0x454db0 (Source 9)
void CheckPoint::OpenAi2dCheckPoint(std::string const & dir)
{
  for (const StreamFile& f : kAi2dFiles) {
    std::string name = f.name_;
    OpenStream(streams_[f.id_], dir + name);
  }
}

// CheckPoint10.cpp @0x4580d0 / CheckPoint11.cpp @0x459ac0: constructor and destructor are
// defaulted in the header (they only construct/destroy the 48 ofstream members).
