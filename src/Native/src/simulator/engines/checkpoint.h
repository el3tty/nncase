#pragma once
// CheckPoint: debug trace ("check point") files for the NPU blocks.  The singleton owns 48 std::ofstream
// streams, one per trace file.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

enum ActOutputType : int;
struct L1Helper;
struct WeightsHelper;
template <typename T> struct Matrix4;

struct CheckPoint {
    // Index of each stream; original byte offset == index * 512.
    enum StreamId : int {
        kPu0 = 0,            // pu_cmd_in
        kPu1 = 1,            // pu_if_in
        kPuWeIn = 2,         // pu_we_in
        kPu3 = 3,            // pu_pe_if_in
        kPuPeWeInC0 = 4,     // pu_pe_we_in_c0
        kPuPePsum = 5,       // pu_pe_psum
        kPu6 = 6,            // pu_psum_acc
        kPu7 = 7,            // pu_psum_l1_wr_data
        kPu8 = 8,            // pu_psum_l1_addr
        kPu9 = 9,            // pu_psum_l1_rd_data
        kPuOutA = 10,        // pu_psum_out (PrintPuOut flag==0, width 8)
        kDmIf = 11,          // dm_if
        kDmOf = 12,          // dm_of (also written by Act0 via PrintStoreOfCheckPoint)
        kActCmd = 13,        // act_cmd
        kAct0Param = 14,     // Act0_param
        kActPsumL1Write = 15,// act_psuml1_write
        kActDmWrite = 16,    // act_dm_write
        kPdp0CmdIn = 17,     // pdp0_cmd_in
        kPdp0DmWeIn = 18,    // pdp0_dm_we_in (PrintPdp0WeSpadWr)
        kPdp0Unk19 = 19,     // pdp0_pe_if_in (pdp0.cpp prints width-3 rows here)
        kPdp0PeWeIn = 20,    // pdp0_pe_we_in
        kPdp0PeOut = 21,     // pdp0_pe_out
        kPuOutB = 22,        // pdp0_pu_out_data.chk
        kLoadCmd = 23,       // load_cmd
        kLoadDdrRaddr = 24,  // load_ddr_raddr
        kLoadDdrRdata = 25,  // load_ddr_rdata
        kLoadGlbWrite = 26,  // load_glb_write
        kStoreCmd = 27,      // store_cmd
        kStoreDdrWaddr = 28, // store_ddr_waddr
        kStoreDdrWdata = 29, // store_ddr_wdata
        kStoreGlbRaddr = 30, // store_glb_raddr
        kStoreGlbRdata = 31, // store_glb_rdata
        kAi2d0 = 32,         // ai2d_para_check.txt
        kAi2d1 = 33,         // ai_2d_cmd.chk
        kAi2d2 = 34,         // ai_2d_rx_req.chk
        kAi2dRxDdrRaddr = 35,// ai_2d_rx_ddr_raddr
        kAi2d4 = 36,         // ai_2d_rx_ddr_rdata.chk
        kAi2d5 = 37,         // ai_2d_src_position.chk
        kAi2d6 = 38,         // ai_2d_dst_calc_out.chk
        kAi2dFinalCalcOut = 39, // ai_2d_final_calc_out
        kAi2d8 = 40,         // ai_2d_csc_out.chk
        kAi2dTxReq = 41,     // ai_2d_tx_req
        kAi2dTxDdrWaddr = 42,// ai_2d_tx_ddr_waddr
        kAi2d11 = 43,        // ai_2d_tx_ddr_wdata.chk
        kAi2dSrcCalcIn = 44, // ai_2d_src_calc_in
        kAi2d13 = 45,        // ai_2d_glb_raddr.chk
        kAi2d14 = 46,        // ai_2d_glb_rdata.chk
        kAi2dGlbWrite = 47,  // ai_2d_glb_write
        kNumStreams = 48
    };

    // / : the original constructor/destructor only
    // construct/destroy the 48 ofstreams, so the defaults are equivalent.
    CheckPoint() = default;
    ~CheckPoint() = default;
    CheckPoint(const CheckPoint&) = delete;
    CheckPoint& operator=(const CheckPoint&) = delete;

    std::ofstream streams_[kNumStreams];  // 512 bytes each in the original

    std::ofstream& Stream(StreamId id) { return streams_[id]; }
    std::ofstream& Stream(uint32_t byteOffset) { return streams_[byteOffset / 512]; }

    void PrintActPsumL1Write(L1Helper & helper, ActOutputType outType);
    static void PrintCheckPoint(std::ofstream & os, std::vector<int> const & values, int width, bool reverse);
    void PrintPuOut(L1Helper & helper, int useWide);
    void PrintPdp0WeSpadWr(WeightsHelper & weights, Matrix4<uint8_t> & bias, bool rawWeights);
    static void PrintStoreOfCheckPoint(std::ofstream & os, std::vector<uint8_t> & bytes, uint32_t & address);
    static CheckPoint* GetCheckPoint();
    void OpenLoadStoreCheckPoint(std::string const & dir);
    void OpenPuCheckPoint(std::string const & dir);
    void OpenAi2dCheckPoint(std::string const & dir);
    // Closes all streams and returns them to the default (closed, no error) state. Called when a simulator run ends,
    // because the singleton survives between runs when the simulator is loaded as a DLL.
    void Reset();

private:
    static void OpenStream(std::ofstream & os, std::string const & path);
};
