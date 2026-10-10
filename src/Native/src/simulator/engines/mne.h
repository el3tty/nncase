#pragma once
// MNE: per-lane "mini numeric engine" of the MeshNet MfuAct1 pipeline.
// Each MNE object (48 bytes, 16 of them inside MeshNet) holds a function pointer to the
// selected bf16 operation (+0) and the op configuration word (+44).
// Lifted from IDA/Hex-Rays output (MNE1..MNE15).
#include <cstddef>
#include <cstdint>
#include "math/numeric_types.h"

#include "engines/mfu_const.h"

struct MNE {
    // Uniform signature of all operations (the decompiler showed `this` as the first operand):
    //   in0, in1, in2(select/cond), out, mode, table base pointers, function-set index.
    typedef void (*Op)(BF16::bfloat16 *in0, BF16::bfloat16 *in1, BF16::bfloat16 *in2,
                       BF16::bfloat16 *out, uint32_t mode, uint8_t **tables, uint16_t fset);

    Op      op_;          // selected operation (set by MneProc)
    uint32_t op_config_;        // op configuration passed to MneProc

    static void mne_phold  (BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);
    static void mne_inout  (BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);
    static void mne_trangle(BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);
    static void mne_logmode(BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);
    static void mne_exp    (BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);
    static void mne_sel    (BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);
    static void mne_comp   (BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);
    static void mne_round  (BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);
    static void mne_mul    (BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);
    static void mne_div    (BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);
    static void mne_linefit(BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);
    static void mne_ucalc  (BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);
    static void mne_sqrt   (BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);
    static void mne_addsub (BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t);

    // Select the operation for opcode `opcode` and remember the configuration word.
    // Returns the selected function pointer (as the original did).
    void *MneProc(uint8_t opcode, uint32_t config);
};
