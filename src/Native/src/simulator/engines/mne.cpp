// MNE math helpers on bf16 / fp24.  Lifted from IDA/Hex-Rays output (MNE1..MNE15).
// Note: Hex-Rays dropped most SSE arithmetic (addss/mulss/divss/ucomiss operands); places where the
// operation had to be inferred were re-checked against the machine code (see 'verified against asm' notes).
#include "engines/mne.h"
#include "globals.h"
#include "math/numeric_types.h"
#include "engines/memaccessor.h"
#include <cmath>
#include <cstring>

namespace {

// bf16 bit pattern with denormals flushed to (signed) zero.
inline uint16_t ftz(uint16_t v) { return (v & 0x7F80) ? v : (v & 0x8000); }

inline float bf_to_f(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f; }

// Wrap raw bits into BF16::bfloat16 / FP24::fp24 value objects.
// TODO(layout): bfloat16.h/fp24.h declare these types without data members; copy as many bytes as exist.
template <class T, class U> T make(U v) { T t{}; std::memcpy(&t, &v, sizeof(T) < sizeof(U) ? sizeof(T) : sizeof(U)); return t; }
inline BF16::bfloat16 B(uint16_t v) { BF16::bfloat16 t(0); std::memcpy(&t, &v, sizeof(t) < 2 ? sizeof(t) : 2); return t; }
inline FP24::fp24 F(uint32_t v) { return make<FP24::fp24>(v); }

inline uint16_t bf_mul(uint16_t a, uint16_t b) { return (uint16_t)BF16::operator*(B(a), B(b)); }
inline uint16_t bf_add(uint16_t a, uint16_t b) { return (uint16_t)BF16::AddTwoFp16Simp(B(a), B(b)); }
inline uint16_t bf_round(float f) { return (uint16_t)BF16::bfloat16::round_to_bfloat16(f); }
inline uint16_t bf_from_int(int v) { BF16::bfloat16 t(v); uint16_t r = 0; std::memcpy(&r, &t, sizeof(t) < 2 ? sizeof(t) : 2); return r; }
inline uint32_t fp24_round(float f) { FP24::fp24 d{}; return d.round_to_fp24(f).bits_; }
inline float fp24_to_f(uint32_t v) { uint32_t u = v << 8; float f; std::memcpy(&f, &u, 4); return f; }

// Index used by the sin/cos/ln/exp tables: low two mantissa bits are mirrored inside each group of four.
inline unsigned mirror4(uint16_t v) { return (uint16_t)((v & 0xFFFC) + 3) - (v & 3); }

inline uint16_t rd16(const BF16::bfloat16 *p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline void wr16(BF16::bfloat16 *p, uint16_t v) { std::memcpy(p, &v, 2); }

} // namespace

// @0x444630 (MNE1)
void MNE::mne_phold(BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *, uint32_t, uint8_t **, uint16_t)
{
}

// @0x444640 (MNE2): pass-through
void MNE::mne_inout(BF16::bfloat16 *in0, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *out, uint32_t, uint8_t **, uint16_t)
{
    wr16(out, rd16(in0));
}

// @0x444650 (MNE3): sin (mode 0) / cos (otherwise) table lookup
void MNE::mne_trangle(BF16::bfloat16 *in0, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *out, uint32_t mode, uint8_t **, uint16_t)
{
    uint16_t x = rd16(in0);
    unsigned idx;
    if (x & 0x7F80)
        idx = mirror4(x);
    else
        idx = (uint16_t)((x & 0x8000) + 3);   // denormal -> signed zero
    wr16(out, mode == 0 ? mfu_const::sin_bf16[idx] : mfu_const::cos_bf16[idx]);
}

// @0x4446b0 (MNE4): natural-log table lookup
void MNE::mne_logmode(BF16::bfloat16 *in0, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *out, uint32_t mode, uint8_t **, uint16_t)
{
    uint16_t x = ftz(rd16(in0));
    unsigned idx = (uint16_t)(x | 3) - (x & 3);
    if (mode == 1) {
        wr16(out, mfu_const::ln_bf16_mod0[idx]);
    } else if (mode == 2) {
        uint16_t r = mfu_const::ln_bf16_mod0[idx];
        if (x & 0x8000) r += 0x8000;          // restore sign of the input
        wr16(out, r);
    } else if (mode != 0) {
        wr16(out, 0x7FC0);                    // bf16 NaN
    } else {
        wr16(out, mfu_const::ln_bf16_mod1[idx]);
    }
}

// @0x444750 (MNE5): exp table lookup
void MNE::mne_exp(BF16::bfloat16 *in0, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *out, uint32_t, uint8_t **, uint16_t)
{
    uint16_t x = rd16(in0);
    uint16_t base; unsigned low;
    if (x & 0x7F80) { base = x & 0xFFFC; low = x & 3; }
    else            { base = x & 0x8000; low = 0; }
    wr16(out, mfu_const::exp_bf16[(uint16_t)(base + 3) - low]);
}

// @0x444790 (MNE6): select. out = (float(cond) == 0) ? in1 : in0   (NaN cond selects in0)
void MNE::mne_sel(BF16::bfloat16 *in0, BF16::bfloat16 *in1, BF16::bfloat16 *cond, BF16::bfloat16 *out, uint32_t, uint8_t **, uint16_t)
{
    float c = bf_to_f(rd16(cond));
    const BF16::bfloat16 *src = (c == 0.0f) ? in1 : in0;
    wr16(out, rd16(src));
}

// @0x444c20 (MNE7): compare / min / max / arithmetic by mode
void MNE::mne_comp(BF16::bfloat16 *in0, BF16::bfloat16 *in1, BF16::bfloat16 *, BF16::bfloat16 *out, uint32_t mode, uint8_t **, uint16_t)
{
    uint16_t a = ftz(rd16(in0));
    uint16_t b = ftz(rd16(in1));
    float fa = bf_to_f(a), fb = bf_to_f(b);
    switch (mode) {
    case 0:  // min; NaN operand loses
        if (std::isnan(fa))      wr16(out, b);
        else if (std::isnan(fb)) wr16(out, a);
        else                     wr16(out, fa <= fb ? a : b);
        break;
    case 1:  // max; NaN operand loses
        if (std::isnan(fa))      wr16(out, b);
        else if (std::isnan(fb)) wr16(out, a);
        else                     wr16(out, fa <= fb ? b : a);
        break;
    // verified against asm @0x444c20 (jump table @0x481434): comparison results as 1.0 / 0.0 rounded to bf16
    case 2:  // b > a  (vcomiss + seta)
        wr16(out, bf_round(fb > fa ? 1.0f : 0.0f));
        break;
    case 3:  // a >= b (vcomiss + setae)
        wr16(out, bf_round(fa >= fb ? 1.0f : 0.0f));
        break;
    case 4:  // a == b, ordered (vucomiss, setnp / cmovne)
        wr16(out, bf_round(fa == fb ? 1.0f : 0.0f));
        break;
    case 5:  // a != b, NaN counts as not equal
        wr16(out, bf_round(fa != fb ? 1.0f : 0.0f));
        break;
    default:
        wr16(out, 0x7FC0);
        break;
    }
}

// @0x444d90 (MNE8): round
void MNE::mne_round(BF16::bfloat16 *in0, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *out, uint32_t mode, uint8_t **, uint16_t)
{
    uint16_t x = ftz(rd16(in0));
    if (mode <= 2) {
        // verified against asm @0x444d90: vroundss imm 0xc (mode 0, current rounding mode = nearest even),
        // 0x9 (mode 1, floor), 0xa (mode 2, ceil); the result is rounded to bf16.
        const float f = bf_to_f(x);
        const float r = mode == 0 ? std::nearbyint(f) : (mode == 1 ? std::floor(f) : std::ceil(f));
        wr16(out, bf_round(r));
    } else {
        wr16(out, 0x7FC0);
    }
}

// @0x445000 (MNE9): out = (1 - k) * in0 * in1
void MNE::mne_mul(BF16::bfloat16 *in0, BF16::bfloat16 *in1, BF16::bfloat16 *, BF16::bfloat16 *out, uint32_t mode, uint8_t **, uint16_t)
{
    // verified against asm @0x445000: k = round_to_bfloat16((float)(2 * (mode & 1))), i.e. 0.0 or 2.0; (1 - k) is
    // therefore +1.0 (modes 6/8) or -1.0 (modes 7/9, sign flip).
    uint16_t k = bf_round((float)(2 * (mode & 1)));
    uint16_t oneMinusK = bf_add(0x3F80 /* 1.0 */, (uint16_t)(k - 0x8000) /* -k */);
    uint16_t t = bf_mul(oneMinusK, rd16(in0));
    wr16(out, bf_mul(t, rd16(in1)));
}

// @0x445080 (MNE10): out = in0 * c * (1 / in1)
void MNE::mne_div(BF16::bfloat16 *in0, BF16::bfloat16 *in1, BF16::bfloat16 *, BF16::bfloat16 *out, uint32_t mode, uint8_t **, uint16_t)
{
    uint16_t x = ftz(rd16(in0));
    uint16_t y = ftz(rd16(in1));
    float fy = bf_to_f(y);
    // verified against asm @0x445080: recip = round_to_bfloat16(1.0f / y) (vdivss, constant 1.0f @0x5220e8) and
    // c = round_to_bfloat16(1.0f - (float)(2 * (mode & 1))), i.e. +1.0 or -1.0.
    uint16_t recip = bf_round(1.0f / fy);
    uint16_t c = bf_round(1.0f - (float)(2 * (mode & 1)));
    uint16_t t = bf_mul(c, x);
    wr16(out, bf_mul(t, recip));
}

// @0x445130 (MNE11): piecewise-linear fit. `config` encodes table index (bits 28..31) and offset (bits 0..27).
// Coefficients are read from a MemAccessor: [fset*94 + 8 + 2i] thresholds, [+38] slopes, [+70] intercepts.
void MNE::mne_linefit(BF16::bfloat16 *in0, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *out, uint32_t config, uint8_t **tables, uint16_t fset)
{
    uint16_t raw = rd16(in0);
    uint16_t x = ftz(raw);
    MemAccessor mem(tables[config >> 28] + (config & 0xFFFFFFF));
    float fx = bf_to_f(x);
    uint16_t r;
    if (0.0f <= fx || std::isnan(fx)) {
        if (std::isnan(fx)) {
            r = mem.MemAt<uint16_t>(2 * (uint16_t)((raw >> 15) + 2));
        } else {
            const int base = 94 * fset;
            int seg = 30;
            for (int i = 0; i < 30; i += 2) {
                uint16_t thr = mem.MemAt<uint16_t>(base + 8 + i);
                if (bf_to_f(thr) > fx) { seg = i; break; }
            }
            uint16_t slope = mem.MemAt<uint16_t>(base + seg + 38);
            uint16_t icpt  = mem.MemAt<uint16_t>(base + seg + 70);
            r = bf_add(bf_mul(x, slope), icpt);
        }
    } else {
        r = mem.MemAt<uint16_t>(2 * (uint16_t)(raw >> 15));
    }
    wr16(out, r);
}

// @0x4452c0 (MNE12): unary ops: 0 = absolute value, 1 = sign, 2 = negate
void MNE::mne_ucalc(BF16::bfloat16 *in0, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *out, uint32_t mode, uint8_t **, uint16_t)
{
    uint16_t x = ftz(rd16(in0));
    if (mode == 1) {
        float fx = bf_to_f(x);
        // verified against asm @0x445300: sign(x) -> 1 / 0 / -1 (vcomiss + setp/cmove/neg); NaN gives -1.
        int s = (fx > 0.0f) ? 1 : (fx == 0.0f) ? 0 : -1;
        wr16(out, bf_from_int(s));
    } else if (mode == 2) {
        wr16(out, bf_mul(x, 0xBF80 /* -1.0 */));
    } else if (mode != 0) {
        wr16(out, 0x7FC0);
    } else {
        wr16(out, bf_round(std::fabs(bf_to_f(x))));   // verified @0x445330: vandps with the abs mask @0x481400
    }
}

// @0x445380 (MNE13): square root variants
void MNE::mne_sqrt(BF16::bfloat16 *in0, BF16::bfloat16 *, BF16::bfloat16 *, BF16::bfloat16 *out, uint32_t mode, uint8_t **, uint16_t)
{
    uint16_t raw = rd16(in0);
    uint16_t x = ftz(raw);
    float fx = bf_to_f(x);
    // verified against asm @0x445380 (all three modes round |x| / sqrt through round_to_bfloat16; the sqrt of a
    // negative value calls libm sqrtf, i.e. NaN).
    if (mode == 1) {
        // sqrt(|x|)
        float f = bf_to_f(bf_round(std::fabs(fx)));
        wr16(out, bf_round(sqrtf(f)));
    } else if (mode == 2) {
        // signed sqrt: sign(x) * sqrt(|x|), sign = -1 if x < 0 else +1 (NaN -> +1)
        uint16_t sgn = bf_from_int(fx < 0.0f ? -1 : 1);
        float f = bf_to_f(bf_round(std::fabs(fx)));
        wr16(out, bf_mul(bf_round(sqrtf(f)), sgn));
    } else if (mode == 0) {
        // plain sqrt; x < 0 gives the NaN pattern 0x7FC0
        if (0.0f > fx) { wr16(out, 0x7FC0); return; }
        wr16(out, bf_round(sqrtf(fx)));
    } else {
        wr16(out, 0x7FC0);
    }
}

// @0x445500 (MNE14): out = bf16( fp24(c1*in0) + fp24(c2*in1) )  (truncated from fp24 to the top 16 bits)
void MNE::mne_addsub(BF16::bfloat16 *in0, BF16::bfloat16 *in1, BF16::bfloat16 *, BF16::bfloat16 *out, uint32_t mode, uint8_t **, uint16_t)
{
    // verified against asm @0x445500: c1 = round_to_bfloat16(1.0f - (float)(2 * (mode & 1))),
    // c2 = round_to_bfloat16(1.0f - (float)(mode & 2)), i.e. +/-1.0 (bit 0 negates in0, bit 1 negates in1).
    uint16_t c1 = bf_round(1.0f - (float)(2 * (mode & 1))), c2 = bf_round(1.0f - (float)(mode & 2));
    uint32_t p1 = fp24_round(bf_to_f(bf_mul(c1, rd16(in0))));
    uint32_t p2 = fp24_round(bf_to_f(bf_mul(c2, rd16(in1))));
    uint32_t sum = FP24::operator+(F(p2), F(p1)).bits_;
    wr16(out, (uint16_t)(sum >> 8));
}

// @0x445ca0 (MNE15)
void *MNE::MneProc(uint8_t opcode, uint32_t config)
{
    op_config_ = config;
    switch (opcode) {
    case 0: case 1:                       op_ = mne_comp;    break;
    case 2: case 3: case 4: case 5:       op_ = mne_addsub;  break;
    case 6: case 7: case 8: case 9:       op_ = mne_mul;     break;
    case 10:                              op_ = mne_div;     break;
    case 11:                              op_ = mne_round;   break;
    case 12:                              op_ = mne_sqrt;    break;
    case 13:                              op_ = mne_trangle; break;
    case 14:                              op_ = mne_logmode; break;
    case 15:                              op_ = mne_ucalc;   break;
    case 16:                              op_ = mne_exp;     break;
    case 17:                              op_ = mne_linefit; break;
    case 18:                              op_ = mne_sel;     break;
    case 19: case 20: case 21: case 22: case 23: case 24: case 25:
    case 26: case 27: case 28: case 29: case 30: case 31: case 32: case 33:
                                          op_ = mne_inout;   break;
    default:                              op_ = mne_phold;   break;
    }
    return (void *)op_;
}
