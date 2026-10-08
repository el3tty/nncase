// mfu_const lookup tables (bf16 bit patterns), 65536 entries each, dumped verbatim from .rodata of
// nncase.simulator.k230.sc (symbols _ZN9mfu_constL*E; 0x20000 bytes each).
#pragma once
#include <cstdint>
namespace mfu_const {
extern const uint16_t ln_bf16_mod0[65536];
extern const uint16_t ln_bf16_mod1[65536];
extern const uint16_t cos_bf16[65536];
extern const uint16_t exp_bf16[65536];
extern const uint16_t sin_bf16[65536];
}  // namespace mfu_const
