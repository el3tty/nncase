// Lifted from IDA/Hex-Rays output (MemAccessor1..MemAccessor9).
// MemAccessor1..4 (MemAt<uint16_t>, MemAt<uint8_t>, MemAt<signed char>, MemAt<short>) are
// instantiations of the generic template defined inline in memaccessor.h.
#include "engines/memaccessor.h"

// MemAccessor5.cpp  @0x46f360
MemAccessor::MemAccessor(uint8_t * base)
  : base_(base)
{
}

// MemAccessor6.cpp  @0x46f370
MemAccessor::MemAccessor()
  : base_(nullptr)
{
}

// MemAccessor7.cpp  @0x46f380
MemAccessor::~MemAccessor()
{
}

// MemAccessor8.cpp  @0x46f390
int16_t MemAccessor::MemAt(int off) const
{
  return MemAt<uint16_t>(off);
}

// MemAccessor9.cpp  @0x46f3b0
int64_t MemAccessor::MemAt(uint32_t off) const
{
  return (int64_t)(((uint32_t)base_[off + 2] << 16) | MemAt<uint16_t>((int)off));
}
