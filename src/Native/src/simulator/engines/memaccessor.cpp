#include "engines/memaccessor.h"

MemAccessor::MemAccessor(uint8_t * base)
  : base_(base) {
}

MemAccessor::MemAccessor()
  : base_(nullptr) {
}

MemAccessor::~MemAccessor() {
}

int16_t MemAccessor::MemAt(int off) const {
  return MemAt<uint16_t>(off);
}

int64_t MemAccessor::MemAt(uint32_t off) const {
  return (int64_t)(((uint32_t)base_[off + 2] << 16) | MemAt<uint16_t>((int)off));
}
