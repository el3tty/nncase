#include "isa/kinstruction.h"
#include "globals.h"

void KInstruction::operation() {
}

void KInstruction::get_next_pc() {
}

KInstruction::~KInstruction() {
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// KInstruction4.cpp: duplicate of the preceding definition (C1/C2 constructor alias) omitted.
