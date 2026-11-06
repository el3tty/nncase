// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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
