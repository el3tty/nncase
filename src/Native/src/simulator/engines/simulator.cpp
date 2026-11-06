// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#include "engines/simulator.h"
#include "globals.h"

// The binary frees two buffers and a COW std::string; all are handled by member destructors.
Simulator::~Simulator() {
}
