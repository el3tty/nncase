#include "engines/simulator.h"
#include "globals.h"

// The binary frees two buffers and a COW std::string; all are handled by member destructors.
Simulator::~Simulator() {
}
