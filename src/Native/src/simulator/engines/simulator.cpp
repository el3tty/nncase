// Lifted from IDA/Hex-Rays output (Simulator1).
#include "engines/simulator.h"
#include "globals.h"

// Simulator1.cpp  @0x410a70
// The decompilation shows: free the buffer at +504 (capacity f520), free the buffer at +160
// (capacity f176) and drop the COW std::string at +144.  All three are handled by member destructors.
Simulator::~Simulator()
{
}
