#pragma once
// Test description: the tensors and options given in the comments of an assembly file (see README.md).
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace k230::isa_test {

struct TestError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class ElemType { I8, U8, I16, U16, I32, U32, F32, F16 };

struct TypeInfo {
    const char *name;
    ElemType type;
    size_t size;
};

struct Tensor {
    const TypeInfo *type;
    size_t count;
    std::vector<uint8_t> bytes;
    int line;
    int64_t addr = -1; // explicit GLB address or -1 = automatic layout
};

struct TestCase {
    std::string path;
    std::string source;              // assembly text
    std::vector<Tensor> inputs;      // GLB, copied to the GLB before the run
    std::vector<Tensor> outputs;     // GLB, compared after the run
    std::vector<Tensor> ddr_inputs;  // DDR_IN region
    std::vector<Tensor> ddr_outputs; // DDR_OUT region
    bool expect_fail = false;
};

// float <-> IEEE half (round to nearest even)
uint16_t float_to_half(float f);
float half_to_float(uint16_t h);

// Parses the test description from the comments of `source`; throws TestError.
TestCase parse_test_case(const std::string &source);

// "1234 (0x04d2)" style text of the element at `p`
std::string format_element(const TypeInfo &t, const uint8_t *p);

} // namespace k230::isa_test
