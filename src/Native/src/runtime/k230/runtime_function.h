#pragma once

#include "nncase/runtime/runtime_function.h"

namespace nncase {
namespace runtime {

class runtime_module;

} // end namespace runtime
} // end namespace nncase

namespace nncase::runtime::k230 {

struct memory_range {
    uint32_t start;
    uint32_t size;   // or `length` — field name not confirmed, only its 4-byte size and position
};

class k230_runtime_function: public nncase::runtime::runtime_function {
public:
    k230_runtime_function(nncase::runtime::runtime_module& module):
        nncase::runtime::runtime_function(module) {}

    memory_range &input_desc(size_t index) noexcept;
    memory_range &output_desc(size_t index) noexcept;

    nncase::result<void> initialize_core(
        runtime_function_init_context &context) noexcept override;

    nncase::result<value_t> invoke_core(
        gsl::span<value_t> parameters, 
        value_t return_value) noexcept override;
private:
    const gsl::byte* text_begin_;
    const gsl::byte* text_end_;
    std::vector<memory_range> input_descs_;
    std::vector<memory_range> output_descs_;
};
} // end namespace nncase::runtime::k230

