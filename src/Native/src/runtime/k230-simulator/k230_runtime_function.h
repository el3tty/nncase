#pragma once
#include "k230_runtime_module.h"
#include <nncase/runtime/runtime_function.h>

BEGIN_NS_NNCASE_RT_K230

/** Location of a parameter inside the simulated device memory (".desc" section). */
struct memory_range
{
    uint32_t start;
    uint32_t size;
};

class k230_runtime_function : public runtime_function
{
  public:
    using runtime_function::runtime_function;
    ~k230_runtime_function() override;

    k230_runtime_module &module() const noexcept;
    const memory_range &input_desc(size_t index) const noexcept;
    const memory_range &output_desc(size_t index) const noexcept;

  protected:
    result<void> initialize_core(runtime_function_init_context &context) noexcept override;
    result<value_t> invoke_core(gsl::span<value_t> parameters, value_t return_value) noexcept override;

  private:
    /** Writes the gmodel files (bin/desc/ctrl/args) of this call for offline replay. */
    int dump_gmodel(gsl::span<value_t> &parameters);

    gsl::span<const gsl::byte> text_;
    std::vector<memory_range> input_descs_;
    std::vector<memory_range> output_descs_;
    std::vector<value_t> extra_values_;
};

END_NS_NNCASE_RT_K230
