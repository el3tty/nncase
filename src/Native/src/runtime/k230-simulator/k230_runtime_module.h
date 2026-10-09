#pragma once
#define NNCASE_MODULES_K230_DLL

#include <nncase/runtime/k230/runtime_module.h>
#include <nncase/runtime/runtime_module.h>

BEGIN_NS_NNCASE_RT_K230

class k230_runtime_module : public runtime_module
{
  public:
    gsl::span<const gsl::byte> rdata() const noexcept { return rdata_; }
    gsl::span<const gsl::byte> dsp_text() const noexcept { return dsp_text_; }
    gsl::span<const gsl::byte> text() const noexcept { return text_; }

  protected:
    result<void> initialize_before_functions(runtime_module_init_context &context) noexcept override;
    result<std::unique_ptr<runtime_function>> create_function() noexcept override;

  private:
    gsl::span<const gsl::byte> rdata_;
    gsl::span<const gsl::byte> dsp_text_;
    gsl::span<const gsl::byte> text_;
    host_buffer_t rdata_storage_;
    host_buffer_t dsp_text_storage_;
    host_buffer_t text_storage_;
};

END_NS_NNCASE_RT_K230
