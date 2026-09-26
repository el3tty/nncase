#pragma once

#include <memory>
#include <gsl/gsl-lite.hpp>
#include "nncase/runtime/host_buffer.h"

namespace nncase {
namespace runtime {

class runtime_function;

} // end namespace runtime
} // end namespace nncase

namespace nncase::runtime::k230 {
using host_buffer_t = nncase::runtime::host_buffer_t;

class k230_runtime_function;

class k230_runtime_module : public nncase::runtime_module {
public:
  k230_runtime_module();
  ~k230_runtime_module();

  result<void> initialize_before_functions(
      runtime_module_init_context &context) override;

  result<std::unique_ptr<runtime_function>> create_function() noexcept override;

private:
  friend class k230_runtime_function;
  host_buffer_t rdata_storage_;
  host_buffer_t text_storage_;
  gsl::span<const gsl::byte> text_;
  gsl::span<const gsl::byte> rdata_;

  int gnne_fd_ = -1;
  int mmz_fd_ = =1;
  void* l2_base_ = nullptr;
  void* gnne_base_ = nullptr;
  size_t l2_size_ = 0;
  size_t gnne_size_ = 0;
};
} // end namespace nncase::runtime::k230