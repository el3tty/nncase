// Reconstructed from IDA/Hex-Rays output of
//   nncase::functional::k230::dynamic_gnne_matmul(
//       gsl::span<const gsl::byte>,
//       const std::vector<nncase::object_t<nncase::value_node>> &,
//       const nncase::kernels::kernel_context &)
// (object built for riscv64, GCC, optimized, nncase K230 runtime).
//
// Nearly all of the ~1000 lines of pseudocode are inlined nncase library code
// (intrusive_ptr ref counting, result<T> plumbing, small_vector copies, the dbg()
// macro expansion inside span_reader::advance). What is left here is the
// source-level logic those inlines were generated from.
//
// Behaviour (verified against the pseudocode):
//   attrs      : 1 byte, the typecode of the output tensor
//   inputs[0]  : Tensor holding the GNNE kernel binary (mapped read-only on host)
//   inputs[1]  : lhs Tensor; its shape is the template for the output shape
//   inputs[2]  : rhs Tensor; its shape[3] becomes output shape[3]
//   inputs[1:] : forwarded to dynamic_function_invoke_core together with the
//                freshly allocated host output tensor
//   Any failed cast / null input -> std::errc::invalid_argument (22)
//   Any failed callee            -> that callee's std::error_code propagated
 
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/small_vector.hpp>
#include <nncase/runtime/span_reader.h>
#include <nncase/runtime/util.h>
#include <nncase/tensor.h>
#include <nncase/type.h>
#include <nncase/value.h>
 
#include <exception>
#include <system_error>
#include <vector>
 
namespace nncase::functional::k230 {
 
// Defined elsewhere (called, not inlined, in the binary).
// Signature deduced from the call site: (a0,a1)=span<const byte>,
// (a2,a3)=span<const value_t>, a4=tensor passed by value (copy is made and
// destroyed around the call), a5=kernel_context; result is an error_code pair
// returned in a0/a1, i.e. result<void>.
result<void> dynamic_function_invoke_core(gsl::span<const gsl::byte> func_binary,
                                          gsl::span<const value_t> args,
                                          tensor output,
                                          const kernels::kernel_context &context);
 
 
result<value_t> dynamic_gnne_matmul(gsl::span<const gsl::byte> attrs,
                                    const std::vector<value_t> &inputs,
                                    const kernels::kernel_context &context) {
    // inputs[0]: GNNE kernel binary stored as a tensor
    try_var(func_tensor, inputs[0].as<tensor>());
    try_var(func_binary, get_readonly_span(func_tensor));
 
    // inputs[1], inputs[2]: matmul operands
    try_var(lhs, inputs[1].as<tensor>());
    try_var(rhs, inputs[2].as<tensor>());
 
    // Output shape = lhs shape with dim 3 replaced by rhs dim 3.
    // (rhs dim access terminates on out-of-range; the lhs copy uses
    //  itlib::small_vector::at which asserts.)
    dims_t out_shape(lhs->shape().begin(), lhs->shape().end());
    const auto &rhs_shape = rhs->shape();
    if (rhs_shape.size() <= 3) {
        std::terminate();
    }
    out_shape.at(3) = rhs_shape[3];
 
    // Output element type comes from the first attribute byte.
    span_reader reader(attrs);
    auto out_typecode = reader.read<typecode_t>();
    datatype_t out_dtype(out_typecode);
    try_var(out_prim, out_dtype.as<prim_type_t>());
 
    // Allocate the host output tensor.
    try_var(out_rt, runtime::host_runtime_tensor::create(out_prim->typecode(), out_shape));
    tensor output = out_rt.impl();
 
    // inputs.subspan(1): terminates if inputs is empty.
    auto kernel_args = gsl::make_span(inputs).subspan(1);
    try_(dynamic_function_invoke_core(func_binary, kernel_args, output, context));
 
    return ok(value_t(output));
}
 
} // namespace nncase::functional::k230