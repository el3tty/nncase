#include <nncase/runtime/util.h>
#include <nncase/runtime/span_reader.h>
#include <nncase/functional/k230/dynamic_gnne_matmul.h>
#include <nncase/functional/k230/dynamic_function.h>
 
#include <vector>
 
BEGIN_NS_NNCASE_FUNCTIONAL_K230
using namespace nncase;
using namespace nncase::runtime;
 
result<value_t> dynamic_gnne_matmul(gsl::span<const gsl::byte> attrs,
                                    const std::vector<value_t> &inputs,
                                    const kernels::kernel_context &context) {
    // inputs[0]: GNNE kernel binary stored as a tensor
    try_var(func_tensor, inputs[0].as<tensor>());
    try_var(func_binary, get_input_span(func_tensor));
 
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

END_NS_NNCASE_FUNCTIONAL_K230
