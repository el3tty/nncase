#include <nncase/functional/k230/dynamic_function.h>
#include <nncase/functional/k230/dynamic_gnne_matmul.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::functional::k230;

result<value_t> nncase::functional::k230::dynamic_gnne_matmul(gsl::span<const gsl::byte> field_span,
    const std::vector<value_t> &inputs, const kernels::kernel_context &context) {
    // program text
    try_var(text_tensor, inputs[0].as<tensor>());
    try_var(text, get_input_span(text_tensor));

    try_var(lhs, inputs[1].as<tensor>());
    try_var(rhs, inputs[2].as<tensor>());

    // output shape: copy of lhs shape, [3] = rhs.shape[3]
    dims_t out_shape = lhs->shape();
    gsl::span<const size_t> rhs_shape(rhs->shape());
    out_shape[3] = rhs_shape[3]; // gsl bounds check -> terminate when rhs rank <= 3

    datatype_t out_dtype(static_cast<typecode_t>(field_span[0]));
    value_t output;
    try_(alloc_output(output, out_dtype, out_shape));

    // forward everything but the program tensor
    gsl::span<const value_t> args(inputs);
    try_(dynamic_function_invoke_core(text, args.subspan(1), output, context));
    return ok(output);
}
