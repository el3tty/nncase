#include "gmodel_builder.h"
#include "platform.h"
#include "shared_memory.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <nncase/functional/k230/dynamic_function.h>
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/dump_manager.h>
#include <nncase/runtime/util.h>
#include <sstream>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k230;

namespace
{
/** Control block layout of the C model: the gnne arguments are stored as a
 *  downward growing stack of 32 bit words ending at ctrl + 0x40000.  The
 *  argument list is reversed (last argument first); within an argument the
 *  first element sits at the highest address. */
void write_ctrl_stack(void *ctrl, const std::vector<std::vector<int>> &args)
{
    auto *base = static_cast<char *>(ctrl);
    *reinterpret_cast<uint64_t *>(base + 4) = 0;

    int64_t index = -1;
    for (auto it = args.rbegin(); it != args.rend(); ++it)
    {
        for (int word : *it)
        {
            *reinterpret_cast<int *>(base + 0x40000 + 4 * index) = word;
            --index;
        }
    }
}
} // namespace

result<void> dynamic_function_invoke_cmodel(gsl::span<const gsl::byte> text, gsl::span<const value_t> inputs,
    value_t output, const kernels::kernel_context &context)
{
    auto mem_name = get_random_file_name();
    auto ctrl_name = get_random_file_name();
    shared_memory mem(mem_name, 0x80000000ull, shared_memory_openmode::create);
    shared_memory ctrl(ctrl_name, 0x400000, shared_memory_openmode::create);
    auto *mem_base = static_cast<char *>(mem.data());

    std::vector<std::vector<int>> args;
    std::vector<size_t> output_offsets;
    size_t offset = 0;

    // inputs are copied back to back, their addresses become gnne arguments
    for (auto &in : inputs)
    {
        if (!in.is_a<tensor>())
            return err(std::errc::invalid_argument);
        auto t = in.as<tensor>().unwrap();
        try_var(span, get_input_span(t));
        std::memcpy(mem_base + offset, span.data(), span.size());
        try_var(arg, to_gnne_arg(t, offset));
        args.push_back(std::move(arg));
        offset += span.size();
    }

    // text follows the inputs, aligned to 8 bytes
    offset = (offset + 7) & ~size_t(7);
    std::memcpy(mem_base + offset, text.data(), text.size());
    size_t text_offset = offset;
    offset += text.size();

    // outputs are cleared and get their own addresses
    auto add_output = [&](const value_t &v) -> result<void> {
        if (!v.is_a<tensor>())
            return err(std::errc::invalid_argument);
        auto t = v.as<tensor>().unwrap();
        try_var(span, get_input_span(t));
        std::memset(mem_base + offset, 0, span.size());
        try_var(arg, to_gnne_arg(t, offset));
        args.push_back(std::move(arg));
        output_offsets.push_back(offset);
        offset += span.size();
        return ok();
    };
    if (output.is_a<tensor>())
    {
        try_(add_output(output));
    }
    else if (output.is_a<tuple>())
    {
        auto tup = output.as<tuple>().unwrap();
        for (auto &f : tup->fields())
        {
            try_(add_output(f));
        }
    }
    else
    {
        return err(std::errc::invalid_argument);
    }

    write_ctrl_stack(ctrl.data(), args);

    int ret = run_cmodel(mem_name, ctrl_name, text_offset,
        context.dump_manager ? std::string(context.dump_manager->dump_path()) : std::string("."));
    if (ret)
    {
        std::cerr << "Failed to run k230 scmodel(" << ret << ")." << std::endl;
        return err(std::errc::io_error);
    }

    // copy the results back
    auto read_output = [&](const value_t &v, size_t from) -> result<void> {
        auto t = v.as<tensor>().unwrap();
        try_var(span, get_output_span(t));
        std::memcpy(span.data(), mem_base + from, span.size());
        return ok();
    };
    if (output.is_a<tensor>())
    {
        try_(read_output(output, output_offsets[0]));
    }
    else
    {
        auto tup = output.as<tuple>().unwrap();
        size_t i = 0;
        for (auto &f : tup->fields())
        {
            try_(read_output(f, output_offsets[i++]));
        }
    }
    return ok();
}

result<void> dynamic_function_invoke_scmodel(gsl::span<const gsl::byte> text, gsl::span<const value_t> inputs,
    value_t output, const std::string &scmodel_path, const kernels::kernel_context &context)
{
    std::string dump_root;
    if (context.dump_manager)
        dump_root = context.dump_manager->get_dump_root();

    int ret;
    {
        // the builder owns "<scmodel_path>/gmodel_dump_dir/<n>/"
        gmodel_builder builder(text, inputs, output, scmodel_path, dump_root, {}, true);
        if ((ret = builder.write_bin()) == 0 && (ret = builder.write_desc()) == 0
            && (ret = builder.write_glb_ctrl()) == 0)
        {
            builder.write_invoke_args();
            builder.write_vkpu_invoke_args();
            if ((ret = builder.run_gmodel()) == 0)
                ret = builder.read_output();
        }
    } // ~gmodel_builder removes the dump directory of the run before the previous one

    if (ret)
        return err(static_cast<std::errc>(ret));
    return ok();
}

BEGIN_NS_NNCASE_FUNCTIONAL_K230

result<void> dynamic_function_invoke_core(gsl::span<const gsl::byte> text, gsl::span<const value_t> inputs,
    value_t output, const kernels::kernel_context &context) noexcept
{
    // K230_SCMODEL_PATH selects the RTL based simulator, otherwise the C model is used
    if (auto path = getenv("K230_SCMODEL_PATH"))
        return dynamic_function_invoke_scmodel(text, inputs, output, std::string(path), context);
    return dynamic_function_invoke_cmodel(text, inputs, output, context);
}

END_NS_NNCASE_FUNCTIONAL_K230
