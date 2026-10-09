#include <nncase/runtime/k230/shared_memory.h>
#include "k230_runtime_function.h"
#include "gmodel_builder.h"
#include "platform.h"
#include <cstring>
#include <iostream>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/util.h>
#include <sstream>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::k230;

BEGIN_NS_NNCASE_RT_K230

k230_runtime_function::~k230_runtime_function() = default;

k230_runtime_module &k230_runtime_function::module() const noexcept
{
    return static_cast<k230_runtime_module &>(runtime_function::module());
}

const memory_range &k230_runtime_function::input_desc(size_t index) const noexcept
{
    return input_descs_[index];
}

const memory_range &k230_runtime_function::output_desc(size_t index) const noexcept
{
    return output_descs_[index];
}

result<void> k230_runtime_function::initialize_core(runtime_function_init_context &context) noexcept
{
    // slice of the module text that belongs to this function
    auto text = module().text();
    auto &header = context.header();
    if (header.entrypoint > text.size() || header.text_size > text.size() - header.entrypoint)
        std::terminate();
    text_ = text.subspan(header.entrypoint, header.text_size);

    // ".desc": { uint64 reserved; uint32 inputs; uint32 outputs; memory_range ranges[inputs + outputs]; }
    auto read_ranges = [&](const memory_range *ranges, uint32_t inputs, uint32_t outputs) -> result<void> {
        if (parameters_size() != inputs + outputs)
            return err(std::errc::invalid_argument);
        input_descs_.insert(input_descs_.end(), ranges, ranges + inputs);
        output_descs_.insert(output_descs_.end(), ranges + inputs, ranges + inputs + outputs);
        return ok();
    };

    auto section = context.section(".desc");
    if (section.is_ok())
    {
        auto data = section.unwrap();
        auto inputs = *reinterpret_cast<const uint32_t *>(data.data() + 8);
        auto outputs = *reinterpret_cast<const uint32_t *>(data.data() + 12);
        return read_ranges(reinterpret_cast<const memory_range *>(data.data() + 16), inputs, outputs);
    }

    section_header sh;
    try_var(sr, context.seek_section(".desc", sh));
    uint8_t head[16];
    sr->read_span(gsl::span<uint8_t>(head));
    auto inputs = *reinterpret_cast<const uint32_t *>(head + 8);
    auto outputs = *reinterpret_cast<const uint32_t *>(head + 12);
    if (parameters_size() != inputs + outputs)
        return err(std::errc::invalid_argument);
    for (uint32_t i = 0; i < inputs; i++)
        input_descs_.push_back(sr->read<memory_range>());
    for (uint32_t i = 0; i < outputs; i++)
        output_descs_.push_back(sr->read<memory_range>());
    return ok();
}

int k230_runtime_function::dump_gmodel(gsl::span<value_t> &parameters)
{
    // inputs are the first parameters, the (second) parameter receives the outputs
    gsl::span<const value_t> inputs(parameters.data(), input_descs_.size());
    auto dump_manager = module().interp().dump_manager();
    std::string dump_root = dump_manager ? dump_manager->get_dump_root() : ".";

    if (parameters.size() <= 1)
        std::terminate();

    int ret;
    {
        gmodel_builder builder(text_, inputs, parameters[1], std::string(), dump_root, module().rdata(), false);
        if ((ret = builder.write_bin()) == 0 && (ret = builder.write_desc()) == 0
            && (ret = builder.write_glb_ctrl()) == 0)
        {
            builder.write_invoke_args();
            builder.write_vkpu_invoke_args();
            ret = 0;
        }
    }
    return ret;
}

result<value_t> k230_runtime_function::invoke_core(gsl::span<value_t> parameters, value_t return_value) noexcept
{
    auto mem_name = get_random_file_name();
    auto ctrl_name = get_random_file_name();
    k230::shared_memory mem(mem_name, 0x80000000ull, k230::shared_memory_openmode::create);
    k230::shared_memory ctrl(ctrl_name, 0x400000, k230::shared_memory_openmode::create);
    auto *mem_base = static_cast<char *>(mem.data());
    auto *ctrl_words = static_cast<uint32_t *>(ctrl.data());

    const size_t n_in = input_descs_.size();
    const size_t n_out = output_descs_.size();

    // 1. inputs go to the addresses chosen by the compiler (".desc")
    size_t input_bytes = 0;
    for (size_t i = 0; i < n_in; i++)
    {
        if (!parameters[i].is_a<tensor>())
            return err(std::errc::invalid_argument);
        auto t = parameters[i].as<tensor>().unwrap();
        try_var(span, get_input_span(t));
        std::memcpy(mem_base + input_desc(i).start, span.data(), span.size());
        input_bytes += span.size();
        ctrl_words[i] = input_desc(i).start;
    }

    // 2. rdata follows the inputs (8 byte aligned)
    size_t rdata_offset = (input_bytes + 7) & ~size_t(7);
    auto rdata = module().rdata();
    std::memcpy(mem_base + rdata_offset, rdata.data(), rdata.size());
    ctrl_words[n_in + n_out] = static_cast<uint32_t>(rdata_offset);

    // 3. text follows rdata (8 byte aligned)
    size_t text_offset = (rdata.size() + rdata_offset + 7) & ~size_t(7);
    std::memcpy(mem_base + text_offset, text_.data(), text_.size());

    // 4. space reserved for the (empty) dsp text, cleared
    size_t dsp_offset = text_offset + text_.size();
    std::memset(mem_base + dsp_offset, 0, module().dsp_text().size());
    ctrl_words[n_in + n_out + 1] = static_cast<uint32_t>(dsp_offset);

    // 5. outputs live behind the dsp text at their ".desc" offsets and are cleared
    size_t output_base = dsp_offset + module().dsp_text().size();
    std::vector<size_t> output_addrs;
    for (size_t i = 0; i < n_out; i++)
    {
        auto &param = parameters[n_in + i];
        if (!param.is_a<tensor>())
            return err(std::errc::not_supported);
        auto t = param.as<tensor>().unwrap();
        try_var(span, get_output_span(t));
        std::memset(mem_base + output_base, 0, span.size());
        auto addr = output_desc(i).start + output_base;
        ctrl_words[n_in + i] = static_cast<uint32_t>(addr);
        output_addrs.push_back(addr);
    }

    // 6. run the C model
    std::string dump_path = ".";
    if (auto dump_manager = module().interp().dump_manager())
        dump_path = dump_manager->dump_path();
    int ret = run_cmodel(mem_name, ctrl_name, text_offset, dump_path);
    if (ret)
    {
        std::cerr << "Failed to run k230 scmodel(" << ret << ")." << std::endl;
        return err(std::errc::io_error);
    }

    // 7. copy the results back
    for (size_t i = 0; i < n_out; i++)
    {
        auto &param = parameters[n_in + i];
        if (param.empty())
            continue;
        if (!param.is_a<tensor>())
        {
            if (param.is_a<tuple>())
                return err(std::errc::not_supported);
            continue;
        }
        auto t = param.as<tensor>().unwrap();
        try_var(span, get_output_span(t));
        std::memcpy(span.data(), mem_base + output_addrs[i], span.size());
    }

    return ok<value_t>(tuple(std::in_place));
}

END_NS_NNCASE_RT_K230
