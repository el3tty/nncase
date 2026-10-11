#include <poll.h>
#include <vector>
#include <nncase/tensor.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/k230/gnne.h>

#include "runtime_function.h"
#include "runtime_module_impl.h"

#ifdef DUMP_K230_INVOCATIONS
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#endif

using namespace nncase;

namespace {
using host_buffer_slice = nncase::runtime::host_buffer_slice;
using map_access_t      = nncase::runtime::map_access_t;
using sync_op_t         = nncase::runtime::sync_op_t;   // was missing
namespace hrt           = nncase::runtime::host_runtime_tensor;

result<tensor> alloc_new_device_tensor(tensor old_tensor,
                                       host_buffer_slice slice) noexcept {
    dims_t shape{slice.size_bytes()};
    try_var(new_rt, hrt::create(nncase::dt_uint8, shape));   // runtime_tensor
    tensor new_tensor = new_rt.impl();

    {   // scope the mappings so they are unmapped before the sync below
        try_var(old_host, slice.map(map_access_t::map_read));
        try_var(new_slice, new_tensor->buffer().as_host());
        try_var(new_host, new_slice.map(map_access_t::map_write));

        std::memcpy(new_host.buffer().data(), old_host.buffer().data(),
                    old_host.buffer().size());
    }

    try_(hrt::sync(new_rt, sync_op_t::sync_write_back, true));

    return ok(new_tensor);
}

#ifdef DUMP_K230_INVOCATIONS
// Test-case capture (enabled with -DDUMP_K230_INVOCATIONS, CMake option DUMP_K230_INVOCATIONS).
// Files are written to the current working directory:
//   rdata.bin       module .rdata (written by k230_runtime_module::initialize_before_functions)
//   text_<EP>.bin   code of the function with entrypoint <EP> (hexadecimal, no padding)
//   in_<EP>.bin     input tensors of the latest invocation, concatenated in parameter order
//   out_<EP>.bin    output tensors of the latest invocation, concatenated in parameter order
std::string dump_name(const char *prefix, size_t entrypoint) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_%zx.bin", prefix, entrypoint);
    return buf;
}

// Writes bytes [offset, offset + size) of a host buffer. The module's text_ / rdata_ spans hold physical addresses,
// so the contents are read through a mapping of the buffer.
result<void> dump_range(const std::string &path,
                        nncase::runtime::host_buffer_t buffer, size_t offset,
                        size_t size) noexcept {
    try_var(mapped, buffer->map(map_access_t::map_read));
    auto bytes = mapped.buffer();
    if (offset > bytes.size() || size > bytes.size() - offset)
        return err(std::errc::invalid_argument);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(bytes.data() + offset),
              static_cast<std::streamsize>(size));
    return out ? ok() : err(std::errc::io_error);
}

result<void> dump_tensors(const std::string &path, gsl::span<value_t> params,
                          size_t first, size_t count,
                          bool invalidate) noexcept {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        return err(std::errc::io_error);
    for (size_t i = first; i < first + count; i++) {
        try_var(t, params[i].as<tensor>());
        try_var(host, t->buffer().as_host());
        // Outputs are written by the device: drop stale cache lines first.
        if (invalidate)
            try_(host.sync(sync_op_t::sync_invalidate, true));
        try_var(mapped, host.map(map_access_t::map_read));
        auto bytes = mapped.buffer();
        out.write(reinterpret_cast<const char *>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    return out ? ok() : err(std::errc::io_error);
}
#endif
} // end anonymous namespace

BEGIN_NS_NNCASE_RT_K230

memory_range &k230_runtime_function::input_desc(size_t index) noexcept {
    return input_descs_[index];
}

memory_range &k230_runtime_function::output_desc(size_t index) noexcept {
    return output_descs_[index];
}


result<void> k230_runtime_function::initialize_core(
    runtime_function_init_context &context) noexcept {

    auto &k230_mod = static_cast<k230_runtime_module &>(module());
    auto mod_text = k230_mod.text_;

    auto entrypoint = context.header().entrypoint;
    auto text_size  = context.header().text_size;

    if (mod_text.size() < entrypoint ||
        mod_text.size() - entrypoint < text_size) {
        std::terminate();
    }
    text_begin_ = mod_text.data() + entrypoint;
    text_end_   = mod_text.data() + entrypoint + text_size;

#ifdef DUMP_K230_INVOCATIONS
    if (dump_range(dump_name("text", entrypoint), k230_mod.text_storage_,
                   entrypoint, text_size).is_err())
        std::cerr << "DUMP_K230_INVOCATIONS: cannot dump text" << std::endl;
#endif

    return context.read_section(
        ".desc", [&](auto &reader, size_t) -> result<void> {
            auto header = reader.template read<desc_header>();
            if (header.inputs + header.outputs != (int32_t)parameters_size()) {
                return err(std::errc::invalid_argument);
            }
            for (uint32_t i = 0; i < header.inputs; i++)
                input_descs_.push_back(reader.template read<memory_range>());
            for (uint32_t i = 0; i < header.outputs; i++)
                output_descs_.push_back(reader.template read<memory_range>());

            return ok();
        });
}

result<value_t> k230_runtime_function::invoke_core(
    gsl::span<value_t> parameters, value_t return_value) noexcept {

    auto &mod = static_cast<k230_runtime_module &>(module());
    gnne_set_base(mod.gnne_base_); // MMIO register base � NOT l2_base_
                                   // (corrected earlier)
    gnne_init();

    std::vector<uint32_t> input_addrs, output_addrs;

    // --- Stage 1: marshal inputs ---
    for (size_t i = 0; i < input_descs_.size(); i++) {
        auto &param = parameters[i];
        if (!param.is_a<tensor>())
            return err(std::errc::invalid_argument);

        try_var(t, param.as<tensor>());
        try_var(host, t->buffer().as_host());

        if (!host.has_physical_address()) {
            try_var(t2, alloc_new_device_tensor(t, host));
            try_var(host2, t2->buffer().as_host());
            host = host2;
        }
        assert(host.has_physical_address());
        try_(host.sync(sync_op_t::sync_write_back, false));
        try_var(phys, host.physical_address());
        input_addrs.push_back((uint32_t)phys);
    }

    // --- Stage 2: marshal outputs ---
    for (size_t i = 0; i < output_descs_.size(); i++) {
        auto &param = parameters[input_descs_.size() + i];
        try_var(t, param.as<tensor>());
        try_var(host, t->buffer().as_host());
        try_(host.sync(sync_op_t::sync_invalidate, true));
        try_var(phys, host.physical_address());
        output_addrs.push_back((uint32_t)phys);
    }

    auto *l2_mem = reinterpret_cast<uint32_t *>(mod.l2_base_);

    std::copy(input_addrs.begin(), input_addrs.end(), l2_mem);
    std::copy(output_addrs.begin(), output_addrs.end(), l2_mem + input_addrs.size());

    size_t next = input_addrs.size() + output_addrs.size();

    //try_var(rdata_host, mod.rdata_storage_);
    assert(mod.rdata_storage_->has_physical_address());

    try_var(rdata_phys, mod.rdata_storage_->physical_address());
    l2_mem[next] = (uint32_t)rdata_phys;

    l2_mem[next + 1] = 0;

#ifdef DUMP_K230_INVOCATIONS
    const size_t dump_ep = static_cast<size_t>(text_begin_ - mod.text_.data());
    if (dump_tensors(dump_name("in", dump_ep), parameters, 0,
                     input_descs_.size(), false).is_err())
        std::cerr << "DUMP_K230_INVOCATIONS: cannot dump inputs" << std::endl;
#endif

    // --- Stage 4: dispatch ---
    gnne_set_time_out(200000);

    if (gnne_enable((uint64_t)text_begin_, (uint64_t)text_end_, 0) != 0) {
        // dbg()-logged failure � runtime_function.device.cpp:324
        return err(std::errc::device_or_resource_busy);
    }

    pollfd fds{};
    fds.fd = mod.gnne_fd_;
    fds.events = POLLIN;
    poll(&fds, 1, -1);

#ifdef DUMP_K230_INVOCATIONS
    if (dump_tensors(dump_name("out", dump_ep), parameters,
                     input_descs_.size(), output_descs_.size(), true).is_err())
        std::cerr << "DUMP_K230_INVOCATIONS: cannot dump outputs" << std::endl;
#endif

    return ok(new tuple_node());
}

END_NS_NNCASE_RT_K230
