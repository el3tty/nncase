#include <poll.h>
#include <vector>
#include <nncase/tensor.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/k230/gnne.h>

#include "runtime_function.h"
#include "runtime_module_impl.h"

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
    // outputs without a physical address (plain host buffers) get a device
    // buffer for the KPU, copied back into the original after the run
    std::vector<std::pair<host_buffer_slice, tensor>> copy_back_outputs;
    for (size_t i = 0; i < output_descs_.size(); i++) {
        auto &param = parameters[input_descs_.size() + i];
        try_var(t, param.as<tensor>());
        try_var(host, t->buffer().as_host());

        if (!host.has_physical_address()) {
            try_var(t2, alloc_new_device_tensor(t, host));
            try_var(host2, t2->buffer().as_host());
            copy_back_outputs.emplace_back(host, t2);
            host = host2;
        }
        try_(host.sync(sync_op_t::sync_invalidate, true));
        try_var(phys, host.physical_address());
        output_addrs.push_back((uint32_t)phys);
    }

    // l2_base_ is an uncached /dev/mmz mapping: store word by word, a memcpy
    // there uses misaligned 8-byte stores and faults
    auto *l2_mem = reinterpret_cast<volatile uint32_t *>(mod.l2_base_);
    size_t next = 0;
    if ((input_addrs.size() + output_addrs.size() + 2) * sizeof(uint32_t) >
        mod.l2_size_)
        return err(std::errc::argument_list_too_long);

    for (auto addr : input_addrs)
        l2_mem[next++] = addr;
    for (auto addr : output_addrs)
        l2_mem[next++] = addr;

    // a model without weights has an empty .rdata: mmz hands out no physical
    // address for 0 bytes, and the KPU program never reads it, so pass 0
    uint32_t rdata_addr = 0;
    if (!mod.rdata_.empty()) {
        if (!mod.rdata_storage_->has_physical_address())
            return err(std::errc::invalid_argument);
        try_var(rdata_phys, mod.rdata_storage_->physical_address());
        rdata_addr = (uint32_t)rdata_phys;
    }
    l2_mem[next] = rdata_addr;

    l2_mem[next + 1] = 0;

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

    // --- Stage 5: copy outputs back from their device buffers ---
    for (auto &[orig, dev_tensor] : copy_back_outputs) {
        try_var(dev, dev_tensor->buffer().as_host());
        try_(dev.sync(sync_op_t::sync_invalidate, true));
        try_var(dev_map, dev.map(map_access_t::map_read));
        try_var(orig_map, orig.map(map_access_t::map_write));
        std::memcpy(orig_map.buffer().data(), dev_map.buffer().data(),
                    orig_map.buffer().size());
    }

    return ok(value_t(tuple(std::in_place)));  // not ok(new tuple_node()): that adds a second reference, never freed
}

END_NS_NNCASE_RT_K230
