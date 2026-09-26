#include "nncase/runtime/allocator.h"
#include "nncase/runtime/host_buffer.h"
#include "mmz_allocator.h"

using namespace nncase::runtime;

namespace {
using mmz_allocator = nncase::runtime::k230::mmz_allocator;

class host_buffer_impl: public host_buffer_node {
public:
    using host_buffer_node::host_buffer_node;

    ~host_buffer_impl();

    bool host_buffer_impl::has_physical_address() const noexcept
    result<void> unmap_core(map_access_t access);
    result<gsl::span<gsl::byte>> map_core(map_access_t access);
    result<uintptr_t> physical_address() noexcept
    result<void> sync_core(sync_op_t op);
private:
    friend class host_buffer_allocator;

    gsl::byte *vaddr_ = nullptr;                    // offset 0x58
    uintptr_t physical_address_ = 0;                 // offset 0x60
    std::function<void(gsl::byte *)> release_callback_;  // offset 0x68, 32 bytes};
};


class host_buffer_allocator {
public:
    result<buffer_t> attach(
        gsl::span<gsl::byte> data,
        const buffer_attach_options &options) noexcept override;

    result<void> shrink_memory_pool() noexcept override;
private:
    mmz_allocator mmz_allocator_;
} host_allocator;

//////////////////////////////////////////////////////////////////////////
// implementations
bool host_buffer_impl::has_physical_address() const noexcept {
    return physical_address_ != 0;
}

result<void> host_buffer_impl::unmap_core(map_access_t access) {
    return ok();
}


result<gsl::span<gsl::byte>> host_buffer_impl::map_core(map_access_t access) {
    auto *vaddr = vaddr_;
    auto size   = size_bytes();

    if (size && !vaddr) {
        std::terminate();  // size claimed but no backing pointer — corrupted state
    }
    return ok(gsl::span<gsl::byte>(vaddr, vaddr + size));
}

result<uintptr_t> host_buffer_impl::physical_address() noexcept {
    auto paddr = physical_address_;   // this+0x60
    if (!paddr) {
        return err(std::errc::invalid_argument);  // errc 22
    }
    return ok(static_cast<uintptr_t>(paddr));
}

result<void> host_buffer_impl::sync_core(sync_op_t op) {
    if (op == sync_op_t::sync_invalidate) {
        auto ret = kd_mpi_sys_mmz_flush_cache(physical_address_, vaddr_, (int)size_);
        if (ret != 0) {
            std::cerr << "invalidate failed: ret = " << ret << std::endl;
            abort();
        }
        return ok();
    }
    if (op != sync_op_t::sync_flush) {
        return ok();  // unrecognized op — silently succeeds
    }

    if (!physical_address_) {
        std::terminate();  // flush requested on a buffer with no physical mapping
    }
    auto ret = kd_mpi_sys_mmz_flush_cache(physical_address_, vaddr_, (int)size_);
    if (ret != 0) {
        std::cerr << "write back failed: ret = " << ret << std::endl;
        abort();
    }
    return ok();
}

host_buffer_impl::~host_buffer_impl() {
    auto vaddr = vaddr_;  // this+0x58

    if (release_callback_) {           // this+0x78 — std::function<void(gsl::byte*)>
        release_callback_(vaddr);
    } else {
        std::__throw_bad_function_call();
    }
    // release_callback_'s own internal teardown (std::function destructor)

    if (owned_buffer_)                  // this+0x50, count at this+0x48
        operator delete(owned_buffer_, 4 * count_);

    // ~host_buffer_node() continues from here
}

result<buffer_t> host_buffer_allocator::attach(
    gsl::span<gsl::byte> data,
    const buffer_attach_options &options) noexcept {

    uintptr_t phys_addr = options.has_physical_address ? options.physical_address : 0;

    std::function<void(gsl::byte*)> callback = options.release_callback
        ? options.release_callback
        : [](gsl::byte *) { /* default: no-op release */ };

    auto *buf = new host_buffer_impl(data.size(), *this, host_sync_status_t::valid);
    buf->vaddr_ = data.data();
    buf->physical_address_ = phys_addr;
    buf->release_callback_ = std::move(callback);

    return ok(buffer_t(buf));
}


result<void> host_buffer_allocator::shrink_memory_pool() noexcept {
    return mmz_allocator_.destroy();
}

result<buffer_t> host_buffer_allocator::allocate(
    size_t bytes, const buffer_allocate_options &options) noexcept {

    std::function<void(gsl::byte *)> deleter;
    gsl::byte *vaddr;
    uintptr_t phys_addr = 0;

    if (options.flags & HOST_BUFFER_ALLOCATE_SHARED) {
        try_(mmz_allocator_.allocate(bytes, &vaddr, &phys_addr));
        deleter = [this](gsl::byte *p) { mmz_allocator_.free(p); };
    } else {
        vaddr = (gsl::byte *)::operator new[](bytes, std::nothrow);
        if (!vaddr) {
            std::cerr << "vaddr = nullptr (void *)" << std::endl;   // dbg(), collapsed
            return err(std::errc::not_enough_memory);
        }
        deleter = [](gsl::byte *p) { ::operator delete[](p); };
    }

    auto *buf = new host_buffer_impl(bytes, *this, host_sync_status_t::valid);
    buf->vaddr_ = vaddr;
    buf->physical_address_ = phys_addr;
    buf->release_callback_ = std::move(deleter);

    return ok(buffer_t(buf));

}

}

buffer_allocator &buffer_allocator::host() { return host_allocator; }