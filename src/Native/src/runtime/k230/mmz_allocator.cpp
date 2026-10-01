#include <iostream>

#include "mmz_allocator.h"
#include "mmz.h"

BEGIN_NS_NNCASE_RT_K230

bool is_threaded = (&pthread_key_create == nullptr);

struct mmz_segment {
    uintptr_t physical_address;
    size_t usable_size;
    // buffer bytes begin here, at offset 16
};

struct free_heap_node {
    mmz_segment *segment;      // offset 0
    size_t freed_bytes;         // offset 8
    free_heap_node *next;        // offset 16
};

struct auto_lock {
    pthread_mutex_t& mtx_;

    auto_lock(pthread_mutex_t& mtx): mtx_(mtx) {
        if (is_threaded)
            pthread_mutex_lock(&mtx_);
    }
    ~auto_lock() {
        if (is_threaded)
            pthread_mutex_unlock(&mtx_);
    }
};


constexpr off_t fhn_offset_to_userdata = __builtin_offsetof(free_heap_node, next);
constexpr size_t page_size_bits = 12;
constexpr size_t page_size = 1U << page_size_bits;

static_assert(sizeof(mmz_segment) == 16, "check mmz_segment size");
static_assert(fhn_offset_to_userdata == 16, "check free_heap_node");

mmz_allocator::mmz_allocator():
	free_list_(nullptr), avail_bytes_(0), pending_bytes_(0) {
    mutex_ = PTHREAD_MUTEX_INITIALIZER;
}

mmz_allocator::~mmz_allocator() {
    if (is_threaded)
        pthread_mutex_destroy(&mutex_);
}

result<mmz_segment *> mmz_allocator::allocate_segment(size_t pages) noexcept {
    uintptr_t phys_addr = 0;
    mmz_segment *seg = nullptr;
    size_t byte_size = pages << page_size_bits;

    int ret = kd_mpi_sys_mmz_alloc_cached(&phys_addr, (void**)&seg,
                                          "allocate", "anonymous",
                                          (int)byte_size);
    if (ret != 0) {
        std::cerr << "mmz_allocator.cpp"
                  << "::" << "allocate_segment" << " failed: ret = " << ret << std::endl;
        abort();
    }

    seg->physical_address = phys_addr;
    seg->usable_size = byte_size - sizeof(mmz_segment);

    return ok(seg);
}

void mmz_allocator::free_segment(mmz_segment *segment) noexcept {
    int ret = kd_mpi_sys_mmz_free(segment->physical_address, segment);
    if (ret != 0) {
        std::cerr << "free_segment" << "failed: ret = " << ret << std::endl;
        abort();
    }
}

free_heap_node *mmz_allocator::destroy_node(free_heap_node *node) noexcept {
    if (!node)
        return nullptr;

    node->next = destroy_node(node->next);

    if (node->freed_bytes == node->segment->usable_size) {
        pending_bytes_ -= node->freed_bytes;   // atomic_fetch_add(this, -freed_bytes)
        auto *seg = node->segment;
        node = node->next;
        free_segment(seg);
    }

    return node;
}

void mmz_allocator::sanity_check() noexcept {
    size_t avail = 0;
    for (auto *node = free_list_; node; node = node->next) {
        avail += node->freed_bytes;
    }

    assert(avail_bytes_ == avail);
}

void mmz_allocator::destroy() noexcept {
    sanity_check();
    free_list_ = destroy_node(free_list_);
}

void mmz_allocator::merge_node(free_heap_node *prev, free_heap_node *node,
                               free_heap_node *next) noexcept {
    if (reinterpret_cast<uintptr_t>(next) ==
        reinterpret_cast<uintptr_t>(node) + node->freed_bytes) {
        node->freed_bytes += next->freed_bytes;
        node->next = next->next;
    }

    if (prev && reinterpret_cast<uintptr_t>(node) ==
                reinterpret_cast<uintptr_t>(prev) + prev->freed_bytes) {
        prev->freed_bytes += node->freed_bytes;
        prev->next = node->next;
    }

    //try_free_segment();
    sanity_check();
}

void mmz_allocator::insert_free_node(void *ptr, mmz_segment *segment, size_t bytes) noexcept {
    assert(bytes >= sizeof(free_heap_node));

    sanity_check();
    avail_bytes_ += bytes;

    auto *node = static_cast<free_heap_node *>(ptr);
    node->segment = segment;
    node->freed_bytes = bytes;

    // walk the address-sorted free list to find the insertion point
    free_heap_node *prev = nullptr;
    free_heap_node *cur = free_list_;
    while (cur && cur < node) {
        prev = cur;
        cur = cur->next;
    }
    node->next = cur;

    if (prev) {
        prev->next = node;
    } else {
        free_list_ = node;
    }

    sanity_check();
    merge_node(prev, node, cur);
}

void mmz_allocator::free(void *ptr) noexcept {
    if (!ptr)
        return;
    auto_lock lock(mutex_);

    auto *header = reinterpret_cast<uint8_t *>(ptr) - fhn_offset_to_userdata;
    auto *node = reinterpret_cast<free_heap_node*>(header);
    insert_free_node(node, node->segment, node->freed_bytes);
}

result<void> mmz_allocator::allocate(size_t bytes, void *&out_vaddr,
                                     uintptr_t &out_paddr) noexcept {

    auto_lock lock(mutex_);

    if (bytes == 0) {
        out_vaddr = nullptr;
        out_paddr = 0;
        return ok();
    }

    constexpr size_t min_align = 16;
    size_t aligned = (bytes & (min_align - 1)) ? bytes + min_align - (bytes & (min_align - 1)) : bytes;
    size_t needed = aligned + fhn_offset_to_userdata;   // + this allocation's own embedded header
    free_heap_node *chunk = nullptr;

    bool found = false;
    if (avail_bytes_ >= bytes) {
        sanity_check();

        free_heap_node *prev = nullptr;
        for (auto *node = free_list_; node; prev = node, node = node->next) {
            if (node->freed_bytes >= needed + 24) {
                // big enough to split: carve the tail off as the allocation,
                // shrink the node in place to cover the remaining (front) part
                size_t remaining = node->freed_bytes - needed;
                node->freed_bytes = remaining;
                auto *tail = reinterpret_cast<free_heap_node *>(
                    reinterpret_cast<uint8_t *>(node) + remaining);
                tail->freed_bytes = needed;
                tail->segment = node->segment;
                chunk = tail;
                avail_bytes_ -= needed;
                found = true;
                break;
            }
            if (node->freed_bytes >= needed) {
                // not enough leftover to split � hand out the whole node,
                // absorbing the small excess rather than fragmenting
                if (prev) prev->next = node->next; else free_list_ = node->next;
                avail_bytes_ -= node->freed_bytes;
                chunk = node;
                found = true;
                break;
            }
        }
    }

    if (!found) {
        // nothing suitable free � allocate a whole new segment
        constexpr size_t alloc_header_size = sizeof(mmz_segment) + fhn_offset_to_userdata;
        // minimum allocation is 1 byte in size.
        auto seg_result = allocate_segment(
            (needed + page_size + alloc_header_size - 1) >> page_size_bits
        );
        if (seg_result.is_err()) {
            return err(seg_result.unwrap_err());  // in practice unreachable: allocate_segment aborts
        }
        auto *segment = seg_result.unwrap();

        auto *node = reinterpret_cast<free_heap_node *>(
            reinterpret_cast<uint8_t *>(segment) + sizeof(mmz_segment));
        node->segment = segment;

        if (segment->usable_size - needed > sizeof(free_heap_node)) {
            insert_free_node(reinterpret_cast<uint8_t *>(segment)
                             + sizeof(mmz_segment) +
                             + fhn_offset_to_userdata
                             + aligned,
                             segment, segment->usable_size - needed);
            node->freed_bytes = needed;
        } else {
            node->freed_bytes = segment->usable_size;  // absorb tiny leftover
        }
        chunk = node;
    }

    out_vaddr = reinterpret_cast<uint8_t *>(chunk) + fhn_offset_to_userdata;
    out_paddr =
        reinterpret_cast<uintptr_t>(chunk) + chunk->segment->physical_address -
        reinterpret_cast<uintptr_t>(chunk->segment) + fhn_offset_to_userdata;

    return ok();
}

END_NS_NNCASE_RT_K230
