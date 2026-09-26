#pragma once

namespace nncase::runtime::k230 {

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

class mmz_allocator {
public:
    void free(void *ptr) noexcept;
    int allocate(size_t bytes, void **out_vaddr, uintptr_t *out_paddr) noexcept;
    void destroy() noexcept;
private:
    result<mmz_segment *> allocate_segment(size_t pages) noexcept;
    void merge_node(free_heap_node *prev, free_heap_node *node,
                               free_heap_node *next) noexcept;
    void insert_free_node(void *ptr, mmz_segment *segment, size_t bytes) noexcept;
    free_heap_node *destroy_node(free_heap_node *node) noexcept;
    void free_segment(mmz_segment *segment) noexcept;
    void sanity_check() noexcept;
};

} // end namespace nncase::runtime::k230