#pragma once

#include "k230_common.h"
#include <pthread.h>

extern "C" int pthread_key_create(pthread_key_t*, void (*)(void*)) __attribute__((weak));

BEGIN_NS_NNCASE_RT_K230

struct mmz_segment;
struct free_heap_node;

class mmz_allocator {
public:
    mmz_allocator();
    ~mmz_allocator();

    void free(void *ptr) noexcept;
    result<void> allocate(size_t bytes, void **out_vaddr, uintptr_t *out_paddr) noexcept;
    void destroy() noexcept;
private:
    result<mmz_segment *> allocate_segment(size_t pages) noexcept;
    void merge_node(free_heap_node *prev, free_heap_node *node,
                               free_heap_node *next) noexcept;
    void insert_free_node(void *ptr, mmz_segment *segment, size_t bytes) noexcept;
    free_heap_node *destroy_node(free_heap_node *node) noexcept;
    void free_segment(mmz_segment *segment) noexcept;
    void sanity_check() noexcept;

    free_heap_node* free_list_;
    size_t avail_bytes_;
    size_t pending_bytes_;
    pthread_mutex_t mutex_;
};

END_NS_NNCASE_RT_K230
