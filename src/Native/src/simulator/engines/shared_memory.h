#pragma once
// POSIX shared-memory (/dev/shm) mapping used by the K230 runtime (invoke_core).
// Lifted from IDA/Hex-Rays output (shared_memory1 = destructor, shared_memory2 = constructor).
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace nncase::runtime::k230 {

class shared_memory {
public:
    // Values of the `openmode` constructor argument.
    static constexpr int create = 0;   // unlink any old object, shm_open(O_CREAT|O_RDWR), ftruncate(size)
    static constexpr int open = 1;     // shm_open(O_RDWR) on an existing object

    // @0x41c1a0 (shared_memory2).  Throws std::runtime_error on failure.
    shared_memory(const std::filesystem::path & path, size_t size, int openmode);
    // @0x41c120 (shared_memory1): munmap, close and shm_unlink the object.
    virtual ~shared_memory();

    shared_memory(const shared_memory &) = delete;
    shared_memory & operator=(const shared_memory &) = delete;

    // Base address of the mapping.
    uint8_t * data() const { return static_cast<uint8_t *>(data_); }
    size_t size() const { return size_; }

private:
    int fd_ = -1;                // +0
    std::string name_;           // +8  : object name passed to shm_open / shm_unlink
    void * data_ = nullptr;      // +16 : mmap result
    size_t size_ = 0;            // +24
};

} // namespace nncase::runtime::k230
