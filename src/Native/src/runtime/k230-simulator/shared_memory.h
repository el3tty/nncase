#pragma once
#include <cstddef>
#include <filesystem>
#include <string>

namespace nncase::runtime::k230 {
enum class shared_memory_openmode {
    create = 0, // POSIX: shm_unlink + shm_open(O_CREAT|O_RDWR) + ftruncate; Windows: CreateFileMapping
    open = 1,   // POSIX: shm_open(O_RDWR);                                  Windows: OpenFileMapping
};

class shared_memory {
public:
    shared_memory(const std::filesystem::path &path, size_t size, shared_memory_openmode mode);
    ~shared_memory();

    shared_memory(const shared_memory &) = delete;
    shared_memory &operator=(const shared_memory &) = delete;

    void *data() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }

private:
#ifdef _WIN32
    void *handle_ = nullptr; // HANDLE of the file mapping
#else
    int fd_ = -1;
#endif
    std::string name_;
    void *ptr_ = nullptr;
    size_t size_;
};
} // namespace nncase::runtime::k230
