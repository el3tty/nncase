#include <nncase/runtime/k230/shared_memory.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace nncase::runtime::k230;

#ifdef _WIN32

namespace {
std::string win_error(const char *what) {
    return std::string(what) + " (error " + std::to_string(GetLastError()) + ")";
}
} // namespace

shared_memory::shared_memory(const std::filesystem::path &path, size_t size, shared_memory_openmode mode)
    : handle_(nullptr), name_(path.string()), ptr_(nullptr), size_(size) {
    if (mode == shared_memory_openmode::create) {
        const uint64_t sz = size;
        handle_ = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (DWORD)(sz >> 32),
            (DWORD)(sz & 0xFFFFFFFFu), name_.c_str());
        if (!handle_)
            throw std::runtime_error(win_error("Failed to open shared memory when create : CreateFileMapping"));
        if (GetLastError() == ERROR_ALREADY_EXISTS) { // POSIX unlinks a stale object first, here we cannot
            CloseHandle(handle_);
            handle_ = nullptr;
            throw std::runtime_error("Failed to open shared memory when create : name already in use");
        }
    } else if (mode == shared_memory_openmode::open) {
        handle_ = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name_.c_str());
        if (!handle_)
            throw std::runtime_error("Failed to open shared memory.");
    } else {
        throw std::runtime_error("Invalid shared memory openmode.");
    }

    void *p = MapViewOfFile(static_cast<HANDLE>(handle_), FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!p) {
        CloseHandle(static_cast<HANDLE>(handle_));
        handle_ = nullptr;
        throw std::runtime_error("Failed to map shared memory.");
    }
    ptr_ = p;
}

shared_memory::~shared_memory() {
    if (ptr_)
        UnmapViewOfFile(ptr_);
    if (handle_)
        CloseHandle(static_cast<HANDLE>(handle_)); // the object dies with its last handle
}

#else

shared_memory::shared_memory(const std::filesystem::path &path, size_t size, shared_memory_openmode mode)
    : fd_(-1), name_(path.string()), ptr_(nullptr), size_(size) {
    if (mode == shared_memory_openmode::create) {
        // Remove a stale object first; a missing one (ENOENT) is fine.
        if (shm_unlink(name_.c_str()) < 0 && errno != ENOENT)
            throw std::runtime_error(std::string("Failed to unlink shared memory when create : ") + strerror(errno));

        fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0755);
        if (fd_ < 0)
            throw std::runtime_error(std::string("Failed to open shared memory when create : ") + strerror(errno));

        if (ftruncate(fd_, size))
            throw std::runtime_error(std::string("Failed to ftruncate shared memory when create : ") + strerror(errno));
    } else if (mode == shared_memory_openmode::open) {
        fd_ = shm_open(name_.c_str(), O_RDWR, 0755);
        if (fd_ < 0)
            throw std::runtime_error("Failed to open shared memory.");
    } else {
        throw std::runtime_error("Invalid shared memory openmode.");
    }

    void *p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED)
        throw std::runtime_error("Failed to map shared memory.");
    ptr_ = p;
}

shared_memory::~shared_memory() {
    if (ptr_)
        munmap(ptr_, size_);
    close(fd_);
    shm_unlink(name_.c_str());
}

#endif
