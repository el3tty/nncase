// Lifted from IDA/Hex-Rays output (shared_memory1, shared_memory2).
#include "engines/shared_memory.h"
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nncase::runtime::k230 {

// shared_memory2.cpp  @0x41c1a0
shared_memory::shared_memory(const std::filesystem::path & path, size_t size, int openmode)
  : fd_(-1), name_(path.string()), data_(nullptr), size_(size)
{
  if (openmode == create) {
    // Remove any stale object first; "does not exist" is fine.
    if (shm_unlink(name_.c_str()) < 0 && errno != ENOENT)
      throw std::runtime_error(std::string("Failed to unlink shared memory when create : ") + strerror(errno));
    fd_ = shm_open(name_.c_str(), O_RDWR | O_CREAT, 0755);       // flags 66, mode 0x1ED
    if (fd_ < 0)
      throw std::runtime_error(std::string("Failed to open shared memory when create : ") + strerror(errno));
    // verified against asm @0x406ed6 (.cold): failure message is
    // "Failed to ftruncate shared memory when create : " + strerror(errno) + " " + to_string(size_) + " " + to_string(fd_)
    if (ftruncate(fd_, (off_t)size) != 0)
      throw std::runtime_error(std::string("Failed to ftruncate shared memory when create : ") + strerror(errno) + " " + std::to_string(size_) + " " + std::to_string(fd_));
  } else if (openmode == open) {
    fd_ = shm_open(name_.c_str(), O_RDWR, 0755);                  // flags 2
    if (fd_ < 0)
      throw std::runtime_error("Failed to open shared memory.");
  } else {
    throw std::runtime_error("Invalid shared memory openmode.");
  }

  data_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (data_ == MAP_FAILED) {
    data_ = nullptr;
    throw std::runtime_error("Failed to map shared memory.");
  }
}

// shared_memory1.cpp  @0x41c120
shared_memory::~shared_memory()
{
  if (data_)
    munmap(data_, size_);
  close(fd_);
  shm_unlink(name_.c_str());
}

} // namespace nncase::runtime::k230
