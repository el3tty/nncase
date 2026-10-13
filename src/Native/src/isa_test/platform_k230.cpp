// run_test() for a K230 board. Runs on the board (RT-Smart / Linux) as root.
//
// The test app talks to the hardware the same way the nncase runtime module (runtime/k230) does, but does not use the
// runtime itself:
//
//   * /dev/mmz       maps the 4 MiB L2 memory (physical 0x80000000) = GLB bank 0 and the GNNE register file
//                    (0x80400000); the DDR tensors and the program live in buffers allocated with libmmz
//   * /dev/k230-gnne signals the end of a run (poll)
//
// Buffers are allocated cached (kd_mpi_sys_mmz_alloc_cached, like the runtime does) and freed again at the end of the
// test. They are flushed (kd_mpi_sys_mmz_flush_cache, write back + invalidate) after the CPU has filled them and
// before the CPU reads the outputs. The L2 memory is mapped through /dev/mmz and accessed directly. The program is started with gnne_enable(), which is the register level driver of the runtime
// (runtime/k230/gnne.c), linked in unchanged.
#include "test_runner.h"

#include <nncase/runtime/k230/gnne.h>

#include "mmz.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <poll.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace k230::isa_test {

namespace {

constexpr int kTimeoutMs = 10000; // a test is a handful of instructions
constexpr uint32_t kGnneTimeout = 200000; // value of the GNNE watchdog register (same as the runtime)
constexpr size_t kPage = 0x1000;

struct PlatformError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::string errno_text(const char *what) { return std::string(what) + ": " + std::strerror(errno); }

// Physically contiguous, uncached buffer.
class MmzBuffer {
public:
    MmzBuffer(size_t bytes, const char *name) : size_(align_to(bytes ? bytes : 1, kPage)) {
        const int ret = kd_mpi_sys_mmz_alloc_cached(&phys_, &virt_, name, "anonymous", static_cast<unsigned int>(size_));
        if (ret != 0 || !virt_)
            throw PlatformError(std::string("kd_mpi_sys_mmz_alloc_cached(") + name + ", " + std::to_string(size_) + ") failed: " + std::to_string(ret));
    }
    ~MmzBuffer() {
        if (virt_)
            kd_mpi_sys_mmz_free(phys_, virt_);
    }
    MmzBuffer(const MmzBuffer &) = delete;
    MmzBuffer &operator=(const MmzBuffer &) = delete;

    // Write back the CPU's changes and drop the cached copy, so the NPU sees the data and the CPU sees the NPU's.
    void flush() const {
        const int ret = kd_mpi_sys_mmz_flush_cache(phys_, virt_, static_cast<unsigned int>(size_));
        if (ret != 0)
            throw PlatformError("kd_mpi_sys_mmz_flush_cache failed: " + std::to_string(ret));
    }

    uint8_t *data() const { return static_cast<uint8_t *>(virt_); }
    uint64_t phys() const { return phys_; }
    size_t size() const { return size_; }

private:
    unsigned long phys_ = 0;
    void *virt_ = nullptr;
    size_t size_;
};

class Device {
public:
    Device() {
        gnne_fd_ = ::open("/dev/k230-gnne", O_RDWR);
        if (gnne_fd_ < 0)
            throw PlatformError(errno_text("open /dev/k230-gnne"));
        mmz_fd_ = ::open("/dev/mmz", O_RDWR | O_SYNC);
        if (mmz_fd_ < 0)
            throw PlatformError(errno_text("open /dev/mmz"));
        l2_ = map(L2_BASE_ADDR, kGlbSize, "mmap L2");
        regs_ = map(GNNE_BASE_ADDR, kPage, "mmap GNNE registers");
        gnne_set_base(regs_);
    }
    ~Device() {
        if (l2_)
            ::munmap(l2_, kGlbSize);
        if (regs_)
            ::munmap(regs_, kPage);
        if (mmz_fd_ >= 0)
            ::close(mmz_fd_);
        if (gnne_fd_ >= 0)
            ::close(gnne_fd_);
    }
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    uint8_t *l2() const { return static_cast<uint8_t *>(l2_); }

    // Runs [begin, end) and waits for the end. Returns false (after printing the reason) if the NPU did not finish
    // cleanly: timeout, or an exception (illegal instruction, CCR error, watchdog).
    bool run(uint64_t begin, uint64_t end, bool verbose) {
        gnne_init(); // disables the engine and waits for the reset
        gnne_set_time_out(kGnneTimeout);
        if (gnne_enable(begin, end, 0) != 0) {
            std::fprintf(stderr, "  the NPU is busy\n");
            return false;
        }
        pollfd fds{};
        fds.fd = gnne_fd_;
        fds.events = POLLIN;
        const int pr = ::poll(&fds, 1, kTimeoutMs);
        bool ok = true;
        if (pr <= 0) {
            std::fprintf(stderr, "  %s after %d ms\n", pr == 0 ? "timeout" : "poll failed", kTimeoutMs);
            ok = false;
        }
        const gnne_status status = gnne_get_status();
        if (status.bits.exception_status != 0) {
            std::fprintf(stderr, "  NPU exception %u (1 illegal instruction, 2 CCR error, 3 timeout)\n",
                unsigned(status.bits.exception_status));
            ok = false;
        }
        if (verbose && ok) {
            std::printf("NPU status after the run:\n");
            gnne_dump_status();
            gnne_dump_pc();
        }
        if (!ok) {
            gnne_dump_status();
            gnne_dump_pc();
            gnne_disable();
        }
        return ok;
    }

private:
    void *map(uint64_t phys, size_t size, const char *what) {
        void *p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, mmz_fd_, static_cast<off_t>(phys));
        if (p == MAP_FAILED)
            throw PlatformError(errno_text(what));
        return p;
    }

    int gnne_fd_ = -1;
    int mmz_fd_ = -1;
    void *l2_ = nullptr;
    void *regs_ = nullptr;
};

} // namespace

int run_test(TestCase &tc, const RunOptions &options) {
    const Layout layout = make_layout(tc);
    try {
        Device device;
        MmzBuffer ddr_in(layout.ddr_in_size, "isa_in");
        MmzBuffer ddr_out(layout.ddr_out_size, "isa_out");

        MemoryImage mem;
        mem.glb = device.l2();
        mem.ddr_in = ddr_in.data();
        mem.ddr_in_addr = ddr_in.phys();
        mem.ddr_out = ddr_out.data();
        mem.ddr_out_addr = ddr_out.phys();

        // The symbols have to be known before assembling: the size of `li` depends on the value.
        const AsmProgram prog = assemble_test(tc, options, program_symbols(mem));
        MmzBuffer text(prog.text.size(), "isa_text");
        std::memcpy(text.data(), prog.text.data(), prog.text.size());
        init_memory(tc, layout, mem);
        text.flush();
        ddr_in.flush();
        ddr_out.flush(); // the 0xA5 fill must not be written back over the NPU's results later

        const bool finished = device.run(text.phys(), text.phys() + prog.text.size(), options.verbose);
        if (tc.expect_fail) {
            if (finished) {
                std::fprintf(stderr, "FAIL %s: the NPU accepted a program that has to fail\n", tc.path.c_str());
                return kMismatch;
            }
            std::printf("PASS %s (the NPU rejected it, as expected)\n", tc.path.c_str());
            return kPass;
        }
        if (!finished) {
            std::fprintf(stderr, "FAIL %s: the NPU did not finish the program\n", tc.path.c_str());
            return kPlatformFailure;
        }

        ddr_out.flush();
        if (const int mismatches = verify_outputs(tc, layout, mem)) {
            std::fprintf(stderr, "FAIL %s: %d element(s) differ\n", tc.path.c_str(), mismatches);
            return kMismatch;
        }
        std::printf("PASS %s (%zu bytes of code)\n", tc.path.c_str(), prog.text.size());
        return kPass;
    } catch (const PlatformError &e) {
        std::fprintf(stderr, "FAIL %s: %s\n", tc.path.c_str(), e.what());
        return kPlatformFailure;
    }
}

} // namespace k230::isa_test
