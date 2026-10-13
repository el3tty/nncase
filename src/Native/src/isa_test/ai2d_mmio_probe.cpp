// Probe: AI2D identity copy (5x6 -> pitch 7, DDR -> DDR) by writing the AI2D registers from the CPU, like the
// standalone AI2D driver (runtime/k230/ai2d_builder.cpp). No NPU program involved.
#include "mmz.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char **argv) {
    const int variant = argc > 1 ? atoi(argv[1]) : 0; // 0: plain; 1: intr_mask 1 (bit 30 of word 31)
    int mmz_fd = open("/dev/mmz", O_RDWR | O_SYNC);
    int ai2d_fd = open("/dev/k230-ai2d", O_RDWR);
    if (mmz_fd < 0 || ai2d_fd < 0) {
        std::perror("open");
        return 1;
    }
    auto *page = (volatile uint8_t *)mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, mmz_fd, 0x80400000);
    if (page == MAP_FAILED) {
        std::perror("mmap");
        return 1;
    }
    auto *regs = (volatile uint32_t *)(page + 0xC00);

    unsigned long in_phys = 0, out_phys = 0;
    void *in_virt = nullptr, *out_virt = nullptr;
    kd_mpi_sys_mmz_alloc_cached(&in_phys, &in_virt, "a2in", "anonymous", 0x1000);
    kd_mpi_sys_mmz_alloc_cached(&out_phys, &out_virt, "a2out", "anonymous", 0x1000);
    auto *in = (uint8_t *)in_virt;
    auto *out = (uint8_t *)out_virt;
    for (int i = 0; i < 30; i++)
        in[i] = uint8_t(i * 7 + 1);
    std::memset(out, 0xA5, 64);
    kd_mpi_sys_mmz_flush_cache(in_phys, in_virt, 0x1000);
    kd_mpi_sys_mmz_flush_cache(out_phys, out_virt, 0x1000);

    // variant & 16: GLB -> GLB (src_ind = dst_ind = 0), pointers are byte offsets into the 4 MiB L2 mapped below
    volatile uint8_t *l2 = nullptr;
    if (variant & 16) {
        l2 = (volatile uint8_t *)mmap(nullptr, 0x400000, PROT_READ | PROT_WRITE, MAP_SHARED, mmz_fd, 0x80000000);
        if (l2 == MAP_FAILED) {
            std::perror("mmap L2");
            return 1;
        }
        for (int i = 0; i < 30; i++)
            l2[0x4000 + i] = uint8_t(i * 7 + 1);
        for (int i = 0; i < 64; i++)
            l2[0x5000 + i] = 0xA5;
    }
    uint32_t w[36] = {};
    w[0] = (uint32_t)in_phys;
    w[4] = (uint32_t)out_phys;
    w[8] = 6;
    w[10] = 7;
    w[12] = 0x44800000;
    w[15] = 0x44800000;
    w[18] = 0x33000009;
    w[25] = 0x07000000;
    w[30] = 0x00050006;
    w[31] = 0x00050006 | (variant ? 1u << 30 : 0);
    w[35] = 0x80000000;
    if (variant & 2) { // reset values of the registers the driver also writes (YUV coefficients, README section 5.3)
        w[20] = 0x00000100;
        w[21] = 0x00F6E124;
        w[22] = 0x00F9B100;
        w[23] = 0x0007DF6B;
        w[24] = 0x00EFC000;
        w[26] = 0x00208100;
        w[31] &= ~(1u << 30);
    }
    if (variant & 8) // cmd_id 0 like the standalone driver (the compiler's NPU stream uses 1)
        w[25] &= ~(1u << 26);
    if (variant & 16) {
        w[0] = 0x4000;
        w[4] = 0x5000;
        w[25] &= ~(3u << 24);
    }
    if (variant & 4) // dst_y/dst_x = 0 but written as separate single writes in descending order (probe of the write order)
        w[35] = 0x80000000;

    regs[48] = 0x1388;
    regs[49] = regs[50] = regs[51] = 0;
    std::printf("in 0x%lx out 0x%lx, timeout readback 0x%x\n", in_phys, out_phys, regs[48]);

    for (int i = 0; i < 36; i += 4) {
        regs[i + 0] = w[i + 0];
        regs[i + 1] = w[i + 1];
        regs[i + 2] = w[i + 2];
        regs[i + 3] = w[i + 3];
    }
    struct pollfd fds {
        ai2d_fd, POLLIN, 0
    };
    int pr = 0;
    for (int i = 0; i < 10 && !pr; i++) {
        pr = poll(&fds, 1, 100);
        std::printf("t=%d00ms intr_num=0x%04x w35=0x%08x\n", i, *(volatile uint16_t *)(page + 0xC00 + 0x90), regs[35]);
    }
    std::printf("poll = %d, intr_num = 0x%04x\n", pr, *(volatile uint16_t *)(page + 0xC00 + 0x90));
    for (int i = 0; i < 36; i += 4)
        std::printf("reg[%2d..%2d] = %08x %08x %08x %08x\n", i, i + 3, regs[i], regs[i + 1], regs[i + 2], regs[i + 3]);
    kd_mpi_sys_mmz_flush_cache(out_phys, out_virt, 0x1000);
    if (variant & 16)
        out = (uint8_t *)(l2 + 0x5000);
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 7; x++)
            std::printf("%02x ", out[y * 7 + x]);
        std::printf("\n");
    }
    return 0;
}
