// Diagnostic: runs the AI2D register sequence of an asm/ai2d test by writing the registers from the CPU, the way the
// standalone AI2D driver does (runtime/k230/ai2d_builder.cpp), and compares the result with the test's expectation.
//   ai2d_mmio_probe <test.s> [glb|ddr]
// glb: GLB -> GLB as in the test (pointers are L2 offsets).  ddr: the same data in DDR buffers (src_ind = dst_ind = 1,
// pointers are physical addresses). cmd_id is cleared and intr_mask is 0, so the CPU gets the interrupt.
#include "mmz.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <poll.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

struct Region {
    uint32_t addr;
    std::vector<uint8_t> data;
};

static std::vector<uint8_t> parse_list(const std::string &line) {
    std::vector<uint8_t> v;
    size_t p = line.find('[');
    const char *s = line.c_str() + p + 1;
    char *end;
    while (*s && *s != ']') {
        v.push_back((uint8_t)strtol(s, &end, 10));
        s = end;
        while (*s == ',' || *s == ' ')
            s++;
    }
    return v;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s test.s [glb|ddr]\n", argv[0]);
        return 1;
    }
    const bool ddr = argc > 2 && std::string(argv[2]) == "ddr";
    std::vector<Region> ins, outs;
    uint32_t w[36] = {};
    std::ifstream f(argv[1]);
    std::string line;
    uint32_t cur = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.rfind("# IN @", 0) == 0 || line.rfind("# OUT @", 0) == 0) {
            Region r;
            r.addr = (uint32_t)strtoul(line.c_str() + line.find('@') + 1, nullptr, 0);
            r.data = parse_list(line);
            (line[2] == 'I' ? ins : outs).push_back(r);
        } else if (line.find("li x1,") != std::string::npos) {
            cur = (uint32_t)strtoul(line.c_str() + line.find("x1,") + 3, nullptr, 0);
        } else if (line.find("extrw") != std::string::npos || line.find("extraw") != std::string::npos) {
            size_t sp = line.find(' ', line.find("ext"));
            int addr = atoi(line.c_str() + sp);
            const bool zero = line.find("x0") != std::string::npos;
            w[addr / 4] = zero ? 0 : cur;
        }
    }
    int mmz_fd = open("/dev/mmz", O_RDWR | O_SYNC);
    int ai2d_fd = open("/dev/k230-ai2d", O_RDWR);
    if (mmz_fd < 0 || ai2d_fd < 0) {
        std::perror("open");
        return 1;
    }
    auto *page = (volatile uint8_t *)mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, mmz_fd, 0x80400000);
    auto *l2 = (volatile uint8_t *)mmap(nullptr, 0x400000, PROT_READ | PROT_WRITE, MAP_SHARED, mmz_fd, 0x80000000);
    if (page == MAP_FAILED || l2 == MAP_FAILED) {
        std::perror("mmap");
        return 1;
    }
    auto *regs = (volatile uint32_t *)(page + 0xC00);

    unsigned long in_phys = 0, out_phys = 0;
    void *in_virt = nullptr, *out_virt = nullptr;
    const uint32_t in_base = ins[0].addr, out_base = outs[0].addr;
    uint32_t out_base_min = out_base;
    for (auto &o : outs)
        out_base_min = o.addr < out_base_min ? o.addr : out_base_min;
    if (ddr) {
        kd_mpi_sys_mmz_alloc_cached(&in_phys, &in_virt, "a2in", "anonymous", 0x1000);
        kd_mpi_sys_mmz_alloc_cached(&out_phys, &out_virt, "a2out", "anonymous", 0x1000);
        std::memset(in_virt, 0, 0x1000);
        std::memset(out_virt, 0xA5, 0x1000);
        for (auto &r : ins)
            std::memcpy((uint8_t *)in_virt + (r.addr - in_base), r.data.data(), r.data.size());
        kd_mpi_sys_mmz_flush_cache(in_phys, in_virt, 0x1000);
        kd_mpi_sys_mmz_flush_cache(out_phys, out_virt, 0x1000);
        for (int i = 0; i < 4; i++) {
            w[i] = w[i] ? (uint32_t)in_phys + (w[i] - in_base) : 0;
            w[4 + i] = w[4 + i] ? (uint32_t)out_phys + (w[4 + i] - out_base_min) : 0;
        }
        w[25] |= 3u << 24;
    } else {
        for (auto &r : ins)
            for (size_t i = 0; i < r.data.size(); i++)
                l2[r.addr + i] = r.data[i];
        for (auto &r : outs)
            for (size_t i = 0; i < r.data.size(); i++)
                l2[r.addr + i] = 0xA5;
    }
    w[25] &= ~(1u << 26); // cmd_id 0
    w[31] &= ~(1u << 30); // intr_mask 0

    regs[48] = 0x1388;
    regs[49] = regs[50] = regs[51] = 0;
    for (int i = 0; i < 36; i += 4) {
        regs[i + 0] = w[i + 0];
        regs[i + 1] = w[i + 1];
        regs[i + 2] = w[i + 2];
        regs[i + 3] = w[i + 3];
    }
    struct pollfd fds {
        ai2d_fd, POLLIN, 0
    };
    int pr = poll(&fds, 1, 1000);
    std::printf("%s (%s): poll = %d, intr_num = %u\n", argv[1], ddr ? "ddr" : "glb", pr,
        *(volatile uint16_t *)(page + 0xC00 + 0x90));
    if (ddr)
        kd_mpi_sys_mmz_flush_cache(out_phys, out_virt, 0x1000);
    int bad = 0;
    for (size_t k = 0; k < outs.size(); k++) {
        for (size_t i = 0; i < outs[k].data.size(); i++) {
            const uint8_t got = ddr ? ((uint8_t *)out_virt)[outs[k].addr - out_base_min + i] : l2[outs[k].addr + i];
            if (got != outs[k].data[i]) {
                if (bad < 40)
                    std::printf("  OUT #%zu [%zu]: expected %u, got %u\n", k, i, outs[k].data[i], got);
                bad++;
            }
        }
    }
    std::printf("%s: %d element(s) differ\n", bad ? "FAIL" : "PASS", bad);
    return bad != 0;
}
