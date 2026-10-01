#include <fcntl.h>
#include <sys/mman.h>
#include <iostream>

#include <nncase/functional/k230/dynamic_gnne_matmul.h>
#include <nncase/runtime/k230/gnne.h>
#include <nncase/runtime/runtime_loader.h>

#include "runtime_module_impl.h"
#include "runtime_function.h"

BEGIN_NS_NNCASE_RT_K230
using namespace nncase;

k230_runtime_module::~k230_runtime_module() {
    if (munmap(l2_base_, l2_size_) != 0) {
        std::cerr << "munmap l2 base vaddr_ failed: " << strerror(errno);
        abort();
    }
    l2_base_ = nullptr;
    l2_size_ = 0;

    if (munmap(gnne_base_, gnne_size_) != 0) {
        std::cerr << "munmap gnne base vaddr_ failed: " << strerror(errno);
        abort();
    }
    gnne_base_ = nullptr;
    gnne_size_ = 0;

    if (mmz_fd_ > 0) {
        close(mmz_fd_);
        mmz_fd_ = -1;
    }
    if (gnne_fd_ > 0) {
        close(gnne_fd_);
        gnne_fd_ = -1;
    }
}

k230_runtime_module::k230_runtime_module() {
    int fd1 = open("/dev/k230-gnne", O_RDWR);   // exact path unverified
    if (fd1 < 0) {
        std::cerr << "open " << "/dev/k230-gnne" << " failed: " << strerror(errno) << std::endl;
        abort();
    }
    gnne_fd_ =
        fd1; // this+0xc8 � retained, but not visibly reused in this function

    int fd2 = open("/dev/mmz", O_RDWR | O_SYNC);        // exact path unverified
    if (fd2 < 0) {
        std::cerr << "open " << "/dev/mmz" << " failed: " << strerror(errno) << std::endl;
        abort();
    }
    mmz_fd_ = fd2;  // this+0xcc

    l2_size_ = 0x200;
    void *l2_base = mmap(nullptr, 0x200, PROT_READ | PROT_WRITE, MAP_SHARED,
                          fd2, L2_BASE_ADDR);
    if (l2_base == MAP_FAILED) {
        std::cerr << "mmap failed: " << strerror(errno) << std::endl;
        abort();
    }
    l2_base_ = l2_base;  // this+0xa0

    gnne_size_ = 0x210;
    void *gnne_base = mmap(nullptr, 0x210, PROT_READ | PROT_WRITE, MAP_SHARED,
                            fd2, GNNE_BASE_ADDR);
    if (gnne_base == MAP_FAILED) {
        std::cerr << "mmap failed: " << strerror(errno) << std::endl;
        abort();
    }
    gnne_base_ = gnne_base;  // this+0xb8

    gnne_set_base(gnne_base);
}

result<void> k230_runtime_module::initialize_before_functions(
    runtime_module_init_context &context) noexcept {
    try_var(rdata, context.get_or_read_section(".rdata", rdata_storage_, true));
    try_var(text, context.get_or_read_section(".text", text_storage_, true));
    this->text_ = text;
    this->rdata_ = rdata;

    return ok();
}

result<std::unique_ptr<runtime_function>> k230_runtime_module::create_function() noexcept {
    auto *fn = new (std::nothrow) k230_runtime_function(*this);
    if (!fn) {
        return err(std::errc::not_enough_memory);
    }
    return ok(std::unique_ptr<runtime_function>(fn));
}

result<std::unique_ptr<runtime_module>> create_k230_runtime_module() {
    auto *mod = new (std::nothrow) k230_runtime_module();
    if (!mod) {
        return err(std::errc::not_enough_memory);
    }
    return ok(std::unique_ptr<runtime_module>(mod));
}

result<std::vector<std::pair<std::string, runtime_module::custom_call_type>>>
create_k230_custom_calls() {
    return ok(std::vector<std::pair<std::string, runtime_module::custom_call_type>>{
        {"K230DynamicGNNEMatMul", &nncase::functional::k230::dynamic_gnne_matmul}
    });
}

END_NS_NNCASE_RT_K230


extern "C" NNCASE_MODULES_K230_API
void create_runtime_module(nncase::result<std::unique_ptr<nncase::runtime::runtime_module>> &out) {
    out = nncase::runtime::k230::create_k230_runtime_module();
}

extern "C" NNCASE_MODULES_K230_API
void collect_custom_call(
    nncase::result<std::vector<std::pair<std::string, nncase::runtime::runtime_module::custom_call_type>>> &out) {
    out = nncase::runtime::k230::create_k230_custom_calls();
}

BEGIN_NS_NNCASE_RUNTIME
runtime_registration builtin_runtimes[] = {"k230", create_runtime_module,
                                           collect_custom_call};
END_NS_NNCASE_RUNTIME