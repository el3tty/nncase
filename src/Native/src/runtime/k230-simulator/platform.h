// Portability helpers shared by the K230 simulator front end (Linux / Windows).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#ifdef K230_SIMULATOR_DLL
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace nncase::runtime::k230 {

inline std::string get_random_file_name() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    uint64_t hi = rng(), lo = rng();
    hi = (hi & 0xFFFFFFFFFFFF0FFFull) | 0x0000000000004000ull; // version 4
    lo = (lo & 0x3FFFFFFFFFFFFFFFull) | 0x8000000000000000ull; // variant 1
    char text[37];
    std::snprintf(text, sizeof(text), "%08x-%04x-%04x-%04x-%012llx", (unsigned)(hi >> 32), (unsigned)((hi >> 16) & 0xFFFF),
        (unsigned)(hi & 0xFFFF), (unsigned)(lo >> 48), (unsigned long long)(lo & 0xFFFFFFFFFFFFull));
    return text;
}

#ifdef K230_SIMULATOR_DLL
// Debug mode: the C model is loaded as a shared library exporting `int SimulatorMain(const char *commandline)`
// (see simulator/main.cpp) instead of being started as a separate process, so it can be debugged in-process.
inline std::string get_cmodel_library() {
    if (const char *env = std::getenv("K230_SIMULATOR"))
        if (*env)
            return env;
#ifdef _WIN32
    return "nncase.simulator.k230.sc.dll";
#else
    return "libnncase.simulator.k230.sc.so";
#endif
}

inline int run_cmodel(const std::string &mem_name, const std::string &ctrl_name, size_t text_offset,
    const std::string &dump_path) {
    using simulator_main_t = int (*)(const char *);
    static simulator_main_t entry = []() -> simulator_main_t {
        const std::string lib = get_cmodel_library();
#ifdef _WIN32
        HMODULE h = LoadLibraryA(lib.c_str());
        if (!h) {
            std::fprintf(stderr, "cannot load %s (error %lu)\n", lib.c_str(), GetLastError());
            return nullptr;
        }
        auto fn = reinterpret_cast<simulator_main_t>(GetProcAddress(h, "SimulatorMain"));
#else
        void *h = dlopen(lib.c_str(), RTLD_NOW);
        if (!h) {
            std::fprintf(stderr, "cannot load %s: %s\n", lib.c_str(), dlerror());
            return nullptr;
        }
        auto fn = reinterpret_cast<simulator_main_t>(dlsym(h, "SimulatorMain"));
#endif
        if (!fn)
            std::fprintf(stderr, "%s does not export SimulatorMain\n", lib.c_str());
        return fn;
    }();
    if (!entry)
        return -1;
    // Same arguments as the executable gets, without the program name.
    const std::string cmd = mem_name + " " + ctrl_name + " " + std::to_string(text_offset) + " \"" + dump_path + "\"";
    return entry(cmd.c_str());
}
#else
inline std::string get_cmodel_executable() {
    if (const char *env = std::getenv("K230_SIMULATOR"))
        if (*env)
            return env;
#ifdef K230_SIMULATOR_EXE
    return K230_SIMULATOR_EXE;
#elif defined(_WIN32)
    return "nncase.simulator.k230.sc.exe";
#else
    return "nncase.simulator.k230.sc";
#endif
}

inline int run_cmodel(const std::string &mem_name, const std::string &ctrl_name, size_t text_offset,
    const std::string &dump_path) {
    std::string cmd = "\"" + get_cmodel_executable() + "\" " + mem_name + " " + ctrl_name + " " + std::to_string(text_offset)
        + " \"" + dump_path + "\"";
#ifdef _WIN32
    cmd = "\"" + cmd + "\""; // cmd.exe /c strips one pair of outer quotes
#endif
    return std::system(cmd.c_str());
}

#endif // K230_SIMULATOR_DLL

} // namespace nncase::runtime::k230
