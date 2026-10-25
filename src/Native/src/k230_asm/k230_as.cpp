// k230-as: assembles a K230 NPU program (syntax: assembler.h) into the raw little-endian instruction stream.
//
//   k230-as [-o out.bin] [-l listing.txt] [-D NAME=VALUE]... <input.s | ->
//
//   -o   write the machine code (the text of a K230 function, as it is in a kmodel)
//   -l   write a listing: byte offset, machine code and source line of every instruction ("-" = stdout)
//   -D   define a symbol for li (e.g. -D DDR_IN=0x80000000)
//
// Exit code: 0 ok, 1 assembly error, 2 usage / file error.
#include "assembler.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace k230::assembler;

namespace {

int usage(const char *argv0) {
    std::fprintf(stderr, "usage: %s [-o out.bin] [-l listing.txt] [-D NAME=VALUE]... <input.s | ->\n", argv0);
    return 2;
}

std::vector<std::string> split_lines(const std::string &text) {
    std::vector<std::string> lines;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        lines.push_back(line);
    }
    return lines;
}

void write_listing(std::ostream &out, const AsmProgram &prog, const std::vector<std::string> &source) {
    for (size_t i = 0; i < prog.offsets.size(); ++i) {
        const uint32_t begin = prog.offsets[i];
        const uint32_t end = i + 1 < prog.offsets.size() ? prog.offsets[i + 1] : static_cast<uint32_t>(prog.text.size());
        uint32_t word = 0;
        for (uint32_t b = begin; b < end; ++b)
            word |= static_cast<uint32_t>(prog.text[b]) << (8 * (b - begin));
        char head[48];
        if (end - begin == 2)
            std::snprintf(head, sizeof(head), "%06x:     %04x  ", begin, word);
        else
            std::snprintf(head, sizeof(head), "%06x: %08x  ", begin, word);
        const int line = prog.lines[i];
        out << head << (line >= 1 && static_cast<size_t>(line) <= source.size() ? source[line - 1] : std::string()) << '\n';
    }
}

} // namespace

int main(int argc, char **argv) {
    std::string input, output, listing;
    SymbolTable symbols;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-o" || arg == "-l" || arg == "-D") && i + 1 < argc) {
            const std::string value = argv[++i];
            if (arg == "-o") {
                output = value;
            } else if (arg == "-l") {
                listing = value;
            } else {
                const size_t eq = value.find('=');
                const auto number = eq == std::string::npos ? std::nullopt : parse_integer(value.substr(eq + 1));
                if (eq == 0 || !number) {
                    std::fprintf(stderr, "%s: -D expects NAME=VALUE, got '%s'\n", argv[0], value.c_str());
                    return 2;
                }
                symbols[value.substr(0, eq)] = *number;
            }
        } else if (input.empty() && (arg == "-" || arg[0] != '-')) {
            input = arg;
        } else {
            return usage(argv[0]);
        }
    }
    if (input.empty() || (output.empty() && listing.empty()))
        return usage(argv[0]);

    std::string source;
    if (input == "-") {
        source.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
    } else {
        std::ifstream in(input, std::ios::binary);
        if (!in) {
            std::fprintf(stderr, "%s: cannot read %s\n", argv[0], input.c_str());
            return 2;
        }
        source.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    AsmProgram prog;
    try {
        prog = assemble(source, symbols);
    } catch (const AsmError &e) {
        std::fprintf(stderr, "%s: %s\n", input == "-" ? "<stdin>" : input.c_str(), e.what());
        return 1;
    }

    if (!output.empty()) {
        std::ofstream out(output, std::ios::binary);
        out.write(reinterpret_cast<const char *>(prog.text.data()), static_cast<std::streamsize>(prog.text.size()));
        if (!out) {
            std::fprintf(stderr, "%s: cannot write %s\n", argv[0], output.c_str());
            return 2;
        }
    }
    if (!listing.empty()) {
        const auto lines = split_lines(source);
        if (listing == "-") {
            write_listing(std::cout, prog, lines);
        } else {
            std::ofstream out(listing);
            write_listing(out, prog, lines);
            if (!out) {
                std::fprintf(stderr, "%s: cannot write %s\n", argv[0], listing.c_str());
                return 2;
            }
        }
    }
    return 0;
}
