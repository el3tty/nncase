#pragma once
#include <filesystem>
#include <map>
#include <nncase/runtime/buffer.h>
#include <nncase/runtime/result.h>
#include <nncase/tensor.h>
#include <string>
#include <vector>

namespace nncase::runtime
{
/** Returns the gnne argument list of a tensor: shape, address, dtype code
 *  (or, for scalar int32/uint32 tensors, the scalar value itself). */
result<std::vector<int>> to_gnne_arg(tensor tensor, size_t address);

/** 8 digit, zero padded, lower-case hexadecimal. */
template <class T>
std::string to_x8(T value);

/** Writes the elements of `seq` separated by `sep` (no trailing separator). */
std::ostream &write_seq(std::ostream &os, const dims_t &seq, const std::string &sep);

class gmodel_builder
{
  public:
    struct BasementPosition
    {
        uint32_t start;  // absolute address (region base + offset)
        uint32_t offset; // offset inside the region
        uint32_t size;   // tensor size in bytes
    };

    /** The three comma separated lists written to the "arg.txt" file. */
    struct output_info
    {
        std::string dtypes;  // gnne dtype code of every output
        std::string sizes;   // number of elements of every output
        std::string offsets; // start address of every output
    };

    gmodel_builder(gsl::span<const gsl::byte> text, gsl::span<const value_t> inputs, value_t output,
        const std::string &sim_dir, const std::string &dump_root,
        gsl::span<const gsl::byte> rdata, bool ctrl_binary);
    ~gmodel_builder();

    output_info get_output_info();
    int alloc_buffer();
    int write_bin();
    int write_desc();
    int write_glb_ctrl();
    void write_invoke_args();
    void write_vkpu_invoke_args();
    int run_gmodel();
    int read_output();

    static int number;

  private:
    uint32_t in_base_ = 0;
    uint32_t out_base_ = 0;
    uint32_t rdata_start_ = 0;
    uint32_t end_ = 0;
    uint32_t text_start_ = 0;
    std::map<const buffer_slice *, BasementPosition> basement_;
    std::vector<value_t> inputs_;
    std::vector<value_t> tensor_inputs_;
    std::vector<value_t> outputs_;
    std::string bin_path_;
    std::string desc_path_;
    std::string ctrl_path_;
    std::string arg_path_;
    std::filesystem::path sim_dir_;
    std::string dump_dir_;
    std::string vkpu_args_;
    gsl::span<const gsl::byte> text_;
    gsl::span<const gsl::byte> rdata_;
    bool ctrl_binary_;
    std::string dump_root_;
};
} // namespace nncase::runtime
