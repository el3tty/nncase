#include "gmodel_builder.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nncase/runtime/util.h>
#include <sstream>
#include <stdexcept>

using namespace nncase;
using namespace nncase::runtime;

namespace nncase::runtime
{
/** Short type name used in the ".desc" file. In the original library this is a weak
 *  (header inline) function, get_display_name(datatype_t) at 0x2c380, so it is defined here. */
static std::string get_display_name(const datatype_t &dtype)
{
    switch (dtype->typecode())
    {
    case dt_boolean:
        return "bool";
    case dt_utf8char:
        return "u8char";
    case dt_int8:
        return "i8";
    case dt_int16:
        return "i16";
    case dt_int32:
        return "i32";
    case dt_int64:
        return "i64";
    case dt_uint8:
        return "u8";
    case dt_uint16:
        return "u16";
    case dt_uint32:
        return "u32";
    case dt_uint64:
        return "u64";
    case dt_float16:
        return "f16";
    case dt_float32:
        return "f32";
    case dt_float64:
        return "f64";
    case dt_bfloat16:
        return "bf16";
    case dt_pointer:
        return "*";
    case dt_valuetype:
        return "val";
    default:
        return "Unsupported data type";
    }
}

int gmodel_builder::number = 0;

namespace
{
/** gnne dtype codes indexed by (typecode - 2) for int8..float32 (CSWTCH_907). */
constexpr int gnne_dtype_by_typecode[10] = {
    1,    // int8
    2,    // int16
    9999, // int32
    9999, // int64
    0,    // uint8
    9999, // uint16
    9999, // uint32
    9999, // uint64
    3,    // float16
    4     // float32
};

int gnne_dtype_code(const datatype_t &dtype)
{
    unsigned idx = static_cast<uint8_t>(dtype->typecode()) - 2u;
    return idx <= 9u ? gnne_dtype_by_typecode[idx] : 9999;
}

const char *dtype_short_name(typecode_t code)
{
    switch (code)
    {
    case dt_boolean: return "bool";
    case dt_utf8char: return "u8char";
    case dt_int8: return "i8";
    case dt_int16: return "i16";
    case dt_int32: return "i32";
    case dt_int64: return "i64";
    case dt_uint8: return "u8";
    case dt_uint16: return "u16";
    case dt_uint32: return "u32";
    case dt_uint64: return "u64";
    case dt_float16: return "f16";
    case dt_float32: return "f32";
    case dt_float64: return "f64";
    case dt_bfloat16: return "bf16";
    case dt_pointer: return "*";
    case dt_valuetype: return "val";
    default: return "Unsupported data type";
    }
}

std::string strip_last(const std::string &s)
{
    return s.empty() ? s : s.substr(0, s.size() - 1);
}

tensor as_tensor_or_fail(const value_t &v, const char *message)
{
    if (!v.is_a<tensor>())
        fail_fast(message);
    return v.as<tensor>().unwrap_or_throw();
}
} // namespace

// ---------------------------------------------------------------------------
// runtime1.cpp
// ---------------------------------------------------------------------------
result<std::vector<int>> to_gnne_arg(tensor t, size_t address)
{
    std::vector<int> args;
    if (!t->shape().empty())
    {
        for (auto dim : t->shape())
            args.push_back(static_cast<int>(dim));
        args.push_back(static_cast<int>(address));

        auto prim = t->dtype().as<prim_type_t>();
        if (prim.is_err())
            return err(std::errc::invalid_argument);
        auto code = prim.unwrap()->typecode();
        if (code < dt_int8 || code > dt_bfloat16)
            return err(std::errc::not_supported);
        args.push_back(static_cast<int>(code) - dt_int8);
        return ok(std::move(args));
    }

    // scalar tensors are passed by value; only 32 bit integers are supported
    auto code = t->dtype().as<prim_type_t>().unwrap()->typecode();
    if (code != dt_int32 && code != dt_uint32)
        return err(std::errc::not_supported);
    try_var(span, get_input_span(t));
    args.push_back(*reinterpret_cast<const int *>(span.data()));
    return ok(std::move(args));
}

// ---------------------------------------------------------------------------
// runtime2.cpp
// ---------------------------------------------------------------------------
template <>
std::string to_x8<unsigned int>(unsigned int value)
{
    std::stringstream ss;
    ss << std::hex << std::setw(8) << std::setfill('0') << value;
    return ss.str();
}

// ---------------------------------------------------------------------------
// runtime3.cpp
// ---------------------------------------------------------------------------
std::ostream &write_seq(std::ostream &os, const dims_t &seq, const std::string &sep)
{
    std::stringstream ss;
    for (size_t i = 0; i < seq.size(); i++)
        ss << seq[i] << sep;
    std::string s = ss.str();
    size_t n = s.size() > sep.size() ? s.size() - sep.size() : 0;
    os << s.substr(0, n);
    return os;
}

// ---------------------------------------------------------------------------
// gmodel_builder10.cpp: constructor
// ---------------------------------------------------------------------------
gmodel_builder::gmodel_builder(gsl::span<const gsl::byte> text, gsl::span<const value_t> inputs, value_t output,
    const std::string &sim_dir, const std::string &dump_root, gsl::span<const gsl::byte> rdata, bool ctrl_binary)
    : inputs_(inputs.begin(), inputs.end()),
      text_(text),
      rdata_(rdata),
      ctrl_binary_(ctrl_binary),
      dump_root_(dump_root)
{
    for (auto &in : inputs)
    {
        auto t = as_tensor_or_fail(in, "input should be a tensor");
        if (!t->shape().empty())
            tensor_inputs_.push_back(in);
    }

    if (output.is_a<tuple>())
    {
        auto tup = output.as<tuple>().unwrap_or_throw();
        for (auto &f : tup->fields())
            outputs_.push_back(f);
    }
    else if (output.is_a<tensor>())
    {
        outputs_.push_back(output);
    }
    else
    {
        fail_fast("unsupported value");
    }

    // every builder gets its own "<sim_dir>/gmodel_dump_dir/<n>/" directory
    auto dir = std::filesystem::path(sim_dir) / "gmodel_dump_dir" / std::to_string(number++);
    dump_dir_ = dir.string() + "/";
    std::filesystem::create_directories(dir);

    arg_path_ = dump_dir_ + "arg";
    bin_path_ = dump_dir_ + "gmodel.bin";
    desc_path_ = dump_dir_ + "gmodel.desc";
    ctrl_path_ = dump_dir_ + "gmodel.pc_addr_ctrl";
    sim_dir_ = std::filesystem::path(sim_dir);
}

// ---------------------------------------------------------------------------
// gmodel_builder11.cpp: destructor
// ---------------------------------------------------------------------------
gmodel_builder::~gmodel_builder()
{
    // keep the last two dump directories, remove the one before them
    if (number > 1)
    {
        auto old = std::filesystem::path(sim_dir_) / "gmodel_dump_dir" / std::to_string(number - 2);
        std::filesystem::remove_all(old);
    }
}

// ---------------------------------------------------------------------------
// gmodel_builder1.cpp
// ---------------------------------------------------------------------------
gmodel_builder::output_info gmodel_builder::get_output_info()
{
    std::stringstream offsets, sizes, dtypes;
    for (auto &out : outputs_)
    {
        auto t = as_tensor_or_fail(out, "output is not a tensor");
        auto &pos = basement_[&t->buffer()];
        offsets << pos.start << ",";
        sizes << pos.size / t->dtype()->size_bytes() << ",";
        dtypes << gnne_dtype_code(t->dtype()) << ",";
    }
    // drop the trailing comma of every list
    return { strip_last(dtypes.str()), strip_last(sizes.str()), strip_last(offsets.str()) };
}

// ---------------------------------------------------------------------------
// gmodel_builder2.cpp
// ---------------------------------------------------------------------------
void gmodel_builder::write_vkpu_invoke_args()
{
    if (outputs_.size() != 1)
        return;
    auto out = as_tensor_or_fail(outputs_[0], "output[0] is not a tensor");

    std::ofstream sc(arg_path_ + ".sc");
    sc << (sim_dir_ / "ic_env/ld-linux-x86-64.so.2").string() << std::endl;

    uint32_t numel = 1;
    for (auto d : out->shape())
        numel *= static_cast<uint32_t>(d);
    int dtype = gnne_dtype_code(out->dtype());

    auto abs_parent = std::filesystem::absolute(sim_dir_.parent_path()).string() + "/";
    auto ic_env = (sim_dir_ / "ic_env").string() + "/";
    auto vkpu = std::filesystem::absolute(sim_dir_ / "ic_env" / "Vkpu_top").string();

    std::stringstream args;
    args << "--library-path" << ' ' << ic_env << ' ' << vkpu << ' ' << bin_path_ << ' ' << text_start_ << ' '
         << out_base_ << ' ' << numel << ' ' << abs_parent << ' ' << dtype << ' ' << "+perf" << ' ' << "+ckp" << ' '
         << "+hang_10000" << ' ' << "\n" << ' ';
    vkpu_args_ = args.str();
    sc << vkpu_args_;
}

// ---------------------------------------------------------------------------
// gmodel_builder3.cpp
// ---------------------------------------------------------------------------
int gmodel_builder::run_gmodel()
{
#ifdef _WIN32
    // the RTL model (ic_env/Vkpu_top) is a Linux x86-64 executable started through the
    // bundled dynamic loader; there is no Windows build of it
    return ENOSYS;
#else
    auto ld = (sim_dir_ / "ic_env/ld-linux-x86-64.so.2").string();
    std::string command = ld + " " + vkpu_args_;
    system(command.c_str());
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// gmodel_builder4.cpp
// ---------------------------------------------------------------------------
int gmodel_builder::alloc_buffer()
{
    // inputs live at in_base_, outputs at out_base_; both regions start at offset 0
    uint32_t offset = 0;
    for (auto &v : tensor_inputs_)
    {
        if (!v.is_a<tensor>())
            return EINVAL;
        auto t = v.as<tensor>().unwrap();
        uint32_t size = static_cast<uint32_t>(t->buffer().size_bytes());
        basement_[&t->buffer()] = { in_base_ + offset, offset, size };
        offset += size;
    }

    offset = 0;
    for (auto &v : outputs_)
    {
        if (!v.is_a<tensor>())
            return EINVAL;
        auto t = v.as<tensor>().unwrap();
        uint32_t size = static_cast<uint32_t>(t->buffer().size_bytes());
        basement_[&t->buffer()] = { out_base_ + offset, offset, size };
        offset += size;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// gmodel_builder5.cpp
// ---------------------------------------------------------------------------
int gmodel_builder::write_bin()
{
    std::ofstream bin(bin_path_, std::ios::out | std::ios::binary);

    // gmodel.bin = [inputs][pad to 8][rdata][pad to 8][text]; the tail position is
    // remembered in rdata_start_/text_start_/end_
    in_base_ = 0;
    uint32_t inputs_size = 0;
    for (auto &v : tensor_inputs_)
    {
        if (!v.is_a<tensor>())
            return EINVAL;
        inputs_size += static_cast<uint32_t>(v.as<tensor>().unwrap()->buffer().size_bytes());
    }

    auto pad8 = [&]() {
        char zero = 0;
        for (auto n = static_cast<int64_t>(bin.tellp()) & 7; n != 0 && n < 8; ++n)
            bin.write(&zero, 1);
    };

    bin.seekp(in_base_ + inputs_size);
    pad8();
    rdata_start_ = static_cast<uint32_t>(bin.tellp());
    bin.write(reinterpret_cast<const char *>(rdata_.data()), rdata_.size());
    pad8();
    text_start_ = static_cast<uint32_t>(bin.tellp());
    bin.write(reinterpret_cast<const char *>(text_.data()), text_.size());
    end_ = static_cast<uint32_t>(bin.tellp());
    out_base_ = end_;

    if (int ret = alloc_buffer())
        return ret;

    // fill in the input data at the addresses chosen by alloc_buffer()
    for (auto &v : tensor_inputs_)
    {
        auto t = v.as<tensor>().unwrap();
        auto start = basement_[&t->buffer()].start;
        bin.seekp(start);
        auto span = get_input_span(t);
        if (span.is_err())
            return span.unwrap_err().value();
        bin.write(reinterpret_cast<const char *>(span.unwrap().data()), span.unwrap().size());
    }
    return 0;
}

// ---------------------------------------------------------------------------
// gmodel_builder6.cpp
// ---------------------------------------------------------------------------
void gmodel_builder::write_invoke_args()
{
    std::ofstream txt(arg_path_ + ".txt");
    txt << bin_path_ << std::endl;
    txt << text_start_ << std::endl;

    auto info = get_output_info();
    auto or_zero = [](const std::string &s) { return s.empty() ? std::string("0") : s; };
    txt << or_zero(info.offsets) << std::endl;
    txt << or_zero(info.sizes) << std::endl;

    // working directory of the simulator, quoted with \ and " escaped
    std::string dir = std::filesystem::path(bin_path_).parent_path().string();
    std::stringstream quoted;
    quoted << '"';
    for (char c : dir)
    {
        if (c == '\\' || c == '"')
            quoted << '\\';
        quoted << c;
    }
    quoted << '"';
    txt << quoted.str() << std::endl;

    txt << or_zero(info.dtypes) << std::endl;
    txt << "wait-key" << std::endl;
}

// ---------------------------------------------------------------------------
// gmodel_builder7.cpp
// ---------------------------------------------------------------------------
int gmodel_builder::read_output()
{
    auto path = sim_dir_ / "nncase_result_0_sc.bin";
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Cannot open file: " + path.string());

    ifs.seekg(0, std::ios::end);
    auto size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    ifs.read(data.data(), size);

    auto out = as_tensor_or_fail(outputs_[0], "output not a tensor");
    auto span = get_output_span(out);
    if (span.is_err())
        return span.unwrap_err().value();
    std::memcpy(span.unwrap().data(), data.data(), data.size());
    return 0;
}

// ---------------------------------------------------------------------------
// gmodel_builder8.cpp
// ---------------------------------------------------------------------------
int gmodel_builder::write_glb_ctrl()
{
    std::ofstream os(ctrl_path_, std::ios::out | std::ios::binary);

    std::vector<std::vector<int>> args; // gnne arguments of every input/output
    std::vector<size_t> out_addrs;      // running address of every output
    std::vector<uint32_t> words;        // value written in the control registers
    int64_t first = -1;
    uint64_t addr = 0;

    for (auto &v : inputs_)
    {
        if (!v.is_a<tensor>())
            return EINVAL;
        auto t = v.as<tensor>().unwrap();
        auto span = get_input_span(t);
        if (span.is_err())
            return span.unwrap_err().value();
        addr = basement_[&t->buffer()].start;
        auto arg = to_gnne_arg(t, addr);
        if (arg.is_err())
            return arg.unwrap_err().value();
        args.push_back(arg.unwrap());
        if (first < 0)
            first = addr;
        words.push_back(static_cast<uint32_t>(in_base_ + addr - first));
    }

    // outputs follow the (8 byte aligned) last input start + size of the text blob
    addr = (addr + 7) & ~uint64_t(7);
    addr += text_.size();

    first = -1;
    for (auto &v : outputs_)
    {
        if (!v.is_a<tensor>())
            return EINVAL;
        auto t = v.as<tensor>().unwrap();
        auto span = get_input_span(t);
        if (span.is_err())
            return span.unwrap_err().value();
        auto arg = to_gnne_arg(t, 0);
        if (arg.is_err())
            return arg.unwrap_err().value();
        args.push_back(arg.unwrap());
        out_addrs.push_back(addr);
        if (first < 0)
            first = addr;
        words.push_back(static_cast<uint32_t>(out_base_ + addr - first));
        addr += span.unwrap().size();
    }
    words.push_back(rdata_start_);
    words.push_back(end_);

    if (!ctrl_binary_)
    {
        // up to 12 words as 128 bit lines, highest word first, missing words are zero
        auto line = [&](uint32_t line_addr, size_t base) {
            os << to_x8(line_addr) << ' ';
            for (int i = 3; i >= 0; --i)
            {
                size_t idx = base + i;
                os << to_x8(idx < words.size() ? words[idx] : 0u);
            }
            os << std::endl;
        };
        os << "00000000 ";
        line(0, 0);
        if (words.size() > 4)
            line(0x10, 4);
        else
            os << "00000010 " << std::endl;
        if (words.size() > 8)
            line(0x20, 8);
    }
    else
    {
        // arguments form a downward growing stack that ends at 0x40000
        std::reverse(args.begin(), args.end());
        std::vector<int> flat;
        for (auto &a : args)
            flat.insert(flat.end(), a.begin(), a.end());

        for (size_t i = 0, na = 4; i < flat.size(); i += 4, na += 4)
        {
            std::string line;
            for (size_t j = i; j < na; ++j)
                line += j < flat.size() ? to_x8(static_cast<uint32_t>(flat[j])) : "00000000";
            os << to_x8(static_cast<uint32_t>(0x40000 - 4 * na)) << ' ' << line << std::endl;
        }
        os << "00000000 " << "00000000" << to_x8(out_base_) << "00000000" << "00000000" << std::endl;
    }

    os << "00400100 ";
    os << std::hex << std::setfill('0') << std::setw(32) << text_start_ << std::endl;
    os << "00400120 00000001000000010000000000000000" << std::endl;
    return 0;
}

// ---------------------------------------------------------------------------
// gmodel_builder9.cpp
// ---------------------------------------------------------------------------
int gmodel_builder::write_desc()
{
    std::ofstream os(desc_path_, std::ios::out);
    os << tensor_inputs_.size() << " " << outputs_.size() << std::endl;

    auto shape_str = [](tensor t) {
        std::stringstream ss;
        for (auto d : t->shape())
            ss << d << ", ";
        return strip_last(strip_last(ss.str())); // drop the trailing ", "
    };

    // "<dtype> (<shape>)" for every input and output
    for (auto &v : tensor_inputs_)
    {
        if (!v.is_a<tensor>())
            return EINVAL;
        auto t = v.as<tensor>().unwrap();
        os << dtype_short_name(t->dtype().as<prim_type_t>().unwrap()->typecode()) << " (" << shape_str(t) << ")"
           << std::endl;
    }
    for (auto &v : outputs_)
    {
        if (!v.is_a<tensor>())
            return EINVAL;
        auto t = v.as<tensor>().unwrap();
        os << get_display_name(t->dtype()) << " (" << shape_str(t) << ")" << std::endl;
    }

    // "<start> <offset> <size>" of every input and output buffer
    auto write_pos = [&](const std::vector<value_t> &values) -> int {
        for (auto &v : values)
        {
            if (!v.is_a<tensor>())
                return EINVAL;
            auto t = v.as<tensor>().unwrap();
            auto &pos = basement_[&t->buffer()];
            write_seq(os, dims_t { pos.start, pos.offset, pos.size }, " ") << std::endl;
        }
        return 0;
    };
    if (int ret = write_pos(tensor_inputs_))
        return ret;
    if (int ret = write_pos(outputs_))
        return ret;

    // rdata, (end), text segments
    write_seq(os, dims_t { rdata_start_, 0, end_ - rdata_start_ }, " ") << std::endl;
    write_seq(os, dims_t { end_, 0, 0 }, " ") << std::endl;
    write_seq(os, dims_t { text_start_, 0, end_ - text_start_ }, " ") << std::endl;
    return 0;
}
} // namespace nncase::runtime
