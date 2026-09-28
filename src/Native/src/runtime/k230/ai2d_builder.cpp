#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <nncase/runtime/k230/gnne_tile_utils.h>
#include <nncase/functional/ai2d/ai2d_builder.h>
#include "k230_common.h"

BEGIN_NS_NNCASE_F_K230
using namespace nncase::runtime::k230;

ai2d_builder::ai2d_builder(dims_t &input_shape, dims_t &output_shape,
	ai2d_datatype_t ai2d_dtype, ai2d_crop_param_t crop_param,
	ai2d_shift_param_t shift_param, ai2d_pad_param_t pad_param,
	ai2d_resize_param_t resize_param, ai2d_affine_param_t affine_param)
	: ai2d_dtype_(ai2d_dtype)
	, crop_param_(crop_param)
	, shift_param_(shift_param)
	, pad_param_(pad_param)
	, resize_param_(resize_param)
	, affine_param_(affine_param)
	, config_()
	, input_shape_(input_shape)
	, output_shape_(output_shape)
	, dump_asm_(false)
{
	static constexpr char kAi2dDevice[] = "/dev/ai2d";
	ai2d_fd_ = open(kAi2dDevice, O_RDWR);
	if (ai2d_fd_ < 0)
	{
		std::cerr << "open " << kAi2dDevice << " failed: " << strerror(errno);
		abort();
	}

	static constexpr char kMemDevice[] = "/dev/mem";
	mem_fd_ = open(kMemDevice, O_RDWR | O_SYNC);
	if (mem_fd_ < 0)
	{
		std::cerr << "open " << kMemDevice << " failed: " << strerror(errno);
		abort();
	}

	ai2d_addr_.paddr = reinterpret_cast<void *>(AI2D_BASE_ADDR_PAGE_ALIGNED);
	ai2d_addr_.mmap_size = AI2D_MMAP_SIZE;

	ai2d_addr_.vaddr = mmap(nullptr, AI2D_MMAP_SIZE, PROT_READ | PROT_WRITE,
		MAP_SHARED, mem_fd_, AI2D_BASE_ADDR_PAGE_ALIGNED);
	if (ai2d_addr_.vaddr == MAP_FAILED)
	{
		std::cerr << "mmap failed: " << strerror(errno);
		abort();
	}

	ai2d_set_base(reinterpret_cast<volatile uint8_t *>(ai2d_addr_.vaddr) + AI2D_BASE_OFFSET);
	ai2d_set_time_out(0x1388); // 5000 (units unconfirmed — likely ms or a cycle count)

	if (check_config().is_err())
		throw std::runtime_error("wrong ai2d configuration.");
}

ai2d_builder::~ai2d_builder()
{
	// ---- hand-written cleanup (this is the actual source) ----
	if (munmap(const_cast<void *>(ai2d_addr_.vaddr), ai2d_addr_.mmap_size) != 0)
	{
		std::cerr << "munmap failed: " << strerror(errno);
		abort();
	}
	ai2d_addr_.vaddr = nullptr;

	if (ai2d_fd_ > 0)
	{
		close(ai2d_fd_);
		ai2d_fd_ = -1;
	}
	if (mem_fd_ > 0)
	{
		close(mem_fd_);
		mem_fd_ = -1;
	}

}

void ai2d_builder::ai2d_set_base(volatile void *addr)
{
    ai2d_addr_.addr = addr;
}

volatile uint8_t *ai2d_builder::ai2d_get_base()
{
    return reinterpret_cast<volatile uint8_t *>(ai2d_addr_.addr);
}

uint16_t ai2d_builder::ai2d_get_intr_num()
{
    return *reinterpret_cast<volatile uint16_t *>(ai2d_get_base() + 0x90);
}

void ai2d_builder::ai2d_clear_cpu_intr()
{
    volatile uint8_t *base = ai2d_get_base();
    *reinterpret_cast<volatile uint64_t *>(base + 0xA0) = 1;
    *reinterpret_cast<volatile uint32_t *>(base + 0xA8) = 0;
    *reinterpret_cast<volatile uint32_t *>(base + 0xAC) = 0;
}

void ai2d_builder::ai2d_set_time_out(uint32_t val)
{
    auto *regs = reinterpret_cast<volatile uint32_t *>(ai2d_get_base());

    regs[48] = val; // byte offset 0xC0: timeout value
    regs[49] = 0;   // 0xC4
    regs[50] = 0;   // 0xC8
    regs[51] = 0;   // 0xCC
}

void ai2d_builder::dump_asm(bool write_all)
{
}

result<void> ai2d_builder::dump_gmodel()
{
    return ok();
}

// ---------------------------------------------------------------------------
// Corrected invoke(): each split_pos_ interval is one interrupt batch, so EVERY
// register group in [split_pos_[i-1], split_pos_[i]) is written before polling once.
// (The earlier reconstruction wrote only the first group and skipped the poll on an
// empty interval; the binary does neither.)
// ---------------------------------------------------------------------------
result<void> ai2d_builder::invoke(runtime_tensor &input, runtime_tensor &output)
{
    try_(host_runtime_tensor::sync(input, sync_op_t::sync_write_back, false));
    try_(host_runtime_tensor::sync(output, sync_op_t::sync_invalidate, true));

    auto *regfile = reinterpret_cast<volatile uint32_t *>(ai2d_get_base());

    for (size_t i = 1; i < split_pos_.size(); i++)
    {
        for (uint32_t j = split_pos_[i - 1]; j < split_pos_[i]; j++)
        {
            std::vector<uint32_t> reg = regs_[j];
            const uint32_t idx = reg[0] >> 2; // update_regs stored the byte address 4*idx

            uint32_t addr_base = 0;
            if (idx == 0) // src_ch0..3_ptr: relocate by the input buffer's physical address
            {
                auto host = input.impl()->to_host().unwrap_or_throw();
                addr_base = static_cast<uint32_t>(host->buffer().as_host().unwrap().physical_address().unwrap());
            }
            else if (idx == 4) // dst_ch0..3_ptr: relocate by the output buffer's physical address
            {
                auto host = output.impl()->to_host().unwrap_or_throw();
                addr_base = static_cast<uint32_t>(host->buffer().as_host().unwrap().physical_address().unwrap());
            }

            regfile[idx + 0] = reg[4] + addr_base;
            regfile[idx + 1] = reg[3] + addr_base;
            regfile[idx + 2] = reg[2] + addr_base;
            regfile[idx + 3] = reg[1] + addr_base;
        }

        struct pollfd fds { ai2d_fd_, POLLIN, 0 };
        poll(&fds, 1, -1);

        const uint16_t intr = ai2d_get_intr_num();
        if (intr == AI2D_INTR_TIME_OUT)
        {
            std::cerr << "ai2d timeout!" << std::endl;
            return err(std::errc::timed_out);
        }
        if (intr == AI2D_INTR_EXCEPTION)
        {
            std::cerr << "ai2d exception!" << std::endl;
            return err(std::errc::invalid_argument);
        }
    }

    return ok();
}

result<void> ai2d_builder::check_config()
{
    if (input_shape_.size() < 4)
        throw std::out_of_range("small_vector::at");
    input_c_ = static_cast<int32_t>(input_shape_[1]);
    input_h_ = static_cast<int32_t>(input_shape_[2]);
    input_w_ = static_cast<int32_t>(input_shape_[3]);

    if (output_shape_.size() < 4)
        throw std::out_of_range("small_vector::at");
    output_c_ = static_cast<int32_t>(output_shape_[1]);
    output_h_ = static_cast<int32_t>(output_shape_[2]);
    output_w_ = static_cast<int32_t>(output_shape_[3]);

    if (ai2d_dtype_.src_format > ai2d_format::YUV420_I420)
    {
        if (ai2d_dtype_.src_format == ai2d_format::RGB_packed)
        {
            // shape is NHWC-ordered for packed RGB
            input_c_ = static_cast<int32_t>(input_shape_[3]);
            input_h_ = static_cast<int32_t>(input_shape_[1]);
            input_w_ = static_cast<int32_t>(input_shape_[2]);
        }
    }
    else
    {
        // YUV420_NV12 / NV21 / I420: stored plane height is 1.5x logical height
        input_c_ = 3;
        input_h_ = 2 * input_h_ / 3;
    }

    if (ai2d_dtype_.dst_format > ai2d_format::YUV420_I420)
    {
        if (ai2d_dtype_.dst_format == ai2d_format::RGB_packed)
        {
            output_c_ = static_cast<int32_t>(output_shape_[3]);
            output_h_ = static_cast<int32_t>(output_shape_[1]);
            output_w_ = static_cast<int32_t>(output_shape_[2]);
        }
    }
    else
    {
        output_c_ = 3;
        output_h_ = 2 * output_h_ / 3;
    }

    if (pad_param_.paddings.size() < 4)
        throw std::out_of_range("small_vector::at");

    int32_t pad_sum = 0;
    for (int i = 0; i < 4; i++)
        pad_sum += pad_param_.paddings[i].before + pad_param_.paddings[i].after;
    if (pad_sum == 0)
        pad_param_.pad_flag = false;

    if (!pad_param_.pad_flag)
    {
        pad_param_.paddings = {};
        pad_param_.pad_val.clear();
    }

    if (!crop_param_.crop_flag)
    {
        crop_param_.start_x = 0;
        crop_param_.start_y = 0;
        crop_param_.width = input_w_;
        crop_param_.height = input_h_;
    }

    if (affine_param_.affine_flag && resize_param_.resize_flag)
        throw std::runtime_error("We don't affine and resize simultaneously.");

    if (crop_param_.start_x < 0
        || crop_param_.start_x >= input_w_
        || input_w_ < crop_param_.start_x + crop_param_.width)
        throw std::runtime_error("Crop param(x) error.");

    if (crop_param_.start_y < 0
        || crop_param_.start_y >= input_h_
        || input_h_ < crop_param_.start_y + crop_param_.height)
        throw std::runtime_error("Crop param(y) error.");

    if (shift_param_.shift_flag)
    {
        if (ai2d_dtype_.src_format != ai2d_format::RAW16)
            throw std::runtime_error("Only Raw16(src) support shift.");
        if (ai2d_dtype_.dst_format != ai2d_format::NCHW_FMT
            && ai2d_dtype_.dst_format != ai2d_format::RAW16)
            throw std::runtime_error("Only Raw16/NCHW(dst) support shift.");
    }

    return ok();
}

std::unique_ptr<ai2d_builder> ai2d_builder::create(dims_t &input_shape, dims_t &output_shape,
    ai2d_datatype_t &ai2d_dtype, ai2d_crop_param_t &crop_param, ai2d_shift_param_t &shift_param,
    ai2d_pad_param_t &pad_param, ai2d_resize_param_t &resize_param, ai2d_affine_param_t &affine_param)
{
    return std::make_unique<ai2d_builder>(input_shape, output_shape, ai2d_dtype, crop_param,
        shift_param, pad_param, resize_param, affine_param);
}


result<void> ai2d_builder::build_schedule()
{
    // Local copies of the shapes: the NCHW->NCHW channel-broadcast case patches in_shape[1].
    dims_t in_shape = input_shape_;
    dims_t out_shape = output_shape_;

    bool broadcast_in_channel = false;
    if (ai2d_dtype_.src_format == ai2d_format::NCHW_FMT && ai2d_dtype_.dst_format == ai2d_format::NCHW_FMT
        && in_shape.at(1) != out_shape.at(1))
    {
        in_shape[1] = out_shape[1];
        broadcast_in_channel = true; // one input plane feeds every output channel
    }

    // ---- whole-image segments ----
    // ofmap: the padding-free interior of the output. ifmap: the crop window of the input
    // (check_config() already defaulted crop_param_ to the full image when crop is off).
    const ::nncase::padding &pad_h = pad_param_.paddings.at(2);
    const ::nncase::padding &pad_w = pad_param_.paddings.at(3);

    const int32_t out_n = static_cast<int32_t>(out_shape.at(0));
    const int32_t out_c = static_cast<int32_t>(out_shape.at(1));
    tensor4d_segment ofmap {
        { 0, out_n, out_n },
        { 0, out_c, out_c },
        { pad_h.before, output_h_ - pad_h.after, output_h_ - (pad_h.before + pad_h.after) },
        { pad_w.before, output_w_ - pad_w.after, output_w_ - (pad_w.before + pad_w.after) },
    };

    const int32_t in_n = static_cast<int32_t>(in_shape.at(0));
    const int32_t in_c = static_cast<int32_t>(in_shape.at(1));
    tensor4d_segment ifmap {
        { 0, in_n, in_n },
        { 0, in_c, in_c },
        { crop_param_.start_y, crop_param_.start_y + crop_param_.height, crop_param_.height },
        { crop_param_.start_x, crop_param_.start_x + crop_param_.width, crop_param_.width },
    };

    if (ai2d_dtype_.dst_format == ai2d_format::RGB_packed)
    {
        // NHWC layout: channel count is shape[3].
        // NOTE (faithful to binary): the ifmap override is ALSO keyed on dst_format, not src_format.
        const int32_t oc = static_cast<int32_t>(out_shape.at(3));
        const int32_t ic = static_cast<int32_t>(in_shape.at(3));
        ofmap.dim_1 = { 0, oc, oc };
        ifmap.dim_1 = { 0, ic, ic };
    }
    if (ai2d_dtype_.src_format <= ai2d_format::YUV420_I420)
        ifmap.dim_1 = { 0, 3, 3 };
    if (ai2d_dtype_.dst_format <= ai2d_format::YUV420_I420)
        ofmap.dim_1 = { 0, 3, 3 };

    // "gnne" shapes: N from the real shape, C/H/W from the format-normalized members
    // computed in check_config(). These (not input_shape_/output_shape_) go to the utils.
    dims_t in_gnne_shape { in_shape.at(0), static_cast<uint64_t>(input_c_),
        static_cast<uint64_t>(input_h_), static_cast<uint64_t>(input_w_) };
    dims_t out_gnne_shape { out_shape.at(0), static_cast<uint64_t>(output_c_),
        static_cast<uint64_t>(output_h_), static_cast<uint64_t>(output_w_) };

    // ---- static configuration ----
    config_ = ai2d_config();
    ai2d_utils utils; // ai2d_sram = { 256, 65536 }

    utils.update_static_param(config_, in_gnne_shape, out_gnne_shape, ai2d_dtype_, crop_param_,
        shift_param_, pad_param_, resize_param_, affine_param_);
    utils.update_M_param(config_, ifmap, ofmap, ai2d_dtype_, resize_param_, affine_param_);

    std::vector<int32_t> sram_ret(2, 0); // { dst_max_h, dst_max_w }
    if (affine_param_.affine_flag)
        utils.affine_sram_search(config_, ofmap, ifmap, pad_param_, sram_ret);
    else
        utils.resize_sram_search(config_, ofmap, ifmap, sram_ret);

    // ---- output tiling: N x C x H x W ----
    const int32_t c_step = (ai2d_dtype_.src_format == ai2d_format::RAW16) ? 2 : 4;

    auto n_segs = get_segment_start_end_length(0, 1, static_cast<int32_t>(out_shape.at(0)));
    auto c_segs = get_segment_start_end_length(0, c_step, static_cast<int32_t>(out_gnne_shape.at(1)));
    auto h_segs = get_segment_start_end_length(0, sram_ret[0],
        output_h_ - (pad_param_.paddings.at(2).before + pad_param_.paddings.at(2).after));
    auto w_segs = get_segment_start_end_length(0, sram_ret[1],
        output_w_ - (pad_param_.paddings.at(3).before + pad_param_.paddings.at(3).after));

    split_pos_.push_back(0);

    const bool is_yuv420 = ai2d_dtype_.src_format <= ai2d_format::YUV420_I420;
    const int32_t in_w = ifmap.dim_3.length;
    const int32_t in_h = ifmap.dim_2.length;

    auto to_float = [](uint32_t u) { FP32 m; m.u = u; return m.f; };

    // Register bytes queued since the last interrupt. update_regs() emits 36 regs (144 B)
    // when write_all, else only regs 28..35 (32 B). A batch is capped at 1024 B.
    int32_t batch_bytes = 0;

    for (auto &n_seg : n_segs)
    {
        for (auto &c_seg : c_segs)
        {
            bool write_all = true; // first tile of every (n, c) block rewrites all registers

            for (auto &h_seg : h_segs)
            {
                for (auto &w_seg : w_segs)
                {
                    // inclusive corners of this output tile
                    std::vector<float> tl { static_cast<float>(w_seg.start),   static_cast<float>(h_seg.start) };
                    std::vector<float> tr { static_cast<float>(w_seg.end - 1), static_cast<float>(h_seg.start) };
                    std::vector<float> bl { static_cast<float>(w_seg.start),   static_cast<float>(h_seg.end - 1) };
                    std::vector<float> br { static_cast<float>(w_seg.end - 1), static_cast<float>(h_seg.end - 1) };

                    // update_dynamic_param() accumulates into M2/M5, so the base matrix is
                    // rebuilt and re-decoded for every tile.
                    utils.update_M_param(config_, ifmap, ofmap, ai2d_dtype_, resize_param_, affine_param_);
                    std::vector<float> M_scale {
                        to_float(config_.M0) * (1.0f / 1024.0f), to_float(config_.M1) * (1.0f / 1024.0f),
                        to_float(config_.M3) * (1.0f / 1024.0f), to_float(config_.M4) * (1.0f / 1024.0f) };
                    std::vector<float> M_bias {
                        to_float(config_.M2) * (1.0f / 1024.0f), to_float(config_.M5) * (1.0f / 1024.0f) };

                    tensor4d_segment ifmap_sram;
                    tensor4d_segment ofmap_sram;
                    int32_t src_x = 0, src_y = 0;
                    float offset_M2 = 0.0f, offset_M5 = 0.0f;

                    if (!affine_param_.affine_flag)
                    {
                        // ---- resize / plain copy: axis-aligned, two corners suffice ----
                        auto in_tl = utils.M_mul_add(M_scale, M_bias, tl);
                        auto in_br = utils.M_mul_add(M_scale, M_bias, br);

                        int32_t x0 = std::max(0, static_cast<int32_t>(std::floor(in_tl[0])));
                        int32_t y0 = std::max(0, static_cast<int32_t>(std::floor(in_tl[1])));
                        if (is_yuv420)
                        {
                            x0 &= ~1;
                            y0 &= ~1;
                        }

                        int32_t w = std::min(in_w - 1, static_cast<int32_t>(std::ceil(in_br[0]))) - x0 + 1;
                        int32_t h = std::min(in_h - 1, static_cast<int32_t>(std::ceil(in_br[1]))) - y0 + 1;

                        if (is_yuv420)
                        {
                            if (w % 2 == 1)
                            {
                                w = std::min(in_w - 1, static_cast<int32_t>(std::ceil(in_br[0] + 1.0f))) - x0 + 1;
                                if (w % 2 == 1)
                                    throw std::runtime_error("YUV420 width error.");
                            }
                            if (h % 2 == 1)
                            {
                                h = std::min(in_h - 1, static_cast<int32_t>(std::ceil(in_br[1] + 1.0f))) - y0 + 1;
                                if (h % 2 == 1)
                                    throw std::runtime_error("YUV420 height error.");
                            }
                        }

                        // re-origin the bias to this tile; origin == current bias
                        std::vector<float> zero { 0.0f, 0.0f };
                        auto origin = utils.M_mul_add(M_scale, M_bias, zero);
                        if (x0 != 0)
                            offset_M2 = (in_tl[0] - static_cast<float>(x0)) - origin[0];
                        if (y0 != 0)
                            offset_M5 = (in_tl[1] - static_cast<float>(y0)) - origin[1];

                        ifmap_sram = { n_seg, c_seg, { y0, y0 + h, h }, { x0, x0 + w, w } };
                        ofmap_sram = { n_seg, c_seg, h_seg, w_seg };
                        src_x = x0;
                        src_y = y0;
                    }
                    else if (h_segs.size() == 1 && w_segs.size() == 1)
                    {
                        // ---- affine, whole image in one tile: no re-origin at all ----
                        // NOTE (faithful): uses the full ifmap/ofmap, including their full
                        // channel ranges, even if c_segs has more than one segment.
                        ifmap_sram = ifmap;
                        ofmap_sram = ofmap;
                    }
                    else
                    {
                        // ---- affine, tiled: bounding box of all four mapped corners ----
                        auto in_tl = utils.M_mul_add(M_scale, M_bias, tl);
                        auto in_tr = utils.M_mul_add(M_scale, M_bias, tr);
                        auto in_bl = utils.M_mul_add(M_scale, M_bias, bl);
                        auto in_br = utils.M_mul_add(M_scale, M_bias, br);

                        std::vector<float> xs { in_tl[0], in_tr[0], in_bl[0], in_br[0] };
                        std::vector<float> ys { in_tl[1], in_tr[1], in_bl[1], in_br[1] };
                        std::vector<float> min_pt { *std::min_element(xs.begin(), xs.end()),
                                                    *std::min_element(ys.begin(), ys.end()) };
                        std::vector<float> max_pt { *std::max_element(xs.begin(), xs.end()),
                                                    *std::max_element(ys.begin(), ys.end()) };

                        int32_t x0 = std::min(in_w - 1, std::max(0, static_cast<int32_t>(std::floor(min_pt[0]))));
                        int32_t y0 = std::min(in_h - 1, std::max(0, static_cast<int32_t>(std::floor(min_pt[1]))));
                        if (is_yuv420)
                        {
                            if (x0 % 2 == 1) --x0;
                            if (y0 % 2 == 1) --y0;
                        }

                        int32_t x1 = std::min(in_w - 1, std::max(0, static_cast<int32_t>(std::ceil(max_pt[0]))));
                        int32_t y1 = std::min(in_h - 1, std::max(0, static_cast<int32_t>(std::ceil(max_pt[1]))));

                        int32_t w = x1 - x0 + 1;
                        int32_t h = y1 - y0 + 1;
                        if (w == 0) w = 1;
                        if (h == 0) h = 1;

                        if (is_yuv420)
                        {
                            if (w % 2 == 1)
                            {
                                x1 = std::min(x1 + 1, in_w - 1);
                                w = x1 - x0 + 1;
                                if (w % 2 == 1)
                                    throw std::runtime_error("YUV420 width error.");
                            }
                            if (h % 2 == 1)
                            {
                                y1 = std::min(y1 + 1, in_h - 1);
                                h = y1 - y0 + 1;
                                if (h % 2 == 1)
                                    throw std::runtime_error("YUV420 height error.");
                            }
                        }

                        // bias is re-origined from the mapped TOP-LEFT corner (not the bbox
                        // minimum), and unlike the resize path this is not gated on x0/y0 != 0
                        std::vector<float> zero { 0.0f, 0.0f };
                        auto origin = utils.M_mul_add(M_scale, M_bias, zero);
                        offset_M2 = (in_tl[0] - static_cast<float>(x0)) - origin[0];
                        offset_M5 = (in_tl[1] - static_cast<float>(y0)) - origin[1];

                        ifmap_sram = { n_seg, c_seg, { y0, y0 + h, h }, { x0, x0 + w, w } };
                        ofmap_sram = { n_seg, c_seg, h_seg, w_seg };
                        src_x = x0;
                        src_y = y0;
                    }

                    // tile coordinates above are crop-relative; the register wants absolute
                    if (crop_param_.crop_flag)
                    {
                        src_x += crop_param_.start_x;
                        src_y += crop_param_.start_y;
                    }

                    utils.update_dynamic_param(config_, src_x, src_y, ai2d_dtype_, pad_param_,
                        ifmap_sram, ofmap_sram, ifmap, ofmap, offset_M2, offset_M5,
                        in_gnne_shape, out_gnne_shape, broadcast_in_channel);

                    // ---- batching: raise an interrupt (intr_mask = 0) only when the batch
                    //      would overflow 1024 B with the next tile, or on the very last tile ----
                    batch_bytes += write_all ? 144 : 32;

                    const bool last_h = h_seg == h_segs.back();
                    const bool last_w = w_seg == w_segs.back();
                    const bool last_c = c_seg == c_segs.back();

                    // NOTE (faithful): when this is the last tile of the last c-segment but
                    // not of the last n-segment, the next tile's 144 B are not counted.
                    const int32_t next_bytes = (last_h && last_w) ? (last_c ? 0 : 144) : 32;

                    if (batch_bytes + next_bytes > 1024)
                    {
                        config_.intr_mask = 0;
                        batch_bytes = 0;
                    }
                    else
                    {
                        const bool last_tile = n_seg == n_segs.back() && last_c && last_h && last_w;
                        config_.intr_mask = last_tile ? 0 : 1;
                    }

                    utils.update_regs(config_, write_all, regs_);

                    if (config_.intr_mask == 0)
                        split_pos_.push_back(static_cast<uint32_t>(regs_.size()));

                    if (dump_asm_)
                        dump_asm(write_all); // argument not recoverable (callee ignores it)

                    write_all = false;
                }
            }
        }
    }

    if (dump_asm_) {
        auto res = dump_gmodel();
        assert(res.is_ok());
    }

    return ok();
}

END_NS_NNCASE_F_K230
