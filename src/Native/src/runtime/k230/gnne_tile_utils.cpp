#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cstring>

#include <nncase/runtime/small_vector.hpp>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/k230/gnne_tile_utils.h>
#include "k230_common.h"

BEGIN_NS_NNCASE_RT_K230
using namespace nncase;

int32_t get_bytes_from_type(nncase::typecode_t type)
{
    switch (type)
    {
    case typecode_t::dt_int8:
    case typecode_t::dt_uint8:
        return 1;

    case typecode_t::dt_int16:
    case typecode_t::dt_uint16:
    case typecode_t::dt_float16:
    case typecode_t::dt_bfloat16:
        return 2;

    case typecode_t::dt_int32:
    case typecode_t::dt_uint32:
    case typecode_t::dt_float32:
        return 4;

    // dt_int64, dt_uint64, dt_float64 (and anything else)
    // fall through to the assert below
    default:
        assert(false && "unknown datatype");
    }

    return 0; // unreachable
}

int32_t get_size_from_shape(dims_t shape)
{
    size_t result = 1;
    for (auto dim : shape)
    {
        result *= dim;
    }
    return (int32_t)result;
}


namespace
{
    // Only needed for one spot below: writing a 16-bit pad value across two adjacent
    // 8-bit bitfields (const_pad_chN / const_pad_ch(N+1)) the same way the compiled
    // code does with a raw WORD store, which a plain bitfield assignment can't express
    // (a bitfield assignment truncates instead of spilling into the next field).
    struct raw_view
    {
        uint8_t *p;
        explicit raw_view(ai2d_config &c) : p(reinterpret_cast<uint8_t *>(&c)) { }
        void set_u16(size_t off, uint16_t v) { std::memcpy(p + off, &v, sizeof(v)); }
    };
}

void ai2d_utils::update_static_param(
    ai2d_config &config, dims_t &in_shape, dims_t &out_shape,
    ai2d_datatype_t &ai2d_dtype, ai2d_crop_param_t &crop_param, ai2d_shift_param_t &shift_param,
    ai2d_pad_param_t &pad_param, ai2d_resize_param_t &resize_param, ai2d_affine_param_t &affine_param)
{
    // shape[1] is read as the channel count, for an [N, C, H, W]-style shape.
    const uint64_t channels = in_shape.at(1); // throws std::out_of_range if in_shape.size() <= 1
    config.channel = static_cast<uint32_t>(channels) & 0x7u;

    const ai2d_format src_format = ai2d_dtype.src_format;
    ai2d_format dst_format = ai2d_dtype.dst_format; // may be forced to NCHW_FMT below

    uint32_t dst_channel = config.channel; // fallback used only by the `default:` case
    uint32_t dst_format_bits;

    auto pack_dst_channel_and_format = [&]() {
        dst_format_bits = static_cast<uint32_t>(dst_format) & 0xFu;
        config.dst_channel = dst_channel;
    };
    auto force_nchw_output = [&]() {
        config.csc_en = 1; // converting a YUV source to planar NCHW needs colour-space conversion
        config.dst_channel = 3;
        dst_format_bits = 3;
        dst_format = ai2d_format::NCHW_FMT;
    };

    switch (src_format)
    {
    case ai2d_format::YUV420_NV12:
    case ai2d_format::YUV420_NV21:
        dst_channel = 2; // 1 luma plane + 1 interleaved-chroma plane
        config.csc_en = 0;
        if (dst_format != ai2d_format::NCHW_FMT) pack_dst_channel_and_format();
        else                                      force_nchw_output();
        break;

    case ai2d_format::YUV420_I420:
        dst_channel = 3; // 3 separate planes (Y, U, V)
        config.csc_en = 0;
        if (dst_format != ai2d_format::NCHW_FMT) pack_dst_channel_and_format();
        else                                      force_nchw_output();
        break;

    case ai2d_format::RGB_packed:
        dst_channel = 1; // 1 interleaved plane
        config.csc_en = 0;
        if (dst_format != ai2d_format::NCHW_FMT)
        {
            pack_dst_channel_and_format();
        }
        else
        {
            // same target bits as force_nchw_output(), but csc_en is left cleared here
            config.dst_channel = 3;
            dst_format_bits = 3;
            dst_format = ai2d_format::NCHW_FMT;
        }
        break;

    default: // NCHW_FMT or RAW16 source
        config.csc_en = 0;
        if (dst_format == ai2d_format::NCHW_FMT)
        {
            config.dst_channel = dst_channel; // keeps the original `channel` value
            dst_format_bits = 3;
        }
        else
        {
            pack_dst_channel_and_format();
        }
        break;
    }

    // ---- crop / shift / pad / resize / affine "enabled" flags ----
    const bool affine_en = affine_param.affine_flag;
    const bool resize_en = resize_param.resize_flag;
    const bool pad_en    = pad_param.pad_flag;
    const bool crop_en   = crop_param.crop_flag;
    const bool shift_en  = shift_param.shift_flag;

    // NOTE: the disassembly also writes config.cord_round=2 here when
    // (!affine_en && !resize_en && (crop_en||shift_en||pad_en)); it is unconditionally
    // overwritten by the resize-driven assignment below and has no observable effect,
    // so it's omitted (see caveats).

    // resize interpolation mode -> interpolation / cord_round.
    // NOTE: this relies on resize_param and affine_param being mutually exclusive in
    // practice (whichever mode is inactive is left at its default-constructed values) --
    // see caveats.
    const bool resize_is_bilinear =
        (static_cast<uint32_t>(resize_param.interp_method) & ~0x2u) == 1; // tf_bilinear or cv2_bilinear
    if (resize_is_bilinear)
    {
        config.interpolation = 1;
        config.cord_round = 0;
    }
    else
    {
        config.interpolation = 0;
        config.cord_round = (resize_param.interp_mode != ai2d_interp_mode::align_corner) ? 2 : 0;
    }

    config.dst_format = dst_format_bits;
    config.src_format = static_cast<uint32_t>(src_format);

    config.src_ind = ai2d_dtype.src_loc == ai2d_data_loc::ddr;
    config.dst_ind = ai2d_dtype.dst_loc == ai2d_data_loc::ddr;

    const int32_t shift_val = shift_param.shift_val;
    uint8_t shift_byte = static_cast<uint8_t>(shift_val);
    if (shift_val < 0)
        shift_byte = static_cast<uint8_t>(shift_val + 32); // wrap into [0,31]
    config.shift = shift_byte;

    // sign = 0 only for unsigned 8/16-bit source data; 1 for everything else
    // (int8/16, float16/bf16, and wider types the AI2D pixel path doesn't otherwise touch)
    config.sign = (ai2d_dtype.src_type != typecode_t::dt_uint8
        && ai2d_dtype.src_type != typecode_t::dt_uint16) ? 1u : 0u;

    if (affine_en)
    {
        config.interpolation = resize_is_bilinear ? 1u : 0u; // re-derived from the same check above
        config.cord_round = affine_param.cord_round & 0x3u;
        config.bound_ind = affine_param.bound_ind & 0xFu;
        config.bound_val = static_cast<uint16_t>(affine_param.bound_val);
        config.bound_smooth = affine_param.bound_smooth & 0x1u;
    }

    if (resize_en)
    {
        config.pad_mod = 0; // ai2d_pad_mode::constant
    }

    if (pad_en)
    {
        config.pad_mod = static_cast<uint32_t>(pad_param.pad_mode);
    }

    // ---- per-channel constant pad values (only meaningful when pad_mode == constant) ----
    if (config.pad_mod == 0)
    {
        raw_view cfg(config);

        if (dst_format == ai2d_format::RAW16)
        {
            // 16-bit values, each spilling from one const_pad_chN field into the next
            cfg.set_u16(99, static_cast<uint16_t>(pad_param.pad_val.at(0))); // -> ch0/ch1
            if (channels > 1)
                cfg.set_u16(101, static_cast<uint16_t>(pad_param.pad_val.at(1))); // -> ch2/ch3
        }
        else
        {
            config.const_pad_ch0 = static_cast<uint8_t>(pad_param.pad_val.at(0));
            if (channels > 1)
            {
                config.const_pad_ch1 = static_cast<uint8_t>(pad_param.pad_val.at(1));
                if (channels != 2)
                {
                    config.const_pad_ch2 = static_cast<uint8_t>(pad_param.pad_val.at(2));
                    if (channels != 3)
                        config.const_pad_ch3 = static_cast<uint8_t>(pad_param.pad_val.at(3));
                }
            }
        }
    }

    // ---- input-side per-plane row widths, keyed on src_format, from in_shape[3] ----
    if (src_format == ai2d_format::YUV420_I420)
    {
        const uint64_t w = in_shape.at(3); // throws if in_shape.size() <= 3
        const uint16_t half = static_cast<uint16_t>(w >> 1);
        config.src_ch0_width_layout = static_cast<uint16_t>(w);
        config.src_ch1_width_layout = half;
        config.src_ch2_width_layout = half;
    }
    else if (src_format == ai2d_format::RGB_packed)
    {
        config.src_ch0_width_layout = static_cast<uint16_t>(3 * in_shape.at(3));
    }
    else if (src_format == ai2d_format::RAW16)
    {
        const uint16_t doubled = static_cast<uint16_t>(2 * in_shape.at(3));
        config.src_ch0_width_layout = doubled;
        config.src_ch1_width_layout = doubled;
        config.src_ch2_width_layout = doubled;
        config.src_ch3_width_layout = doubled;
    }
    else // YUV420_NV12 / YUV420_NV21 / NCHW_FMT
    {
        const uint16_t w = static_cast<uint16_t>(in_shape.at(3));
        config.src_ch0_width_layout = w;
        config.src_ch1_width_layout = w;
        config.src_ch2_width_layout = w;
        config.src_ch3_width_layout = w;
    }

    // ---- output-side per-plane row widths, keyed on `dst_format` (post force-to-NCHW), from out_shape[3] ----
    if (dst_format == ai2d_format::YUV420_I420)
    {
        const uint64_t w = out_shape.at(3);
        const uint16_t half = static_cast<uint16_t>(w >> 1);
        config.dst_ch0_width_layout = static_cast<uint16_t>(w);
        config.dst_ch1_width_layout = half;
        config.dst_ch2_width_layout = half;
    }
    else if (dst_format == ai2d_format::RGB_packed)
    {
        config.dst_ch0_width_layout = static_cast<uint16_t>(3 * out_shape.at(3));
    }
    else if (dst_format == ai2d_format::RAW16)
    {
        const uint16_t doubled = static_cast<uint16_t>(2 * out_shape.at(3));
        config.dst_ch0_width_layout = doubled;
        config.dst_ch1_width_layout = doubled;
        config.dst_ch2_width_layout = doubled;
        config.dst_ch3_width_layout = doubled;
    }
    else
    {
        const uint16_t w = static_cast<uint16_t>(out_shape.at(3));
        config.dst_ch0_width_layout = w;
        config.dst_ch1_width_layout = w;
        config.dst_ch2_width_layout = w;
        config.dst_ch3_width_layout = w;
    }
}

void ai2d_utils::update_dynamic_param(
    ai2d_config &config, int32_t src_x, int32_t src_y, ai2d_datatype_t &ai2d_dtype,
    ai2d_pad_param_t &pad_param, tensor4d_segment &ifmap_sram, tensor4d_segment &ofmap_sram,
    tensor4d_segment &ifmap, tensor4d_segment &ofmap, float offset_M2, float offset_M5,
    dims_t &input_shape, dims_t &output_shape, bool broadcast_in_channel)
{
    // ---- channel-count refinement for NCHW/RAW16 formats ----
    if (ai2d_dtype.src_format == ai2d_format::NCHW_FMT || ai2d_dtype.src_format == ai2d_format::RAW16)
    {
        const uint32_t v = static_cast<uint32_t>(ifmap_sram.dim_1.length) & 0x7u;
        config.channel = v;
        config.dst_channel = v;
    }

    // ---- SRAM-tile-derived shape/position fields ----
    const int32_t ofmap_h_len = ofmap_sram.dim_2.length;
    const int32_t ifmap_h_len = ifmap_sram.dim_2.length;
    const int32_t ifmap_w_len = ifmap_sram.dim_3.length;
    config.dst_height_shape = static_cast<uint32_t>(ofmap_h_len) & 0x3FFFu;

    const int32_t ofmap_w_len   = ofmap_sram.dim_3.length;
    const int32_t ofmap_w_start = ofmap_sram.dim_3.start;
    config.src_height_shape = static_cast<uint16_t>(ifmap_h_len);
    config.src_width_shape  = static_cast<uint16_t>(ifmap_w_len);
    config.src_x = static_cast<uint16_t>(src_x);
    config.src_y = static_cast<uint16_t>(src_y);
    config.dst_width_shape  = static_cast<uint16_t>(ofmap_w_len);

    // ---- config.dst_x / config.dst_y ----
    const int32_t ofmap_h_start = ofmap_sram.dim_2.start;
    uint16_t dst_x_val = 0;
    uint32_t dst_y_val = 0;

    if (ofmap_w_start != 0)
        dst_x_val = static_cast<uint16_t>(pad_param.paddings.at(3).before + ofmap_w_start);

    if (ofmap_h_start != 0)
    {
        const uint16_t v55 = static_cast<uint16_t>(ofmap_h_start);
        dst_y_val = (static_cast<uint32_t>(pad_param.paddings.at(2).before) + v55) & 0x1FFFu;
    }

    config.dst_x = dst_x_val;
    config.dst_y = dst_y_val & 0x1FFFu;

    // ---- config.pad_t / pad_b / pad_l / pad_r, clipped against the SRAM tile boundaries ----
    config.pad_t = 0;
    config.pad_b = 0;
    config.pad_l = 0;
    config.pad_r = 0;

    if (pad_param.pad_flag)
    {
        const int32_t out_h = static_cast<int32_t>(output_shape.at(2)); // throws if size() <= 2
        const int32_t pad_h_before = pad_param.paddings.at(2).before;
        const int32_t pad_h_after  = pad_param.paddings.at(2).after;
        const int32_t cropped_h = out_h - (pad_h_after + pad_h_before);

        const int32_t out_w = static_cast<int32_t>(output_shape.at(3)); // throws if size() <= 3
        const int32_t pad_w_before = pad_param.paddings.at(3).before;
        const int32_t pad_w_after  = pad_param.paddings.at(3).after;
        const int32_t cropped_w = out_w - (pad_w_after + pad_w_before);

        auto maybe_pad_b = [&]() {
            if (ofmap_sram.dim_2.end == cropped_h)
                config.pad_b = static_cast<uint16_t>(pad_h_after);
        };
        auto warn_if_needed = [&]() {
            if (static_cast<uint16_t>(ofmap_w_len) <= 0x20 && config.pad_l != 0)
                std::cout << "[Warn]: left padding is not supported when width <= 32!, may cause hardware panic!" << std::endl;
        };

        if (ofmap_h_len == cropped_h)
        {
            // this tile spans the full (padding-cropped) height: pad_t/pad_b pass straight through
            config.pad_t = static_cast<uint16_t>(pad_h_before);
            config.pad_b = static_cast<uint16_t>(pad_h_after);

            if (ofmap_w_len == cropped_w)
            {
                config.pad_l = static_cast<uint16_t>(pad_w_before);
                config.pad_r = static_cast<uint16_t>(pad_w_after);
            }
            else if (ofmap_w_start == 0)
            {
                config.pad_l = static_cast<uint16_t>(pad_w_before);
            }
            else if (ofmap_sram.dim_3.end == cropped_w)
            {
                config.pad_r = static_cast<uint16_t>(pad_w_after);
            }
            // (no warning check on this path)
        }
        else
        {
            // this tile is a partial height slice: figure out which edge(s) it actually touches
            if (ofmap_w_len == cropped_w) // touches both left and right
            {
                config.pad_l = static_cast<uint16_t>(pad_w_before);
                config.pad_r = static_cast<uint16_t>(pad_w_after);
                if (ofmap_h_start != 0) maybe_pad_b();
                else                    config.pad_t = static_cast<uint16_t>(pad_h_before);
                warn_if_needed();
            }
            else if (ofmap_w_start == 0) // touches only the left edge
            {
                config.pad_l = static_cast<uint16_t>(pad_w_before);
                if (ofmap_h_start != 0) maybe_pad_b();
                else                    config.pad_t = static_cast<uint16_t>(pad_h_before);
                warn_if_needed();
            }
            else if (ofmap_sram.dim_3.end == cropped_w) // touches only the right edge
            {
                config.pad_r = static_cast<uint16_t>(pad_w_after);
                if (ofmap_h_start != 0) maybe_pad_b();
                else                    config.pad_t = static_cast<uint16_t>(pad_h_before);
                // (no warning check on this path)
            }
            else if (ofmap_h_start != 0) // touches neither left nor right edge
            {
                maybe_pad_b();
                // (no warning check, no pad_t/pad_l/pad_r change on this path)
            }
            else
            {
                config.pad_t = static_cast<uint16_t>(pad_h_before);
                // (no warning check on this path)
            }
        }
    }

    // ---- source-buffer per-channel-plane pointer/stride computation ----
    const int32_t in_h = static_cast<int32_t>(input_shape.at(2)); // throws if size() <= 2
    const int32_t in_w = static_cast<int32_t>(input_shape.at(3)); // throws if size() <= 3
    const int32_t src_bytes_per_elem = get_bytes_from_type(ai2d_dtype.src_type);

    const int32_t plane_bytes = broadcast_in_channel ? 0 : (in_h * in_w * src_bytes_per_elem);
    const int32_t channel_tile_offset_src = ifmap_sram.dim_1.start - ifmap.dim_1.start;

    const int32_t src_ch0 = channel_tile_offset_src * plane_bytes;
    const int32_t src_ch1 = src_ch0 + plane_bytes;
    int32_t src_ch2 = src_ch0 + plane_bytes + plane_bytes;

    if (ai2d_dtype.src_format == ai2d_format::YUV420_I420)
    {
        const int32_t quarter = plane_bytes >> 2; // 4:2:0 chroma planes are quarter-size
        src_ch2 = quarter + src_ch0 + plane_bytes; // NOTE: uses full (non-quartered) plane_bytes here too
    }

    config.src_ch0_ptr = static_cast<uint32_t>(src_ch0);
    config.src_ch1_ptr = static_cast<uint32_t>(src_ch1);
    config.src_ch3_ptr = static_cast<uint32_t>(plane_bytes + src_ch2); // also full plane_bytes, not quartered
    config.src_ch2_ptr = static_cast<uint32_t>(src_ch2);

    // ---- destination-buffer per-channel-plane pointer/stride computation ----
    const int32_t out_hw = static_cast<int32_t>(output_shape.at(2) * output_shape.at(3));
    const int32_t dst_bytes_per_elem = get_bytes_from_type(ai2d_dtype.dst_type);
    const int32_t channel_tile_offset_dst = ofmap_sram.dim_1.start - ofmap.dim_1.start;

    const int32_t dst_plane_bytes = dst_bytes_per_elem * out_hw;
    const int32_t dst_ch0 = dst_plane_bytes * channel_tile_offset_dst;
    const int32_t dst_ch1 = dst_plane_bytes + dst_ch0;
    int32_t dst_ch2 = dst_plane_bytes + dst_ch1;

    if (ai2d_dtype.dst_format == ai2d_format::YUV420_I420)
    {
        const int32_t quarter = dst_plane_bytes >> 2;
        dst_ch2 = quarter + dst_ch1;
    }

    config.dst_ch3_ptr = static_cast<uint32_t>(dst_plane_bytes + dst_ch2);
    config.dst_ch2_ptr = static_cast<uint32_t>(dst_ch2);
    config.dst_ch1_ptr = static_cast<uint32_t>(dst_ch1);
    config.dst_ch0_ptr = static_cast<uint32_t>(dst_ch0);

    // ---- per-channel constant pad values, indexed by this tile's destination-channel offset ----
    if (pad_param.pad_flag && pad_param.pad_mode == ai2d_pad_mode::constant)
    {
        const int32_t channel_count = ofmap_sram.dim_1.length;
        const size_t idx0 = static_cast<size_t>(channel_tile_offset_dst);
        const int32_t v0 = pad_param.pad_val.at(idx0);

        if (ai2d_dtype.dst_format == ai2d_format::RAW16)
        {
            config.const_pad_ch0 = static_cast<uint16_t>(v0);
            if (channel_count > 1)
                config.const_pad_ch2 = static_cast<uint16_t>(pad_param.pad_val.at(idx0 + 1));
        }
        else
        {
            config.const_pad_ch0 = static_cast<uint8_t>(v0);
            if (channel_count > 1)
            {
                config.const_pad_ch1 = static_cast<uint8_t>(pad_param.pad_val.at(idx0 + 1));
                if (channel_count != 2)
                {
                    config.const_pad_ch2 = static_cast<uint8_t>(pad_param.pad_val.at(idx0 + 2));
                    if (channel_count != 3)
                        config.const_pad_ch3 = static_cast<uint8_t>(pad_param.pad_val.at(idx0 + 3));
                }
            }
        }
    }

    // ---- accumulate the affine translation offsets (M2, M5), scaled to the config's fixed-point format ----
    FP32 m2; m2.u = config.M2;
    FP32 m5; m5.u = config.M5;
    FP32 new_m2; new_m2.f = offset_M2 * 1024.0f + m2.f;
    FP32 new_m5; new_m5.f = offset_M5 * 1024.0f + m5.f;
    config.M2 = new_m2.u;
    config.M5 = new_m5.u;
}

void ai2d_utils::inv_M(std::vector<float> &M_ori_scale, std::vector<float> &M_ori_bias,
                        std::vector<float> &M_inv_scale, std::vector<float> &M_inv_bias)
{
    const float *ori_scale = M_ori_scale.data();
    const float d = ori_scale[3];

    // NOTE: corrected from the decompiled pseudocode, which literally printed
    //   ori_scale[0]*d - ori_scale[0]*d     (always 0.0)
    // That has to be a Hex-Rays rendering glitch: the assert message explicitly
    // says "det != 0", and every line below is exactly the standard 2x2 matrix
    // inverse ([[a,b],[c,d]]^-1 = (1/det)*[[d,-b],[-c,a]]) using this determinant.
    const float det = ori_scale[0] * d - ori_scale[1] * ori_scale[2];

    assert(det != 0.0f && "det != 0");

    float *inv_scale = M_inv_scale.data();
    const float *ori_bias = M_ori_bias.data();
    float *inv_bias = M_inv_bias.data();

    inv_scale[0] = d / det;
    inv_scale[3] = ori_scale[0] / det;
    inv_scale[1] = -ori_scale[1] / det;
    inv_scale[2] = -ori_scale[2] / det;

    inv_bias[0] = inv_scale[1] * ori_bias[1] - inv_scale[0] * ori_bias[0];
    inv_bias[1] = inv_scale[3] * ori_bias[1] - inv_scale[2] * ori_bias[0];
}

std::vector<float> ai2d_utils::M_mul_add(std::vector<float> &M_scale, std::vector<float> &M_bias,
                                          std::vector<float> &v_i)
{
    std::vector<float> result(2);

    // result = M * v_i + M_bias, where M = [[M_scale[0], M_scale[1]],
    //                                        [M_scale[2], M_scale[3]]]
    result[0] = M_bias[0] + (M_scale[0] * v_i[0] + M_scale[1] * v_i[1]);
    result[1] = M_bias[1] + (M_scale[2] * v_i[0] + M_scale[3] * v_i[1]);

    return result;
}

std::vector<segment> get_segment_start_end_length(int32_t start, int32_t chunk_size, int32_t upper_bound)
{
    std::vector<segment> result;

    for (int32_t pos = start; pos < upper_bound; pos += chunk_size)
    {
        int32_t end = pos + chunk_size;
        if (end > upper_bound)
            end = upper_bound;

        result.emplace_back(pos, end, end - pos);
    }

    return result;
}

bool ai2d_utils::try_allocate_resize_sram(
    std::vector<float> &M_ori_scale, std::vector<float> &M_ori_bias,
    int32_t dst_max_h, int32_t dst_max_w,
    tensor4d_segment &ifmap, tensor4d_segment &ofmap, ai2d_format &src_format)
{
    // candidate output tiles: chop the ofmap's H/W ranges into chunks no larger than dst_max_h/dst_max_w
    const auto h_segments = get_segment_start_end_length(0, dst_max_h, ofmap.dim_2.length);
    const auto w_segments = get_segment_start_end_length(0, dst_max_w, ofmap.dim_3.length);

    const bool is_yuv420 = static_cast<int32_t>(src_format) <= 2; // NV12 / NV21 / I420

    for (const auto &h_seg : h_segments)
    {
        const float y0f = static_cast<float>(h_seg.start);
        const float y1f = static_cast<float>(h_seg.end - 1);

        for (const auto &w_seg : w_segments)
        {
            std::vector<float> top_left     = {static_cast<float>(w_seg.start), y0f};
            std::vector<float> bottom_right = {static_cast<float>(w_seg.end - 1), y1f};

            // map both corners of this output tile back into input-image coordinates
            std::vector<float> in_top_left     = M_mul_add(M_ori_scale, M_ori_bias, top_left);
            std::vector<float> in_bottom_right = M_mul_add(M_ori_scale, M_ori_bias, bottom_right);

            int32_t in_x0 = static_cast<int32_t>(std::floor(in_top_left[0]));
            if (in_x0 < 0) in_x0 = 0;
            int32_t in_y0 = static_cast<int32_t>(std::floor(in_top_left[1]));
            if (in_y0 < 0) in_y0 = 0;

            if (is_yuv420) // even-align the top-left corner for 2x2 chroma subsampling
            {
                in_x0 &= ~1;
                in_y0 &= ~1;
            }

            int32_t in_x1 = ifmap.dim_3.length - 1;
            const int32_t ceil_x1 = static_cast<int32_t>(std::ceil(in_bottom_right[0]));
            if (ceil_x1 < in_x1) in_x1 = ceil_x1;

            int32_t in_y1 = ifmap.dim_2.length - 1;
            const int32_t ceil_y1 = static_cast<int32_t>(std::ceil(in_bottom_right[1]));
            if (ceil_y1 < in_y1) in_y1 = ceil_y1;

            int32_t in_w = in_x1 - in_x0 + 1;
            int32_t in_h = in_y1 - in_y0 + 1;

            if (is_yuv420)
            {
                // width/height must be even for YUV420; if not, try growing the region by one
                // pixel (still clamped to the input bounds) before giving up
                if (in_w & 1)
                {
                    int32_t x1_retry = ifmap.dim_3.length - 1;
                    const int32_t ceil_x1_retry = static_cast<int32_t>(std::ceil(in_bottom_right[0] + 1.0f));
                    if (ceil_x1_retry < x1_retry) x1_retry = ceil_x1_retry;
                    in_w = x1_retry - in_x0 + 1;
                    if (in_w & 1)
                        throw std::runtime_error("YUV420 width error.");
                }
                if (in_h & 1)
                {
                    int32_t y1_retry = ifmap.dim_2.length - 1;
                    const int32_t ceil_y1_retry = static_cast<int32_t>(std::ceil(in_bottom_right[1] + 1.0f));
                    if (ceil_y1_retry < y1_retry) y1_retry = ceil_y1_retry;
                    in_h = y1_retry - in_y0 + 1;
                    if (in_h & 1)
                        throw std::runtime_error("YUV420 height error.");
                }
            }

            const int32_t required_sram = in_w * in_h;
            if (ai2d_sram.sram_size < required_sram)
                return false; // this tile's needed input region doesn't fit -- caller should shrink dst_max_h/w
        }
    }

    return true; // every candidate tile's input region fits
}

void ai2d_utils::resize_sram_search(ai2d_config &config, tensor4d_segment &ofmap, tensor4d_segment &ifmap,
                                     std::vector<int32_t> &ret)
{
    constexpr float FIXED_POINT_SCALE = 1.0f / 1024.0f; // 0.0009765625

    // config.M0/M1/M3/M4 (scale) and M2/M5 (bias) are stored in the fixed-point format
    // update_dynamic_param writes; undo that to recover the plain-float output->input transform.
    FP32 m0; m0.u = config.M0;
    FP32 m1; m1.u = config.M1;
    FP32 m3; m3.u = config.M3;
    FP32 m4; m4.u = config.M4;
    FP32 m2; m2.u = config.M2;
    FP32 m5; m5.u = config.M5;

    std::vector<float> M_ori_scale = { FIXED_POINT_SCALE * m0.f, FIXED_POINT_SCALE * m1.f,
                                        FIXED_POINT_SCALE * m3.f, FIXED_POINT_SCALE * m4.f };
    std::vector<float> M_ori_bias = { FIXED_POINT_SCALE * m2.f, FIXED_POINT_SCALE * m5.f };

    std::vector<float> M_inv_scale(4), M_inv_bias(2);
    inv_M(M_ori_scale, M_ori_bias, M_inv_scale, M_inv_bias);

    const int32_t W_in = ifmap.dim_3.length;
    const int32_t H_in = ifmap.dim_2.length;
    const int32_t sram_len  = ai2d_sram.sram_len;
    const int32_t sram_size = ai2d_sram.sram_size;

    // pick an initial input-space tile size that (at most) fills the SRAM budget
    int32_t W0, H0;
    if (sram_len < W_in)
    {
        if (sram_len >= H_in)
        {
            if (sram_size / H_in >= W_in) // whole width fits alongside the full height -- no tiling needed
            {
                ret[0] = ofmap.dim_2.length;
                ret[1] = ofmap.dim_3.length;
                return;
            }
            W0 = sram_size / H_in;
            H0 = H_in;
        }
        else // sram_len covers neither dimension
        {
            W0 = sram_len;
            H0 = sram_len;
        }
    }
    else
    {
        H0 = sram_size / W_in;
        if (H0 >= H_in) // no tiling needed
        {
            ret[0] = ofmap.dim_2.length;
            ret[1] = ofmap.dim_3.length;
            return;
        }
        W0 = W_in;
    }

    // map that candidate input-space tile's corners through the INVERSE transform to see
    // what output-space tile size it corresponds to -- this seeds the search below
    std::vector<float> origin       = { 0.0f, 0.0f };
    std::vector<float> input_corner = { static_cast<float>(W0 - 1), static_cast<float>(H0 - 1) };
    std::vector<float> out_origin = M_mul_add(M_inv_scale, M_inv_bias, origin);
    std::vector<float> out_corner = M_mul_add(M_inv_scale, M_inv_bias, input_corner);

    int32_t dst_max_w = static_cast<int32_t>(std::floor(out_corner[0]) - std::ceil(out_origin[0]) + 1.0f);
    int32_t dst_max_h = static_cast<int32_t>(std::floor(out_corner[1]) - std::ceil(out_origin[1]) + 1.0f);

    const bool is_yuv420 = static_cast<uint32_t>(config.src_format) <= 2; // NV12 / NV21 / I420
    int32_t step;

    if (is_yuv420)
    {
        if (dst_max_h % 2 == 1) dst_max_h -= 1; // keep both dimensions even for chroma subsampling
        if (dst_max_w % 2 == 1) dst_max_w -= 1;
        if (dst_max_h <= 0)
        {
            ret[0] = dst_max_h;
            ret[1] = dst_max_w;
            return;
        }
        step = 2;
    }
    else
    {
        step = 1;
        if (dst_max_h <= 0)
        {
            ret[0] = dst_max_h;
            ret[1] = dst_max_w;
            return;
        }
    }

    // sram_len vs. W_in/H_in is invariant across the whole search, so the shrink strategy
    // below is fixed for the entire loop -- the disassembly re-derives it every iteration,
    // but hoisting it out doesn't change behavior since none of those inputs change.
    const bool sram_covers_width  = sram_len >= W_in;
    const bool sram_covers_height = sram_len >= H_in;
    bool shrink_width_next = false;

    while (dst_max_w > 0)
    {
        ai2d_format fmt = static_cast<ai2d_format>(config.src_format);
        if (try_allocate_resize_sram(M_ori_scale, M_ori_bias, dst_max_h, dst_max_w, ifmap, ofmap, fmt))
            break; // found a tile size that fits

        if (sram_covers_width) // sram_len >= W_in: always shrink height
        {
            dst_max_h -= step;
            if (dst_max_h <= 0)
                break;
            continue;
        }
        if (sram_covers_height) // sram_len < W_in, but >= H_in: always shrink width
        {
            dst_max_w -= step;
            continue;
        }

        // sram_len covers neither dimension: alternate shrinking width/height, but avoid
        // shrinking a dimension that's already down to 1 on two consecutive turns
        if (shrink_width_next)
        {
            if (dst_max_w != 1)
            {
                dst_max_w -= step;
                shrink_width_next = false;
            }
            else
            {
                dst_max_h -= step;
                if (dst_max_h <= 0)
                    break;
                shrink_width_next = false;
            }
        }
        else
        {
            if (dst_max_h == 1)
            {
                dst_max_w -= step;
                shrink_width_next = true;
            }
            else
            {
                dst_max_h -= step;
                if (dst_max_h == 0)
                    break;
                shrink_width_next = true;
            }
        }
    }

    ret[0] = dst_max_h;
    ret[1] = dst_max_w;
}

bool ai2d_utils::try_allocate_affine_sram(std::vector<float> &M_ori_scale, std::vector<float> &M_ori_bias,
                                           segment &output_h, segment &output_w)
{
    // four corners of the output tile, mapped through the forward (output->input) affine
    // transform. Three of the four are nudged inward by one pixel on their w/h edge while
    // the fourth (bottom-right) uses the raw `end` coordinates for both -- that asymmetry
    // is exactly what the disassembly does; preserved as-is rather than "symmetrized".
    std::vector<float> corner_a = { static_cast<float>(output_w.start + 1), static_cast<float>(output_h.start + 1) };
    std::vector<float> corner_b = { static_cast<float>(output_w.end),       static_cast<float>(output_h.start + 1) };
    std::vector<float> corner_c = { static_cast<float>(output_w.start + 1), static_cast<float>(output_h.end) };
    std::vector<float> corner_d = { static_cast<float>(output_w.end),       static_cast<float>(output_h.end) };

    std::vector<float> in_a = M_mul_add(M_ori_scale, M_ori_bias, corner_a);
    std::vector<float> in_b = M_mul_add(M_ori_scale, M_ori_bias, corner_b);
    std::vector<float> in_c = M_mul_add(M_ori_scale, M_ori_bias, corner_c);
    std::vector<float> in_d = M_mul_add(M_ori_scale, M_ori_bias, corner_d);

    std::vector<float> xs = { in_a[0], in_b[0], in_c[0], in_d[0] };
    std::vector<float> ys = { in_a[1], in_b[1], in_c[1], in_d[1] };

    const float min_x = *std::min_element(xs.begin(), xs.end());
    const float min_y = *std::min_element(ys.begin(), ys.end());
    const float max_x = *std::max_element(xs.begin(), xs.end());
    const float max_y = *std::max_element(ys.begin(), ys.end());

    // required input-space footprint: the usual inclusive-range "+1", plus a 2-pixel margin
    // (likely to cover interpolation's neighboring-pixel reads near the tile edge)
    const int32_t width  = static_cast<int32_t>((std::ceil(max_x) - std::floor(min_x)) + 3.0f);
    const int32_t height = static_cast<int32_t>((std::ceil(max_y) - std::floor(min_y)) + 3.0f);

    return ai2d_sram.sram_size >= width * height;
}

void ai2d_utils::update_M_param(ai2d_config &config, tensor4d_segment &ifmap, tensor4d_segment &ofmap,
                                 ai2d_datatype_t &ai2d_dtype, ai2d_resize_param_t &resize_param,
                                 ai2d_affine_param_t &affine_param)
{
    (void)ai2d_dtype; // unused in this function -- present only to match the class's declared signature

    if (resize_param.resize_flag)
    {
        dims_t in_shape  = ifmap.to_gnne_shape();
        dims_t out_shape = ofmap.to_gnne_shape();

        const uint64_t in_h  = in_shape.at(2);  // throws std::out_of_range if size() <= 2
        const uint64_t in_w  = in_shape.at(3);  // throws if size() <= 3
        const uint64_t out_h = out_shape.at(2);
        const uint64_t out_w = out_shape.at(3);

        float width_scale  = static_cast<float>(in_w) / static_cast<float>(out_w);
        float height_scale = static_cast<float>(in_h) / static_cast<float>(out_h);
        float width_bias  = 0.0f;
        float height_bias = 0.0f;

        // config.interpolation was already set by update_static_param earlier in this
        // configuration pass, so this reads it back rather than re-deriving it here.
        if (config.interpolation != 0) // bilinear (tf_bilinear or cv2_bilinear)
        {
            if (resize_param.interp_method == ai2d_interp_method::cv2_bilinear
                || resize_param.interp_mode == ai2d_interp_mode::half_pixel)
            {
                // half-pixel-center convention: out = scale*(x+0.5) - 0.5
                width_bias  = width_scale * 0.5f - 0.5f;
                height_bias = height_scale * 0.5f - 0.5f;
            }
            else if (resize_param.interp_mode == ai2d_interp_mode::align_corner)
            {
                if (out_w > 1) width_scale  = static_cast<float>(in_w - 1) / static_cast<float>(out_w - 1);
                if (out_h > 1) height_scale = static_cast<float>(in_h - 1) / static_cast<float>(out_h - 1);
                // bias stays 0
            }
            // else (interp_mode::none): plain in/out ratio, bias stays 0
        }
        else // nearest-neighbor
        {
            if (resize_param.interp_mode == ai2d_interp_mode::half_pixel)
            {
                width_bias  = width_scale * 0.5f;
                height_bias = height_scale * 0.5f;
            }
            else if (resize_param.interp_mode == ai2d_interp_mode::align_corner)
            {
                if (out_w > 1) width_scale  = static_cast<float>(in_w - 1) / static_cast<float>(out_w - 1);
                if (out_h > 1) height_scale = static_cast<float>(in_h - 1) / static_cast<float>(out_h - 1);
            }
        }

        FP32 m0, m4, m2, m5;
        m0.f = width_scale  * 1024.0f;
        m4.f = height_scale * 1024.0f;
        m2.f = width_bias   * 1024.0f;
        m5.f = height_bias  * 1024.0f;
        config.M0 = m0.u;
        config.M4 = m4.u;
        config.M2 = m2.u;
        config.M5 = m5.u;
        // NOTE: M1/M3 (the off-diagonal/shear terms) are left untouched here -- resize is
        // axis-aligned, so it only ever needs the diagonal scale terms.
    }
    else if (affine_param.affine_flag)
    {
        // affine_param.M is the caller-supplied forward (input->output) 2x3 matrix,
        // laid out as {scale0, scale1, bias0, scale2, scale3, bias1}. config.M needs the
        // inverse (output->input) direction, matching how the rest of ai2d_utils uses it.
        std::vector<float> M_ori_scale = { affine_param.M[0], affine_param.M[1],
                                            affine_param.M[3], affine_param.M[4] };
        std::vector<float> M_ori_bias  = { affine_param.M[2], affine_param.M[5] };
        std::vector<float> M_inv_scale(4, 0.0f), M_inv_bias(2, 0.0f);

        inv_M(M_ori_scale, M_ori_bias, M_inv_scale, M_inv_bias);

        FP32 m0, m1, m3, m4, m2, m5;
        m0.f = M_inv_scale[0] * 1024.0f;
        m1.f = M_inv_scale[1] * 1024.0f;
        m3.f = M_inv_scale[2] * 1024.0f;
        m4.f = M_inv_scale[3] * 1024.0f;
        m2.f = M_inv_bias[0]  * 1024.0f;
        m5.f = M_inv_bias[1]  * 1024.0f;
        config.M0 = m0.u;
        config.M1 = m1.u;
        config.M3 = m3.u;
        config.M4 = m4.u;
        config.M2 = m2.u;
        config.M5 = m5.u;
    }
    // else: neither resize nor affine active -- config.M left untouched
}

void ai2d_utils::affine_sram_search(ai2d_config &config, tensor4d_segment &ofmap, tensor4d_segment &ifmap,
                                     ai2d_pad_param_t &pad_param, std::vector<int32_t> &ret)
{
    (void)pad_param; // unused in this function

    if (ai2d_sram.sram_size >= ifmap.dim_3.length * ifmap.dim_2.length)
    {
        // whole ifmap fits in SRAM at once -- gets overwritten below unless the search
        // that follows never finds a single working candidate (in which case this stands
        // as the fallback), so this write is not dead code despite looking that way
        ret[0] = ofmap.dim_2.length;
        ret[1] = ofmap.dim_3.length;
    }

    // undo the fixed-point (x1024) scaling on config's affine coefficients
    FP32 m0, m1, m3, m4, m2, m5;
    m0.u = config.M0; m1.u = config.M1; m3.u = config.M3;
    m4.u = config.M4; m2.u = config.M2; m5.u = config.M5;
    std::vector<float> M_ori_scale = { m0.f / 1024.0f, m1.f / 1024.0f, m3.f / 1024.0f, m4.f / 1024.0f };
    std::vector<float> M_ori_bias  = { m2.f / 1024.0f, m5.f / 1024.0f };

    // NOTE: this fixed 8-value step sequence is inferred from a raw memcpy of an unnamed
    // read-only constant (`C_42_0` in the disassembly) -- its exact bytes weren't
    // recoverable, but {128,64,...,1} is the overwhelmingly likely value: it's the classic
    // binary-search halving sequence, its size (8 x int32 = 32 bytes) matches exactly, and
    // it converges any position in [1,255] starting from sram_len/2 = 128 in exactly 8 steps.
    static const int32_t step_sizes[8] = { 128, 64, 32, 16, 8, 4, 2, 1 };

    const int32_t ofmap_h = ofmap.dim_2.length;
    const int32_t ifmap_h = ifmap.dim_2.length;
    const int32_t ofmap_w = ofmap.dim_3.length;
    const int32_t ifmap_w = ifmap.dim_3.length;

    const float h_ratio = static_cast<float>(ofmap_h) / static_cast<float>(ifmap_h);
    const float w_ratio = static_cast<float>(ofmap_w) / static_cast<float>(ifmap_w);

    int32_t search_pos = ai2d_sram.sram_len / 2; // 128 by default
    int32_t best_h = 2, best_w = 2;              // fallback if nothing ever succeeds

    const bool is_yuv420 = static_cast<uint32_t>(config.src_format) <= 2;

    for (int32_t step : step_sizes)
    {
        // scale the current 1D search position into an output-space tile size,
        // clamped so it never exceeds the ofmap's own dimensions
        const float h_input = (search_pos >= ifmap_h) ? static_cast<float>(ifmap_h) : static_cast<float>(search_pos);
        int32_t tile_h = std::min(ofmap_h, static_cast<int32_t>(std::floor(h_input * h_ratio)));

        const float w_input = (search_pos < ifmap_w) ? static_cast<float>(search_pos) : static_cast<float>(ifmap_w);
        int32_t tile_w = std::min(ofmap_w, static_cast<int32_t>(std::floor(w_input * w_ratio)));

        if (is_yuv420)
        {
            if (tile_h & 1) --tile_h;
            if (tile_w & 1) --tile_w;
        }

        auto h_segments = get_segment_start_end_length(0, tile_h, ofmap_h);
        auto w_segments = get_segment_start_end_length(0, tile_w, ofmap_w);

        // only the first resulting segment is tested -- used as a representative sample
        // rather than checking every tile, presumably because the binary search on
        // search_pos already converges toward the right tile size directly
        if (try_allocate_affine_sram(M_ori_scale, M_ori_bias, h_segments[0], w_segments[0]))
        {
            search_pos += step;
            best_h = tile_h;
            best_w = tile_w;
        }
        else
        {
            search_pos -= step;
        }
    }

    ret[0] = best_h;
    ret[1] = best_w;
}

void ai2d_utils::update_regs(ai2d_config &config, bool write_all, std::vector<std::vector<uint32_t>> &regs)
{
    // ai2d_config is 36 uint32_t "registers" wide (indices 0..35); write_all covers all of
    // them, while a partial update only rewrites the last 8 (indices 28..35) -- presumably
    // the M-matrix / per-tile-position registers that actually change between tiles, since
    // the static format/mode registers earlier in the range don't need re-sending each tile.
    uint32_t idx = write_all ? 0 : 28;

    do
    {
        std::vector<uint32_t> entry(5);
        entry[0] = 4 * idx; // byte address of this 4-register group within the register file
        entry[1] = config.get_addr_value(idx + 3);
        entry[2] = config.get_addr_value(idx + 2);
        entry[3] = config.get_addr_value(idx + 1);
        entry[4] = config.get_addr_value(idx + 0);

        regs.push_back(std::move(entry));

        idx += 4;
    }
    while (idx != 36);
}

END_NS_NNCASE_RT_K230
