// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.TIR.Instructions;

public enum MFU_CONF_FUNCTION : uint
{
    transpose_conf,
    pdp1_conf1,
    pdp1_conf2,
    pdp1_conf3,
    pdp1_conf4,
    pdp1_conf_op_in,
    pdp1_conf_deq,
    pdp1_conf_quant,
    act1_conf_stride,
    act1_conf_src1,
    act1_conf_src2,
    act1_conf_dest,
    act1_conf_deq,
    act1_conf_quant,
    act1_conf
}
