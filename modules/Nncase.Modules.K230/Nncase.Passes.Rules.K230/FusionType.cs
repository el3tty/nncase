// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Rules.K230;

public enum FusionType
{
    Conv2d,
    Transpose,
    Pdp,
    Act,
    Conv2dPdp,
    Conv2dConv2d,
    Conv2dAct1,
    LoadStore,
    Conv2dTranspose,
    Pdp0Dw,
    Pad,
    Lstm,
    L2WAllIn,
    L2IfFullWPp,
    L2IfSplitWFull,
    L2IfSplitWPp,
    Matmul,
    Ai2dPad,
    Ai2dResize
}
