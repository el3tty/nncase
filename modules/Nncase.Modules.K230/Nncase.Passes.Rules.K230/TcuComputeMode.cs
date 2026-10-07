// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Rules.K230;

public enum TcuComputeMode
{
    NormalConv2d,
    DwConv2d,
    TransposeConv2d,
    MatMul,
    Activation
}
