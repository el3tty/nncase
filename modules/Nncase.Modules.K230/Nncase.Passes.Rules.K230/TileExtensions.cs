// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.TIR;

namespace Nncase.Passes.Rules.K230;

public static class TileExtensions
{
    public static int FixedDimensions(this Buffer b, int axis)
    {
        return ((TensorConst)b.Dimensions[axis]).Value.ToScalar<int>();
    }

    public static TensorConst Const(this Buffer b)
    {
        return (TensorConst)((Call)b.MemSpan.Start)[DDrOf.Input];
    }
}
