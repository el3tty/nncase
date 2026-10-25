// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.K230.F;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class ToFakePdpReduce : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = NN.IsReduceWindow2D("r", "call", (ReduceWindow2D _) => true,
        Nncase.PatternMatch.Utility.IsWildcard("input"), Nncase.PatternMatch.Utility.IsTensorConst(),
        Nncase.PatternMatch.Utility.IsTensorConst("filter"), Nncase.PatternMatch.Utility.IsTensorConst("stride"),
        Nncase.PatternMatch.Utility.IsTensorConst("padding"), Nncase.PatternMatch.Utility.IsTensorConst("dilation"),
        Nncase.PatternMatch.Utility.IsTensorConst("ceilMode"),
        Nncase.PatternMatch.Utility.IsTensorConst("countIncludePad"));

    private Expr? GetReplace(Call call, ReduceWindow2D r, Expr input, int[] filter, int[] stride, int[] padding,
        bool ceilMode, Expr countIncludePad)
    {
        // FakePdp always produces float32 (FakePdpEvaluator): an f16 window reduce (e.g. ReduceMax over H and W of an
        // f16 tensor) is run on float32 and cast back, otherwise the rewritten expression would change its type. Other
        // types are left alone (the CPU reduce_window2d kernel only handles float32 either).
        DataType dataType = input.CheckedDataType;
        if (r.ReduceOp == ReduceOp.Prod || (dataType != DataTypes.Float32 && dataType != DataTypes.Float16))
        {
            return null;
        }

        if ((filter[0] != input.CheckedShape.ToValueArray()[2] || filter[1] != input.CheckedShape.ToValueArray()[3]) &&
            (filter[0] * filter[1] > 256 || filter[0] > 64 || filter[1] > 64))
        {
            return null;
        }

        int[,] paddings = new int[2, 2] { { padding[0], padding[1] }, { padding[2], padding[3] } };
        if (ceilMode)
        {
            Func<int, int, int, int, int> func = (int outSize, int k, int s, int p) => (outSize - 1) * s + k - p;
            int extraPadH = func(call.CheckedShape[2].FixedValue, filter[0], stride[0], paddings[0, 0] + paddings[0, 1]) -
                      input.CheckedShape[2].FixedValue;
            int extraPadW = func(call.CheckedShape[3].FixedValue, filter[1], stride[1], paddings[1, 0] + paddings[1, 1]) -
                       input.CheckedShape[3].FixedValue;
            paddings[0, 1] = (extraPadH > 0) ? (paddings[0, 1] + extraPadH) : paddings[0, 1];
            paddings[1, 1] = (extraPadW > 0) ? (paddings[1, 1] + extraPadW) : paddings[1, 1];
        }

        if (dataType == DataTypes.Float32)
        {
            return Nncase.IR.K230.F.Tensors.FakePdp(r.ReduceOp, input, 0, filter, stride, paddings, countIncludePad);
        }

        Expr pdp = Nncase.IR.K230.F.Tensors.FakePdp(r.ReduceOp, Nncase.IR.F.Tensors.Cast(input, DataTypes.Float32), 0,
            filter, stride, paddings, countIncludePad);
        return Nncase.IR.F.Tensors.Cast(pdp, dataType);
    }

    private float GetPadValue(ReduceOp reduceOp)
    {
        return reduceOp switch
        {
            ReduceOp.Mean => float.NaN,
            ReduceOp.Min => float.MaxValue,
            ReduceOp.Max => float.MinValue,
            ReduceOp.Sum => float.NaN,
            _ => throw new ArgumentOutOfRangeException("reduceOp", reduceOp, null),
        };
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Call call = (Call)result["call"];
        ReduceWindow2D r = (ReduceWindow2D)result["r"];
        Expr input = (Expr)result["input"];
        int[] filter = ((TensorConst)result["filter"]).Value.ToArray<int>();
        int[] stride = ((TensorConst)result["stride"]).Value.ToArray<int>();
        int[] padding = ((TensorConst)result["padding"]).Value.ToArray<int>();
        bool ceilMode = ((TensorConst)result["ceilMode"]).Value.ToScalar<bool>();
        Expr countIncludePad = (Expr)result["countIncludePad"];
        return GetReplace(call, r, input, filter, stride, padding, ceilMode, countIncludePad);
    }
}
