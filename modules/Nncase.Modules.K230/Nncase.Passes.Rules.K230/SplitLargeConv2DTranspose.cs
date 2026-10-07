// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class SplitLargeConv2DTranspose : RewriteRule<Pattern>
{
    public override Pattern Pattern
    {
        get
        {
            OpPattern<Conv2DTranspose> targetPattern = Nncase.PatternMatch.Utility.IsOp<Conv2DTranspose>();
            (ParameterInfo, Pattern)[] inputPatterns = new (ParameterInfo, Pattern)[6]
            {
                (Conv2DTranspose.Input,
                    Nncase.PatternMatch.Utility.IsWildcard("input")with
                    {
                        TypePattern = TypePatternUtility.HasFixedShape()
                    }),
                (Conv2DTranspose.Weights, Nncase.PatternMatch.Utility.IsWildcard("weights")),
                (Conv2DTranspose.Stride, Nncase.PatternMatch.Utility.IsTensorConst("strides")),
                (Conv2DTranspose.Padding, Nncase.PatternMatch.Utility.IsTensorConst("paddings")),
                (Conv2DTranspose.Dilation, Nncase.PatternMatch.Utility.IsTensorConst("dilation")),
                (Conv2DTranspose.OutputPadding, Nncase.PatternMatch.Utility.IsTensorConst("outputPaddings"))
            };
            return Nncase.PatternMatch.Utility.IsCallSpecific("call", targetPattern, inputPatterns)with
            {
                TypePattern = TypePatternUtility.HasFixedShape()
            };
        }
    }

    public Expr? GetReplace(Call call, Expr input, Expr weights, Tensor<int> paddings, Tensor<int> outputPaddings,
        int[] strides, int[] dilation)
    {
        int[] outShape = call.CheckedShape.ToValueArray();
        input.CheckedShape.ToValueArray();
        weights.CheckedShape.ToValueArray();
        int[] paddingsValue = paddings.ToArray();
        return SplitLarge.Split(call, input, outShape, delegate((int Dim, int Axis) pair)
            {
                if (pair.Axis - 2 < 0)
                {
                    return false;
                }

                return outShape[pair.Axis] > 65535 && outputPaddings.Sum() == 0;
            }, (int axis) => strides[axis - 2], (int axis) => axis,
            delegate((int Count, int ChunkSize, int CurrentSize, int Index, int Axis) param)
            {
                Padding padding = Padding.Zero();
                (int Count, int ChunkSize, int CurrentSize, int Index, int Axis) tuple = param;
                int item = tuple.Count;
                int item2 = tuple.ChunkSize;
                int item3 = tuple.CurrentSize;
                int item4 = tuple.Index;
                int item5 = tuple.Axis;
                int num = item5 - 2;
                padding = ((item4 == item - 1)
                    ? new Padding(paddingsValue[num * 2], paddingsValue[num * 2 + 1])
                    : new Padding(paddingsValue[num * 2], 0));
                item2 /= strides[num];
                item3 /= strides[num];
                return (NewBegin: item2 * item4, NewEnd: Math.Min(item2 * item4 + item3, outShape[item5]),
                    Pads: padding);
            }, delegate(Expr slice, Padding newPaddings1, int axis)
            {
                int[] array = outShape.ToArray();
                int fixedValue = slice.CheckedShape[axis].FixedValue;
                array[axis] = fixedValue * strides[axis - 2];
                int[] array2 = paddings.ToArray();
                int num = axis - 2;
                array2[2 * num] = newPaddings1.Before;
                array2[2 * num + 1] = newPaddings1.After;
                return ReplaceUtility.ReplaceCallParams(call.Target, call.Arguments.ToArray(),
                    (Conv2DTranspose.Input, slice), (Conv2DTranspose.Padding, Tensor.From(array2, paddings.Shape)),
                    (Conv2DTranspose.OutputShape, array));
            });
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Call call = (Call)__result["call"];
        Expr input = (Expr)__result["input"];
        Expr weights = (Expr)__result["weights"];
        Tensor<int> paddings = ((TensorConst)__result["paddings"]).Value.Cast<int>();
        Tensor<int> outputPaddings = ((TensorConst)__result["outputPaddings"]).Value.Cast<int>();
        int[] strides = ((TensorConst)__result["strides"]).Value.ToArray<int>();
        int[] dilation = ((TensorConst)__result["dilation"]).Value.ToArray<int>();
        return GetReplace(call, input, weights, paddings, outputPaddings, strides, dilation);
    }
}
