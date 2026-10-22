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
                    Nncase.PatternMatch.Utility.IsWildcard("input") with
                    {
                        TypePattern = TypePatternUtility.HasFixedShape(),
                    }),
                (Conv2DTranspose.Weights, Nncase.PatternMatch.Utility.IsWildcard("weights")),
                (Conv2DTranspose.Stride, Nncase.PatternMatch.Utility.IsTensorConst("strides")),
                (Conv2DTranspose.Padding, Nncase.PatternMatch.Utility.IsTensorConst("paddings")),
                (Conv2DTranspose.Dilation, Nncase.PatternMatch.Utility.IsTensorConst("dilation")),
                (Conv2DTranspose.OutputPadding, Nncase.PatternMatch.Utility.IsTensorConst("outputPaddings")),
            };
            return Nncase.PatternMatch.Utility.IsCallSpecific("call", targetPattern, inputPatterns) with
            {
                TypePattern = TypePatternUtility.HasFixedShape(),
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
                (int Count, int ChunkSize, int CurrentSize, int Index, int Axis) chunkInfo = param;
                int chunkCount = chunkInfo.Count;
                int chunkSize = chunkInfo.ChunkSize;
                int currentSize = chunkInfo.CurrentSize;
                int chunkIndex = chunkInfo.Index;
                int axisIndex = chunkInfo.Axis;
                int dimIndex = axisIndex - 2;
                padding = (chunkIndex == chunkCount - 1)
                    ? new Padding(paddingsValue[dimIndex * 2], paddingsValue[dimIndex * 2 + 1])
                    : new Padding(paddingsValue[dimIndex * 2], 0);
                chunkSize /= strides[dimIndex];
                currentSize /= strides[dimIndex];
                return (NewBegin: chunkSize * chunkIndex, NewEnd: Math.Min(chunkSize * chunkIndex + currentSize, outShape[axisIndex]),
                    Pads: padding);
            }, delegate(Expr slice, Padding newPaddings1, int axis)
            {
                int[] outputShape = outShape.ToArray();
                int fixedValue = slice.CheckedShape[axis].FixedValue;
                outputShape[axis] = fixedValue * strides[axis - 2];
                int[] newPaddings = paddings.ToArray();
                int dimIndex = axis - 2;
                newPaddings[2 * dimIndex] = newPaddings1.Before;
                newPaddings[2 * dimIndex + 1] = newPaddings1.After;
                return ReplaceUtility.ReplaceCallParams(call.Target, call.Arguments.ToArray(),
                    (Conv2DTranspose.Input, slice), (Conv2DTranspose.Padding, Tensor.From(newPaddings, paddings.Shape)),
                    (Conv2DTranspose.OutputShape, outputShape));
            });
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Call call = (Call)result["call"];
        Expr input = (Expr)result["input"];
        Expr weights = (Expr)result["weights"];
        Tensor<int> paddings = ((TensorConst)result["paddings"]).Value.Cast<int>();
        Tensor<int> outputPaddings = ((TensorConst)result["outputPaddings"]).Value.Cast<int>();
        int[] strides = ((TensorConst)result["strides"]).Value.ToArray<int>();
        int[] dilation = ((TensorConst)result["dilation"]).Value.ToArray<int>();
        return GetReplace(call, input, weights, paddings, outputPaddings, strides, dilation);
    }
}
