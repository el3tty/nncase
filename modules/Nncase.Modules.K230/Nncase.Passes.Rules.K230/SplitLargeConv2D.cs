// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class SplitLargeConv2D : RewriteRule<Pattern>
{
    public override Pattern Pattern
    {
        get
        {
            OpPattern<Conv2D> targetPattern = Nncase.PatternMatch.Utility.IsOp<Conv2D>();
            (ParameterInfo, Pattern)[] inputPatterns = new (ParameterInfo, Pattern)[5]
            {
                (Conv2D.Input,
                    Nncase.PatternMatch.Utility.IsWildcard("input") with
                    {
                        TypePattern = TypePatternUtility.HasFixedShape(),
                    }),
                (Conv2D.Weights, Nncase.PatternMatch.Utility.IsWildcard("weights")),
                (Conv2D.Stride, Nncase.PatternMatch.Utility.IsTensorConst("strides")),
                (Conv2D.Padding, Nncase.PatternMatch.Utility.IsTensorConst("paddings")),
                (Conv2D.Dilation, Nncase.PatternMatch.Utility.IsTensorConst("dilation")),
            };
            return Nncase.PatternMatch.Utility.IsCallSpecific("call", targetPattern, inputPatterns) with
            {
                TypePattern = TypePatternUtility.HasFixedShape(),
            };
        }
    }

    public Expr? GetReplace(Call call, Expr input, Expr weights, Tensor<int> paddings, int[] strides, int[] dilation)
    {
        int[] outputShape = call.CheckedShape.ToValueArray();
        int oh = outputShape[2];
        int ow = outputShape[3];
        return SplitLarge.Split(call, input, outputShape, ((int Dim, int Axis) pair) => pair.Dim > 65535, (int _) => 1,
            (int axis) => axis, delegate((int Count, int ChunkSize, int CurrentSize, int Index, int Axis) param)
            {
                (int Count, int ChunkSize, int CurrentSize, int Index, int Axis) chunkInfo = param;
                int currentSize = chunkInfo.CurrentSize;
                int chunkIndex = chunkInfo.Index;
                int axisIndex = chunkInfo.Axis;
                int dimIndex = axisIndex - 2;
                Padding p = new Padding(paddings[new int[2] { dimIndex, 0 }], paddings[new int[2] { dimIndex, 1 }]);
                List<Segment1D> outputSegments =
                    TileUtilities.GetSegmentStartEndLength(0, currentSize, input.CheckedShape[axisIndex].FixedValue);
                int h = (axisIndex == 3) ? ow : oh;
                Segment1D inputRowSegment = TileUtilities.GetInputRowSegment(
                    outputSegments[chunkIndex].Start,
                    Math.Min(currentSize, outputSegments[chunkIndex].Length), h, weights.CheckedShape[axisIndex].FixedValue,
                    strides[dimIndex], dilation[dimIndex], in p);
                return (NewBegin: inputRowSegment.Start, NewEnd: inputRowSegment.End, Pads: inputRowSegment.Padding);
            }, delegate(Expr slice, Padding padding, int axis)
            {
                Tensor<int> tensor =
                    Tensor.From(
                        Enumerable.Range(0, 2).SelectMany((int i) =>
                            (i == axis - 2) ? new int[2] { padding.Before, padding.After } : new int[2]).ToArray(),
                        new int[2] { 2, 2 });
                return ReplaceUtility.ReplaceCallParams(call.Target, call.Arguments.ToArray(), (Conv2D.Input, slice),
                    (Conv2D.Padding, tensor));
            });
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Call call = (Call)result["call"];
        Expr input = (Expr)result["input"];
        Expr weights = (Expr)result["weights"];
        Tensor<int> paddings = ((TensorConst)result["paddings"]).Value.Cast<int>();
        int[] strides = ((TensorConst)result["strides"]).Value.ToArray<int>();
        int[] dilation = ((TensorConst)result["dilation"]).Value.ToArray<int>();
        return GetReplace(call, input, weights, paddings, strides, dilation);
    }
}
