// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.NN;
using Nncase.PatternMatch;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class SplitLargeGlobalReduceWindow2D : RewriteRule<Pattern>
{
    public override Pattern Pattern
    {
        get
        {
            OpPattern<ReduceWindow2D> targetPattern =
                Nncase.PatternMatch.Utility.IsOp<ReduceWindow2D>("reduceWindow2D");
            (ParameterInfo, Pattern)[] inputPatterns = new (ParameterInfo, Pattern)[8]
            {
                (ReduceWindow2D.Input,
                    Nncase.PatternMatch.Utility.IsWildcard("input")with
                    {
                        TypePattern = (TypePatternUtility.HasFixedShape() & TypePatternUtility.HasRank(4))
                    }),
                (ReduceWindow2D.InitValue, Nncase.PatternMatch.Utility.IsTensorConst("initValue")),
                (ReduceWindow2D.Filter, Nncase.PatternMatch.Utility.IsTensorConst("filter")),
                (ReduceWindow2D.Stride, Nncase.PatternMatch.Utility.IsTensorConst("strides")),
                (ReduceWindow2D.Padding, Nncase.PatternMatch.Utility.IsTensorConst("padding")),
                (ReduceWindow2D.Dilation, Nncase.PatternMatch.Utility.IsTensorConst("dilation")),
                (ReduceWindow2D.CeilMode, Nncase.PatternMatch.Utility.IsTensorConst("ceilMode")),
                (ReduceWindow2D.CountIncludePad, Nncase.PatternMatch.Utility.IsTensorConst("countIncludePad"))
            };
            return Nncase.PatternMatch.Utility.IsCallSpecific("call", targetPattern, inputPatterns)with
            {
                TypePattern = (TypePatternUtility.HasFixedShape() & TypePatternUtility.HasRank(4))
            };
        }
    }

    public Expr? GetReplace(Call call, ReduceWindow2D reduceWindow2D, Expr initValue, Expr input, int[] filter,
        int[] padding)
    {
        int[] inputShape = input.CheckedShape.ToValueArray();
        int[] outputShape = call.CheckedShape.ToValueArray();
        int outH = outputShape[2];
        int outW = outputShape[3];
        if (inputShape[^2] == filter[0] && inputShape[^1] == filter[1] && padding.All((int p) => p == 0) && outH == 1 &&
            outW == 1 && filter[0] >= 320 && filter[1] >= 320)
        {
            int sliceCount = ((inputShape[^2] < 640) ? 4 : 10);
            int sliceHeight = inputShape[^2] / sliceCount;
            Expr[] slices = new Expr[sliceCount];
            Expr[] pooledSlices = new Expr[sliceCount];
            for (int sliceIndex = 0; sliceIndex < sliceCount; sliceIndex++)
            {
                slices[sliceIndex] = Tensors.Slice(input, new int[1] { sliceIndex * sliceHeight }, new int[1] { sliceIndex * sliceHeight + sliceHeight },
                    new int[1] { 2 }, new int[1] { 1 });
                pooledSlices[sliceIndex] = NN.ReduceWindow2D(reduceWindow2D.ReduceOp, slices[sliceIndex], initValue,
                    new int[2] { sliceHeight, filter[1] }, new int[2] { 1, 1 }, new int[2, 2], new int[2] { 1, 1 }, false,
                    false);
            }

            Call input2 = Tensors.Concat(new Tuple(pooledSlices), 2);
            return NN.ReduceWindow2D(reduceWindow2D.ReduceOp, input2, initValue, new int[2] { sliceCount, 1 },
                new int[2] { 1, 1 }, new int[2, 2], new int[2] { 1, 1 }, false, false);
        }

        return null;
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Call call = (Call)__result["call"];
        ReduceWindow2D reduceWindow2D = (ReduceWindow2D)__result["reduceWindow2D"];
        Expr initValue = (Expr)__result["initValue"];
        Expr input = (Expr)__result["input"];
        int[] filter = ((TensorConst)__result["filter"]).Value.ToArray<int>();
        int[] padding = ((TensorConst)__result["padding"]).Value.ToArray<int>();
        return GetReplace(call, reduceWindow2D, initValue, input, filter, padding);
    }
}
