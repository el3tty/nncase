// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc.ImTools;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class SplitLargeTranspose : RewriteRule<Pattern>
{
    public override Pattern Pattern => Tensors.IsTranspose("tr", "call",
        Nncase.PatternMatch.Utility.IsWildcard("input") with { TypePattern = TypePatternUtility.HasFixedShape() },
        Nncase.PatternMatch.Utility.IsTensorConst("perm"));

    public Expr? GetReplace(Call call, Expr input, Tensor<int> perm)
    {
        int[] splitShape = input.CheckedShape.ToValueArray();
        return SplitLarge.Split(call, input, splitShape, ((int Dim, int Axis) _) => true, (int _) => 1,
            (int axis) => ArrayTools.IndexOf<int>(perm.ToArray(), axis),
            delegate((int Count, int ChunkSize, int CurrentSize, int Index, int Axis) param)
            {
                (int Count, int ChunkSize, int CurrentSize, int Index, int Axis) chunkInfo = param;
                int chunkSize = chunkInfo.ChunkSize;
                int currentSize = chunkInfo.CurrentSize;
                int chunkIndex = chunkInfo.Index;
                return (NewBegin: chunkIndex * chunkSize, NewEnd: chunkIndex * chunkSize + currentSize, Pads: Padding.Zero());
            }, delegate(Expr slice, Padding padding, int i)
            {
                slice.InferenceType();
                return ReplaceUtility.ReplaceCallFirstParam(call.Target, call.Arguments.ToArray(), slice);
            });
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Call call = (Call)result["call"];
        Expr input = (Expr)result["input"];
        Tensor<int> perm = ((TensorConst)result["perm"]).Value.Cast<int>();
        return GetReplace(call, input, perm);
    }
}
