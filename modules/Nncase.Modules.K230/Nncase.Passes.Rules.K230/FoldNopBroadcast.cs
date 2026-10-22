// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class FoldNopBroadcast : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = Tensors.IsBroadcast("broadcast", "call", (Broadcast _) => true,
        Nncase.PatternMatch.Utility.IsWildcard("input") with
        {
            TypePattern = TypePatternUtility.HasRank((int r) => r <= 4, "GNNE not support more than 4D") &
                           TypePatternUtility.HasFixedShape(),
        }, Nncase.PatternMatch.Utility.IsTensorConst("shape"));

    public Expr? GetReplace(Broadcast broadcast, Call call, Expr input, TensorConst shape)
    {
        if (input.CheckedShape.ToValueArray().SequenceEqual(shape.Value.ToArray<int>()))
        {
            return input;
        }

        return null;
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Broadcast broadcast = (Broadcast)result["broadcast"];
        Call call = (Call)result["call"];
        Expr input = (Expr)result["input"];
        TensorConst shape = (TensorConst)result["shape"];
        return GetReplace(broadcast, call, input, shape);
    }
}
