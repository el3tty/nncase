// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.F;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class StackToConcat : IRewriteRule
{
    public IPattern Pattern { get; } = Nncase.PatternMatch.F.Tensors.IsStack("stack", "stackCall",
        Nncase.PatternMatch.Utility.IsTuple("tuple"), Nncase.PatternMatch.Utility.IsTensorConst("axis"));

    private Expr? GetReplace(Expr stack, Call stackCall, Expr tuple, int axis)
    {
        if (axis != 0 || stackCall.CheckedShape.Rank <= 2)
        {
            return null;
        }

        return Nncase.IR.F.Tensors.Reshape(Nncase.IR.F.Tensors.Concat(tuple, axis), stackCall.CheckedShape);
    }

    public Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Expr stack = (Expr)result["stack"];
        Call stackCall = (Call)result["stackCall"];
        Expr tupleExpr = (Expr)result["tuple"];
        int axis = ((TensorConst)result["axis"]).Value.ToScalar<int>();
        return GetReplace(stack, stackCall, tupleExpr, axis);
    }
}
