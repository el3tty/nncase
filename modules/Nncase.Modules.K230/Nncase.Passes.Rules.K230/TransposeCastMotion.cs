// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class TransposeCastMotion : IRewriteRule
{
    public IPattern Pattern { get; } = Nncase.PatternMatch.F.Tensors.IsCast("cast", "castCall", (Cast _) => true,
        Nncase.PatternMatch.F.Tensors.IsTranspose("tp", "tpCall", (Transpose _) => true,
            Nncase.PatternMatch.Utility.IsWildcard("input") with
            {
                TypePattern = TypePatternUtility.HasDataType(DataTypes.Boolean),
            }, Nncase.PatternMatch.Utility.IsTensorConst("perm")));

    private Expr? GetReplace(Expr input, Expr perm, Cast cast, Call castCall)
    {
        return Nncase.IR.F.Tensors.Transpose(
            Nncase.IR.F.Tensors.Cast(input, castCall.CheckedDataType, cast.CastMode),
            perm);
    }

    public Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Expr input = (Expr)result["input"];
        Expr perm = (Expr)result["perm"];
        Cast cast = (Cast)result["cast"];
        Call castCall = (Call)result["castCall"];
        return GetReplace(input, perm, cast, castCall);
    }
}
