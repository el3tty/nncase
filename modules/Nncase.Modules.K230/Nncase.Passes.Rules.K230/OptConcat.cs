// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class OptConcat : IRewriteRule
{
    public IPattern Pattern { get; }

    private Expr? GetReplace(IReadOnlyList<Expr> tupleInputs, Concat concat, Call concatCall)
    {
        Expr[] fields;
        if (concatCall.CheckedShape.Count == 5)
        {
            if (concatCall.CheckedShape[0].FixedValue == 1)
            {
                fields = (from i in tupleInputs.AsEnumerable()
                    select Nncase.IR.F.Tensors.Squeeze(i, new int[1])).ToArray();
                return Nncase.IR.F.Tensors.Unsqueeze(
                    Nncase.IR.F.Tensors.Concat(new Nncase.IR.Tuple(fields), concat.Axis - 1), new int[1]);
            }

            return null;
        }

        fields = (from i in tupleInputs.AsEnumerable()
                select Nncase.IR.F.Tensors.Unsqueeze(i,
                    Enumerable.Range(0, 4 - concatCall.CheckedShape.Rank).ToArray()))
            .ToArray();
        return Nncase.IR.F.Tensors.Squeeze(
            Nncase.IR.F.Tensors.Concat(new Nncase.IR.Tuple(fields), concat.Axis + 4 - concatCall.CheckedShape.Rank),
            Enumerable.Range(0, 4 - concatCall.CheckedShape.Rank).ToArray());
    }

    public Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        IReadOnlyList<Expr> tupleInputs = (IReadOnlyList<Expr>)__result["tupleInputs"];
        Concat concat = (Concat)__result["concat"];
        Call concatCall = (Call)__result["concatCall"];
        return GetReplace(tupleInputs, concat, concatCall);
    }

    public OptConcat()
    {
        Func<Concat, bool> condition = (Concat _) => true;
        Func<Pattern> creator = () => Nncase.PatternMatch.Utility.IsWildcard();
        Pattern = Nncase.PatternMatch.F.Tensors.IsConcat("concat", "concatCall", condition,
                Nncase.PatternMatch.Utility.IsTuple("tuple",
                    Nncase.PatternMatch.Utility.IsVArgsRepeat("tupleInputs", creator))) with
            {
                TypePattern = (TypePatternUtility.IsFloat() &
                               TypePatternUtility.HasRank((int r) => r < 4 || r == 5,
                                   "Only support rank < 4 or rank == 5") & TypePatternUtility.HasFixedShape())
            };
    }
}
