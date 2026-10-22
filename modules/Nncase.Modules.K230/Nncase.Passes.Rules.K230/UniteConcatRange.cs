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
public sealed class UniteConcatRange : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = Nncase.PatternMatch.Utility.IsRangeOfMarker(
        "marker",
        Nncase.PatternMatch.F.Tensors.IsConcat("concat", null, (Concat _) => true, Nncase.PatternMatch.Utility.IsTuple(
            Nncase.PatternMatch.Utility.IsVArgsRepeat("tupleInputs", delegate(ReadOnlySpan<Expr> exprs)
            {
                Pattern[] patterns = new Pattern[exprs.Length];
                for (int i = 0; i < exprs.Length; i++)
                {
                    patterns[i] = Nncase.PatternMatch.Utility.IsAlt(
                        Nncase.PatternMatch.Utility.IsRangeOfMarker(
                            Nncase.PatternMatch.Utility.IsWildcard($"input_{i}"),
                            Nncase.PatternMatch.Utility.IsTensorConst($"input_range_{i}")),
                        Nncase.PatternMatch.Utility.IsWildcard($"input_{i}"));
                }

                return patterns;
            }))), Nncase.PatternMatch.Utility.IsTensorConst("range"));

    private Expr? GetReplace(Marker marker, Tensor<float> range, IReadOnlyList<Expr> tupleInputs, Concat concat,
        IMatchResult result)
    {
        Tensor<float> tensor = range.Clone();
        for (int i = 0; i < tupleInputs.Count; i++)
        {
            if (tupleInputs[i] is Marker)
            {
                Tensor<float> tensor2 = ((TensorConst)result[$"input_range_{i}"]).Value.Cast<float>();
                tensor[new int[1]] = System.Math.Min(tensor[new int[1]], tensor2[new int[1]]);
                tensor[new int[1] { 1 }] = System.Math.Max(tensor[new int[1] { 1 }], tensor2[new int[1] { 1 }]);
            }
        }

        if (!tensor.Equals(range))
        {
            return Nncase.IR.F.Math
                .RangeOfMarker(
                    Nncase.IR.F.Tensors.Concat(new Nncase.IR.Tuple(tupleInputs.ToArray()), concat.Axis),
                    tensor).With(null, null, null, marker.MixQuantInfo, marker.AdaQuantInfo);
        }

        return null;
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Marker marker = (Marker)result["marker"];
        Tensor<float> range = ((TensorConst)result["range"]).Value.Cast<float>();
        IReadOnlyList<Expr> tupleInputs = (IReadOnlyList<Expr>)result["tupleInputs"];
        Concat concat = (Concat)result["concat"];
        return GetReplace(marker, range, tupleInputs, concat, result);
    }
}
