// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class FoldReluConcatWithConv2d : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = Nncase.PatternMatch.Utility.IsRangeOfMarker(
        "reluMarker",
        Nncase.PatternMatch.F.NN.IsRelu("relu", "reluCall", (Relu _) => true,
            Nncase.PatternMatch.Utility.IsRangeOfMarker("concatMarker", Nncase.PatternMatch.F.Tensors.IsConcat(
                "concat",
                "catCall", (Concat _) => true, Nncase.PatternMatch.Utility.IsTuple(
                    Nncase.PatternMatch.Utility.IsVArgsRepeat("tupleInputs", delegate(ReadOnlySpan<Expr> exprs)
                    {
                        Pattern[] patterns = new Pattern[exprs.Length];
                        for (int i = 0; i < patterns.Length; i++)
                        {
                            patterns[i] = Nncase.PatternMatch.Utility.IsRangeOfMarker(
                                $"convMarker{i}",
                                Nncase.PatternMatch.Utility.IsWildcard($"conv2d{i}"),
                                Nncase.PatternMatch.Utility.IsTensorConst($"convRange{i}"));
                        }

                        return patterns;
                    }))), Nncase.PatternMatch.Utility.IsTensorConst("concatRange"))),
        Nncase.PatternMatch.Utility.IsTensorConst("reluRange"));

    public Expr? GetReplace(Call reluCall, Expr reluMarker, Expr concatMarker, IReadOnlyList<Expr> tupleInputs,
        Tensor<float> concatRange, IMatchResult matchResult, Concat concat, Tensor<float> reluRange)
    {
        bool isReluConcat = tupleInputs.All((Expr call) => ((Call)((Marker)call).Target).Target is Conv2D);
        Tensor<float>[] convRanges = (from i in Enumerable.Range(0, tupleInputs.Count)
            select ((TensorConst)matchResult[$"convRange{i}"]).Value.Cast<float>()).ToArray();
        Expr[] fields = (from i in Enumerable.Range(0, tupleInputs.Count)
            select ((Marker)matchResult[$"convMarker{i}"]).With(null, null,
                new float[2] { 0f, convRanges[i][new int[1] { 1 }] })).ToArray();
        Call target = Nncase.IR.F.Tensors.Concat(new Nncase.IR.Tuple(fields), concat.Axis);
        DataType markerQuantType = ((Marker)reluMarker).MixQuantInfo.MarkerQuantType;
        if (isReluConcat && reluCall.Target is Relu && markerQuantType != DataTypes.Int16)
        {
            return Nncase.IR.F.Math.RangeOfMarker(target, reluRange).With(null, null, null,
                adaQuantInfo: ((Marker)reluMarker).AdaQuantInfo, mixQuantInfo: ((Marker)reluMarker).MixQuantInfo);
        }

        return null;
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Call reluCall = (Call)result["reluCall"];
        Expr reluMarker = (Expr)result["reluMarker"];
        Expr concatMarker = (Expr)result["concatMarker"];
        IReadOnlyList<Expr> tupleInputs = (IReadOnlyList<Expr>)result["tupleInputs"];
        Tensor<float> concatRange = ((TensorConst)result["concatRange"]).Value.Cast<float>();
        Concat concat = (Concat)result["concat"];
        Tensor<float> reluRange = ((TensorConst)result["reluRange"]).Value.Cast<float>();
        return GetReplace(reluCall, reluMarker, concatMarker, tupleInputs, concatRange, result, concat, reluRange);
    }
}
