// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using Nncase.Targets;
using Nncase.Utilities;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class ReplaceFakeConv2DWeightsRangeToByChannel : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = Nncase.PatternMatch.F.K230.IsFakeConv2D("fakeConv", "call",
        (FakeConv2D _) => true, Nncase.PatternMatch.Utility.IsRangeOfMarker(
            "inputMarker",
            Nncase.PatternMatch.Utility.IsWildcard("input"),
            Nncase.PatternMatch.Utility.IsConst("inputRange") with { TypePattern = TypePatternUtility.HasFixedShape() }),
        Nncase.PatternMatch.Utility.IsRangeOfMarker(
            "weightsMarker",
            Nncase.PatternMatch.Utility.IsTensorConst("weights"),
            Nncase.PatternMatch.Utility.IsConst("weightsRange") with
            {
                TypePattern = TypePatternUtility.HasFixedShape(),
            }), Nncase.PatternMatch.Utility.IsTensorConst("act"),
        Nncase.PatternMatch.Utility.IsTensorConst("padding"),
        Nncase.PatternMatch.Utility.IsTensorConst("stride"), Nncase.PatternMatch.Utility.IsTensorConst("dilation"),
        Nncase.PatternMatch.Utility.IsTensorConst("groups"), Nncase.PatternMatch.Utility.IsTensorConst("padValue"));

    private Expr? GetReplace(FakeConv2D fakeConv, Marker inputMarker, Marker weightsMarker, Expr input, Expr inputRange,
        Expr weights, Tensor<float> act, Expr padding, Expr stride, Expr dilation, int groups, Expr padValue,
        RunPassContext options)
    {
        if (weightsMarker.MixQuantInfo == null)
        {
            weightsMarker.MixQuantInfo = new K230Target.K230MixQuantInfo();
        }

        float[] weightValues = ((TensorConst)weights).Value.ToArray<float>();
        int fixedValue = weights.CheckedShape[0].FixedValue;
        TensorConst range = new TensorConst(Tensor.From(
            QuantUtility.GetWeightsRangesByChannel(weightValues, fixedValue).ToArray(), new int[2] { fixedValue, 2 }));
        Marker marker = Nncase.IR.F.Math.RangeOfMarker(weights, range);
        if (marker.MixQuantInfo == null)
        {
            marker.MixQuantInfo = new K230Target.K230MixQuantInfo();
        }

        marker.MixQuantInfo = weightsMarker.MixQuantInfo;
        Call call = Nncase.IR.K230.F.Tensors.FakeConv2D(inputMarker, marker, act, padding, stride, dilation, groups,
            padValue, fakeConv.ActParam);
        options.MatchOptions.SuppressPattern(call, Pattern);
        return call;
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        FakeConv2D fakeConv = (FakeConv2D)result["fakeConv"];
        Marker inputMarker = (Marker)result["inputMarker"];
        Marker weightsMarker = (Marker)result["weightsMarker"];
        Expr input = (Expr)result["input"];
        Expr inputRange = (Expr)result["inputRange"];
        Expr weights = (Expr)result["weights"];
        Tensor<float> act = ((TensorConst)result["act"]).Value.Cast<float>();
        Expr padding = (Expr)result["padding"];
        Expr stride = (Expr)result["stride"];
        Expr dilation = (Expr)result["dilation"];
        int groups = ((TensorConst)result["groups"]).Value.ToScalar<int>();
        Expr padValue = (Expr)result["padValue"];
        return GetReplace(fakeConv, inputMarker, weightsMarker, input, inputRange, weights, act, padding, stride,
            dilation, groups, padValue, context);
    }
}
