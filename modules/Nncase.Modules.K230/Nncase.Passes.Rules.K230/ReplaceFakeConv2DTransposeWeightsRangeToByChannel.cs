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
public sealed class ReplaceFakeConv2DTransposeWeightsRangeToByChannel : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = Nncase.PatternMatch.F.K230.IsFakeConv2DTranspose(
        "fakeConvTranspose",
        "call", (FakeConv2DTranspose _) => true,
        Nncase.PatternMatch.Utility.IsRangeOfMarker("inputMarker", Nncase.PatternMatch.Utility.IsWildcard("input"),
            Nncase.PatternMatch.Utility.IsConst("inputRange")),
        Nncase.PatternMatch.Utility.IsRangeOfMarker(
            "weightsMarker",
            Nncase.PatternMatch.Utility.IsTensorConst("weights"), Nncase.PatternMatch.Utility.IsConst("weightsRange")),
        Nncase.PatternMatch.Utility.IsTensorConst("act"), Nncase.PatternMatch.Utility.IsTensorConst("outputShape"),
        Nncase.PatternMatch.Utility.IsTensorConst("padding"),
        Nncase.PatternMatch.Utility.IsTensorConst("outputPadding"), Nncase.PatternMatch.Utility.IsTensorConst("stride"),
        Nncase.PatternMatch.Utility.IsTensorConst("dilation"), Nncase.PatternMatch.Utility.IsTensorConst("groups"),
        Nncase.PatternMatch.Utility.IsTensorConst("padValue"));

    private Expr? GetReplace(FakeConv2DTranspose fakeConvTranspose, Marker inputMarker, Marker weightsMarker,
        Expr input, Expr inputRange, Expr weights, Tensor<float> act, Expr outputShape, Expr padding,
        Expr outputPadding, Expr stride, Expr dilation, int groups, Expr padValue, RunPassContext options)
    {
        if (weightsMarker.MixQuantInfo == null)
        {
            weightsMarker.MixQuantInfo = new K230Target.K230MixQuantInfo();
        }

        float[] weightValues = ((TensorConst)weights).Value.ToArray<float>();
        int outChannels = weights.CheckedShape[0].FixedValue * groups;
        TensorConst range = new TensorConst(Tensor.From(
            QuantUtility.GetWeightsRangesByChannel(weightValues, outChannels).ToArray(),
            new int[2] { outChannels, 2 }));
        Marker marker = Nncase.IR.F.Math.RangeOfMarker(weights, range);
        if (marker.MixQuantInfo == null)
        {
            marker.MixQuantInfo = new K230Target.K230MixQuantInfo();
        }

        marker.MixQuantInfo = weightsMarker.MixQuantInfo;
        Call call = Nncase.IR.K230.F.Tensors.FakeConv2DTranspose(inputMarker, marker, act, outputShape, padding,
            outputPadding, stride, dilation, groups, padValue, fakeConvTranspose.ActParam);
        options.MatchOptions.SuppressPattern(call, Pattern);
        return call;
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        FakeConv2DTranspose fakeConvTranspose = (FakeConv2DTranspose)result["fakeConvTranspose"];
        Marker inputMarker = (Marker)result["inputMarker"];
        Marker weightsMarker = (Marker)result["weightsMarker"];
        Expr input = (Expr)result["input"];
        Expr inputRange = (Expr)result["inputRange"];
        Expr weights = (Expr)result["weights"];
        Tensor<float> act = ((TensorConst)result["act"]).Value.Cast<float>();
        Expr outputShape = (Expr)result["outputShape"];
        Expr padding = (Expr)result["padding"];
        Expr outputPadding = (Expr)result["outputPadding"];
        Expr stride = (Expr)result["stride"];
        Expr dilation = (Expr)result["dilation"];
        int groups = ((TensorConst)result["groups"]).Value.ToScalar<int>();
        Expr padValue = (Expr)result["padValue"];
        return GetReplace(fakeConvTranspose, inputMarker, weightsMarker, input, inputRange, weights, act, outputShape,
            padding, outputPadding, stride, dilation, groups, padValue, context);
    }
}
