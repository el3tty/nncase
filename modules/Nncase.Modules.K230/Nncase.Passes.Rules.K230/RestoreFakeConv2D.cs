// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class RestoreFakeConv2D : RewriteRule<Pattern>
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

    public Expr? GetReplace(FakeConv2D fakeConv, Call call, Expr inputMarker, Expr input, Tensor<float> inputRange,
        Expr weightsMarker, Tensor weights, Tensor<float> weightsRange, TensorConst act, Expr padding, Expr stride,
        Expr dilation, int groups, Tensor<float> padValue)
    {
        int fixedValue = call.CheckedShape[1].FixedValue;
        ActParam2 ap = fakeConv.ActParam;
        if (ap.FusedClamp.All((ValueRange<float> x) => x.Max == ap.FusedClamp[0].Max && x.Min == ap.FusedClamp[0].Min))
        {
            float[] weightValues = weights.ToArray<float>();
            for (int index = 0; index < weightValues.Length; index++)
            {
                weightValues[index] *= ap.Ks[0, index / (weightValues.Length / fixedValue)];
            }

            float[] biasValues = new float[fixedValue];
            for (int channelIndex = 0; channelIndex < biasValues.Length; channelIndex++)
            {
                biasValues[channelIndex] = ap.Bs[0, channelIndex];
            }

            Tensor<float> tensor = Tensor.From(biasValues, new int[1] { fixedValue });
            Tensor<float> tensor2 = Tensor.From(weightValues, weights.Shape);
            return Nncase.IR.F.NN.Conv2D(input, tensor2, tensor, stride, padding, dilation, PadMode.Constant, groups,
                new float[2] { ap.FusedClamp[0].Min, ap.FusedClamp[0].Max });
        }

        return null;
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        FakeConv2D fakeConv = (FakeConv2D)result["fakeConv"];
        Call call = (Call)result["call"];
        Expr inputMarker = (Expr)result["inputMarker"];
        Expr input = (Expr)result["input"];
        Tensor<float> inputRange = ((TensorConst)result["inputRange"]).Value.Cast<float>();
        Expr weightsMarker = (Expr)result["weightsMarker"];
        Tensor value = ((TensorConst)result["weights"]).Value;
        Tensor<float> weightsRange = ((TensorConst)result["weightsRange"]).Value.Cast<float>();
        TensorConst act = (TensorConst)result["act"];
        Expr padding = (Expr)result["padding"];
        Expr stride = (Expr)result["stride"];
        Expr dilation = (Expr)result["dilation"];
        int groups = ((TensorConst)result["groups"]).Value.ToScalar<int>();
        Tensor<float> padValue = ((TensorConst)result["padValue"]).Value.Cast<float>();
        return GetReplace(fakeConv, call, inputMarker, input, inputRange, weightsMarker, value, weightsRange, act,
            padding, stride, dilation, groups, padValue);
    }
}
