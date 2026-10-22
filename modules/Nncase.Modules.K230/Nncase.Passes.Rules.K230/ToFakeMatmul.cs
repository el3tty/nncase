// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class ToFakeMatmul : IRewriteRule
{
    public IPattern Pattern { get; } = Nncase.PatternMatch.Utility.IsRangeOfMarker(
        "outputMarker",
        Nncase.PatternMatch.F.Math.IsMatMul("matmul", "call", (MatMul _) => true,
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "inputAMarker",
                Nncase.PatternMatch.Utility.IsWildcard("inputA") with
                {
                    TypePattern = TypePatternUtility.HasFixedShape(),
                }, Nncase.PatternMatch.Utility.IsTensorConst("inputARange")),
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "inputBMarker",
                Nncase.PatternMatch.Utility.IsWildcard("inputB") with
                {
                    TypePattern = TypePatternUtility.HasFixedShape(),
                }, Nncase.PatternMatch.Utility.IsTensorConst("inputBRange"))),
        Nncase.PatternMatch.Utility.IsTensorConst("outputRange"));

    private Expr? GetReplace(Marker inputAMarker, Expr inputA, Marker inputBMarker, Expr inputB, Marker outputMarker,
        Call call, Tensor<float> inputARange, Tensor<float> inputBRange, Tensor<float> outputRange)
    {
        if (inputAMarker.MixQuantInfo != null && (inputAMarker.MixQuantInfo.MarkerQuantType == DataTypes.Float16 ||
                                                  inputAMarker.MixQuantInfo.MarkerQuantType == DataTypes.Float32))
        {
            return null;
        }

        if (inputBMarker.MixQuantInfo != null && (inputBMarker.MixQuantInfo.MarkerQuantType == DataTypes.Float16 ||
                                                  inputBMarker.MixQuantInfo.MarkerQuantType == DataTypes.Float32))
        {
            return null;
        }

        int[] inputAShape = new int[4] { 1, 1, 1, 1 };
        Array.Copy(inputAMarker.CheckedShape.ToValueArray(), 0, inputAShape, inputAShape.Length - inputAMarker.CheckedShape.Count,
            inputAMarker.CheckedShape.Count);
        int[] inputBShape = new int[4] { 1, 1, 1, 1 };
        Array.Copy(inputBMarker.CheckedShape.ToValueArray(), 0, inputBShape, inputBShape.Length - inputBMarker.CheckedShape.Count,
            inputBMarker.CheckedShape.Count);
        int[] outputShape = new int[4] { 1, 1, 1, 1 };
        Array.Copy(call.CheckedShape.ToValueArray(), 0, outputShape, outputShape.Length - call.CheckedShape.Count,
            call.CheckedShape.Count);
        ActParam2 actParam = new ActParam2(outputShape[1] * outputShape[2], new QuantParam(0, 1f));
        Marker inputA2 = Nncase.IR.F.Math.RangeOfMarker(Nncase.IR.F.Tensors.Reshape(inputAMarker, inputAShape), inputARange)
            .With(null, null, null, adaQuantInfo: inputAMarker.AdaQuantInfo, mixQuantInfo: inputAMarker.MixQuantInfo);
        Marker inputB2 = Nncase.IR.F.Math.RangeOfMarker(Nncase.IR.F.Tensors.Reshape(inputBMarker, inputBShape), inputBRange)
            .With(null, null, null, adaQuantInfo: inputBMarker.AdaQuantInfo, mixQuantInfo: inputBMarker.MixQuantInfo);
        return Nncase.IR.F.Math
            .RangeOfMarker(
                Nncase.IR.F.Tensors.Reshape(
                    Nncase.IR.F.Math
                        .RangeOfMarker(
                            Nncase.IR.K230.F.Tensors.FakeMatMul(inputA2, inputB2, actParam.ToFakeActData(), actParam),
                            outputRange)
                        .With(null, null, null, adaQuantInfo: outputMarker.AdaQuantInfo,
                            mixQuantInfo: outputMarker.MixQuantInfo), call.CheckedShape), outputRange).With(null, null,
                null, adaQuantInfo: outputMarker.AdaQuantInfo, mixQuantInfo: outputMarker.MixQuantInfo);
    }

    public Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Marker inputAMarker = (Marker)result["inputAMarker"];
        Expr inputA = (Expr)result["inputA"];
        Marker inputBMarker = (Marker)result["inputBMarker"];
        Expr inputB = (Expr)result["inputB"];
        Marker outputMarker = (Marker)result["outputMarker"];
        Call call = (Call)result["call"];
        Tensor<float> inputARange = ((TensorConst)result["inputARange"]).Value.Cast<float>();
        Tensor<float> inputBRange = ((TensorConst)result["inputBRange"]).Value.Cast<float>();
        Tensor<float> outputRange = ((TensorConst)result["outputRange"]).Value.Cast<float>();
        return GetReplace(inputAMarker, inputA, inputBMarker, inputB, outputMarker, call, inputARange, inputBRange,
            outputRange);
    }
}
