// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using Nncase.Utilities;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class ToGNNEActivation : GNNEActLowerRule
{
    public override Pattern Pattern { get; } = Nncase.PatternMatch.Utility.IsRangeOfMarker("outputMarker",
        Nncase.PatternMatch.F.K230.IsFakeActivation("fakeActivation", "call", (FakeActivation _) => true,
            Nncase.PatternMatch.Utility.IsRangeOfMarker("inputAMarker",
                Nncase.PatternMatch.Utility.IsWildcard("inputA"),
                Nncase.PatternMatch.Utility.IsTensorConst("inputARange")),
            Nncase.PatternMatch.Utility.IsAlt(
                Nncase.PatternMatch.Utility.IsRangeOfMarker("inputBMarker",
                    Nncase.PatternMatch.Utility.IsWildcard("inputB"),
                    Nncase.PatternMatch.Utility.IsTensorConst("inputBRange")), Nncase.PatternMatch.Utility.IsNone()),
            Nncase.PatternMatch.Utility.IsTensorConst("act"), Nncase.PatternMatch.Utility.IsTensorConst("outChannels"),
            Nncase.PatternMatch.Utility.IsWildcard(), Nncase.PatternMatch.Utility.IsWildcard(),
            Nncase.PatternMatch.Utility.IsWildcard(), Nncase.PatternMatch.Utility.IsWildcard("is16Segment")),
        Nncase.PatternMatch.Utility.IsTensorConst("outputRange"));

    public Expr? GetReplace(FakeActivation fakeActivation, Call call, Expr inputA, Tensor<float> inputARange,
        Tensor<int> outChannels, Expr is16Segment, Expr inputAMarker, Expr outputMarker, IMatchResult result,
        Expr outputRange)
    {
        Call activationTable = Nncase.IR.K230.F.Tensors.GNNELoadW(DataTypes.Float16, fakeActivation.ActParam.ToAct1Data());

        // All shapes are right-aligned into rank 4: [1, 1, 1, 1] padded on the left.
        int[] inputAShape = PadShapeToRank4(inputA.CheckedShape);
        int[] outputShape = PadShapeToRank4(call.CheckedShape);
        bool needsOutputReshape = outputShape.Length != call.CheckedShape.Count;

        // The second operand is optional. When the pattern matched the "none" alternative
        // there is no "inputB" capture at all.
        if (result.GetValueOrDefault("inputB") is not Expr inputB)
        {
            return BuildUnaryActivation(
                fakeActivation, call, inputAMarker, inputARange, inputAShape, outputShape, needsOutputReshape,
                outChannels, is16Segment, activationTable, outputMarker, outputRange);
        }

        var inputBMarker = (Expr)result["inputBMarker"];
        Tensor inputBRange = ((TensorConst)result["inputBRange"]).Value;
        int[] inputBShape = PadShapeToRank4(inputB.CheckedShape);

        Call reshapedInputA = Nncase.IR.F.Tensors.Reshape(inputAMarker, inputAShape);
        Call reshapedInputB = Nncase.IR.F.Tensors.Reshape(inputBMarker, inputBShape);

        Call loadedInputA;
        Call loadedInputB;
        DeQuantizeParam inputADequantParam;
        DeQuantizeParam inputBDequantParam;

        if (inputB is TensorConst)
        {
            // Constant second operand: it is always loaded as float16 without quantization.
            (loadedInputA, inputADequantParam) = LoadQuantized(reshapedInputA, inputAMarker, inputARange);
            if (InputA is TensorConst constInputA)
            {
                loadedInputA = Nncase.IR.K230.F.Tensors.GNNELoad(
                    DataTypes.Float16,
                    Const.FromTensor(Tensor.From(constInputA.Value.ToArray<float>(), inputAShape)));
            }

            loadedInputB = Nncase.IR.K230.F.Tensors.GNNELoad(DataTypes.Float16, reshapedInputB);
            if (InputB is TensorConst constInputB)
            {
                loadedInputB = Nncase.IR.K230.F.Tensors.GNNELoad(
                    DataTypes.Float16,
                    Const.FromTensor(Tensor.From(constInputB.Value.ToArray<float>(), inputBShape)));
            }

            inputBDequantParam = new DeQuantizeParam(0, 1f);
        }
        else
        {
            // Both operands are dynamic: each one is quantized with its own marker type and range.
            (loadedInputA, inputADequantParam) = LoadQuantized(reshapedInputA, inputAMarker, inputARange);
            (loadedInputB, inputBDequantParam) = LoadQuantized(reshapedInputB, inputBMarker, inputBRange);
        }

        return BuildActivation(
            fakeActivation, call, loadedInputA, loadedInputB, inputADequantParam, inputBDequantParam, outputShape,
            needsOutputReshape, outChannels, is16Segment, activationTable, outputMarker, outputRange);
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        FakeActivation fakeActivation = (FakeActivation)__result["fakeActivation"];
        Call call = (Call)__result["call"];
        Expr inputA = (Expr)__result["inputA"];
        Tensor<float> inputARange = ((TensorConst)__result["inputARange"]).Value.Cast<float>();
        Tensor<int> outChannels = ((TensorConst)__result["outChannels"]).Value.Cast<int>();
        Expr is16Segment = (Expr)__result["is16Segment"];
        Expr inputAMarker = (Expr)__result["inputAMarker"];
        Expr outputMarker = (Expr)__result["outputMarker"];
        Expr outputRange = (Expr)__result["outputRange"];
        base.Option = __context;
        base.MatchResult = __result;
        Init();
        return GetReplace(fakeActivation, call, inputA, inputARange, outChannels, is16Segment, inputAMarker,
            outputMarker, __result, outputRange);
    }

    private static int[] PadShapeToRank4(Shape shape)
    {
        int[] padded = { 1, 1, 1, 1 };
        Array.Copy(shape.ToValueArray(), 0, padded, padded.Length - shape.Count, shape.Count);
        return padded;
    }

    /// <summary>
    /// Float16, float32 and int16 operands are not quantized by the activation kernel;
    /// they are loaded as float16.
    /// </summary>
    private static bool IsLoadedAsFloat16(DataType dataType)
    {
        return dataType == DataTypes.Float16 || dataType == DataTypes.Float32 || dataType == DataTypes.Int16;
    }

    private DataType GetMarkerQuantType(Expr marker)
    {
        MixQuantInfo? mixQuantInfo = ((Marker)marker).MixQuantInfo;
        return mixQuantInfo?.MarkerQuantType ?? QuantType;
    }

    /// <summary>
    /// Quantizes <paramref name="source"/> to the quant type chosen by <paramref name="typeMarker"/> and
    /// loads it into GLB, returning the load together with the matching dequantize parameters.
    /// </summary>
    private (Call Load, DeQuantizeParam DequantParam) LoadQuantized(Expr source, Expr typeMarker, Tensor range)
    {
        DataType dataType = GetMarkerQuantType(typeMarker);
        if (IsLoadedAsFloat16(dataType))
        {
            return (
                Nncase.IR.K230.F.Tensors.GNNELoad(DataTypes.Float16, source),
                new DeQuantizeParam(0, 1f));
        }

        QuantMode quantMode = dataType == DataTypes.UInt8 ? QuantMode.UnsignedMode : QuantMode.SignedSymmetricMode;
        float[] rangeValues = range.ToArray<float>();
        QuantParam quantParam = QuantUtility.GetQuantParam(new ValueRange<float>(rangeValues[0], rangeValues[1]), 8, quantMode);
        Call quantized = Nncase.IR.F.Math.Quantize(
            source, new QuantParam(quantParam.ZeroPoint, quantParam.Scale), dataType);
        return (
            Nncase.IR.K230.F.Tensors.GNNELoad((PrimType)dataType, quantized),
            new DeQuantizeParam(quantParam.ZeroPoint, quantParam.Scale));
    }

    /// <summary>
    /// Activation with a single operand (no inputB).
    /// </summary>
    private Expr BuildUnaryActivation(
        FakeActivation fakeActivation, Call call, Expr inputAMarker, Tensor<float> inputARange, int[] inputAShape,
        int[] outputShape, bool needsOutputReshape, Tensor<int> outChannels, Expr is16Segment, Call activationTable,
        Expr outputMarker, Expr outputRange)
    {
        // Unlike the binary case, the reshaped operand is re-marked with its own range.
        Marker reshapedInputA =
            Nncase.IR.F.Math.RangeOfMarker(Nncase.IR.F.Tensors.Reshape(inputAMarker, inputAShape), inputARange);
        (Call loadedInputA, DeQuantizeParam inputADequantParam) =
            LoadQuantized(reshapedInputA, inputAMarker, inputARange);

        return BuildActivation(
            fakeActivation, call, loadedInputA, None.Default, inputADequantParam, new DeQuantizeParam(0, 1f),
            outputShape, needsOutputReshape, outChannels, is16Segment, activationTable, outputMarker, outputRange);
    }

    private static Expr BuildActivation(
        FakeActivation fakeActivation, Call call, Expr loadedInputA, Expr loadedInputB,
        DeQuantizeParam inputADequantParam, DeQuantizeParam inputBDequantParam, int[] outputShape,
        bool needsOutputReshape, Tensor<int> outChannels, Expr is16Segment, Call activationTable,
        Expr outputMarker, Expr outputRange)
    {
        Call stored = Nncase.IR.K230.F.Tensors.GNNEStore(
            DataTypes.Float32,
            Nncase.IR.K230.F.Tensors.GNNEActivation(
                loadedInputA,
                loadedInputB,
                activationTable,
                0,
                0,
                0,
                Tensor.FromArray(new[] { inputADequantParam }),
                Tensor.FromArray(new[] { inputBDequantParam }),
                outChannels,
                fakeActivation.Type,
                is16Segment,
                DataTypes.Float16,
                fakeActivation.ActParam,
                outputShape));
        stored.CheckedType = new TensorType(DataTypes.Float32, outputShape);

        Call result = stored;
        if (needsOutputReshape)
        {
            result = Nncase.IR.F.Tensors.Reshape(stored, call.CheckedShape);
            result.CheckedType = new TensorType(DataTypes.Float32, call.CheckedShape);
        }

        var marker = (Marker)outputMarker;
        return Nncase.IR.F.Math.RangeOfMarker(result, outputRange).With(
            null, null, null, adaQuantInfo: marker.AdaQuantInfo, mixQuantInfo: marker.MixQuantInfo);
    }
}
