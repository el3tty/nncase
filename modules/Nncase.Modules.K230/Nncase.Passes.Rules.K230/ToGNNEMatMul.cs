// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using Nncase.Quantization.K230;
using Nncase.Utilities;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class ToGNNEMatMul : GNNEDIFQuantRule
{
    public override Pattern Pattern { get; } = Nncase.PatternMatch.Utility.IsRangeOfMarker(
        Nncase.PatternMatch.F.K230.IsFakeMatMul("fakeMatmul", "call", (FakeMatMul _) => true,
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "inputAMarker",
                Nncase.PatternMatch.Utility.IsWildcard("inputA"), Nncase.PatternMatch.Utility.IsConst("inputARange")),
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "inputBMarker",
                Nncase.PatternMatch.Utility.IsWildcard("inputB"), Nncase.PatternMatch.Utility.IsConst("inputBRange"))),
        Nncase.PatternMatch.Utility.IsTensorConst("outputRange"));

    private Expr? GetReplace(FakeMatMul fakeMatmul, Call call, Expr inputA, Tensor<float> inputARange,
        Expr inputAMarker, Expr inputB, Tensor<float> inputBRange, Expr inputBMarker, Tensor<float> outputRange)
    {
        ActParam2 actParam = new ActParam2(fakeMatmul.ActParam2);
        MixQuantInfo mixQuantInfo = ((Marker)inputAMarker).MixQuantInfo;
        MixQuantInfo mixQuantInfo2 = ((Marker)inputBMarker).MixQuantInfo;
        DataType dataType = (mixQuantInfo?.MarkerQuantType == null) ? QuantType : mixQuantInfo.MarkerQuantType;
        if (dataType == DataTypes.Int16)
        {
            dataType = DataTypes.UInt8;
        }

        DataType inputBDataType =
            (mixQuantInfo2?.MarkerQuantType == null) ? QuantType : mixQuantInfo2.MarkerQuantType;
        Call inputA2 = LoadQuantInA(inputARange, dataType, inputAMarker);
        Call inputB2 = (inputB is TensorConst)
            ? LoadInB((TensorConst)inputB, inputBRange.AsValueRange(), inputBDataType)
            : LoadQuantInB(inputB, inputBRange, inputBDataType, inputBMarker);
        Call inputABias = LoadInABias(inputA, inputARange.AsValueRange(), dataType);
        Call act = LoadAct(actParam, inputARange.AsValueRange(), inputBRange.AsValueRange(), dataType, inputBDataType);
        Call input = Nncase.IR.K230.F.Tensors.GNNEMatMul(actParam, DataTypes.Float16, inputA2, inputB2, act, inputABias,
            0, 0, 0,
            Enumerable.Repeat(
                GetInBDeqQuantParam(inputBDataType, inputBRange.AsValueRange()).ZeroPoint,
                call.CheckedShape[1].FixedValue).ToArray());
        Call call2 = Nncase.IR.K230.F.Tensors.GNNEStore(DataTypes.Float32, input);
        call2.CheckedType = new TensorType(DataTypes.Float32, call.CheckedShape);
        return Nncase.IR.F.Math.RangeOfMarker(call2, outputRange);
    }

    private Call LoadQuantInA(Tensor<float> inARange, DataType quanType, Expr prevInputMarker)
    {
        DeQuantizeParam inADeqQuantParam = GetInADeqQuantParam(quanType, inARange.AsValueRange());
        Call input = Nncase.IR.F.Math.Quantize(
            prevInputMarker,
            Tensor.FromScalar(new QuantParam(inADeqQuantParam.ZeroPoint, inADeqQuantParam.Scale)), quanType);
        return Nncase.IR.K230.F.Tensors.GNNELoad((PrimType)quanType, input);
    }

    private Call LoadQuantInB(Expr prevInput, Tensor<float> inBRange, DataType quanType, Expr prevInputMarker)
    {
        DeQuantizeParam inBDeqQuantParam = GetInBDeqQuantParam(quanType, inBRange.AsValueRange());
        int missingRank = 4 - prevInput.CheckedShape.Rank;
        if (missingRank > 0)
        {
            Nncase.IR.F.Tensors.Unsqueeze(prevInput, Enumerable.Range(0, missingRank).ToArray());
        }

        Call input = Nncase.IR.F.Math.Quantize(
            prevInputMarker,
            Tensor.FromScalar(new QuantParam(inBDeqQuantParam.ZeroPoint, inBDeqQuantParam.Scale)), quanType);
        return Nncase.IR.K230.F.Tensors.GNNELoad((PrimType)quanType, input);
    }

    private Call LoadInABias(Expr prevInput, ValueRange<float> inARange, DataType quanType)
    {
        DeQuantizeParam inADeqQuantParam = GetInADeqQuantParam(quanType, inARange);
        byte[] biasBytes = new byte[prevInput.CheckedShape[1].FixedValue * prevInput.CheckedShape[2].FixedValue];
        for (int i = 0; i < biasBytes.Length; i++)
        {
            biasBytes[i] = checked((byte)inADeqQuantParam.ZeroPoint);
        }

        return Nncase.IR.K230.F.Tensors.GNNELoadW(DataTypes.UInt8, biasBytes);
    }

    private Call LoadInB(TensorConst inputb, ValueRange<float> inBRange, DataType quanType)
    {
        if (quanType == DataTypes.Int8)
        {
            Tensor<sbyte> tensor = new Tensor<sbyte>(GetQuantInBI8(inputb, inBRange, quanType), inputb.CheckedShape);
            return Nncase.IR.K230.F.Tensors.GNNELoad((PrimType)quanType, tensor);
        }

        if (quanType == DataTypes.UInt8)
        {
            Tensor<byte> tensor2 = new Tensor<byte>(GetQuantInBu8(inputb, inBRange, quanType), inputb.CheckedShape);
            return Nncase.IR.K230.F.Tensors.GNNELoad((PrimType)quanType, tensor2);
        }

        if (quanType == DataTypes.Int16)
        {
            Tensor<short> tensor3 = new Tensor<short>(GetQuantInBi16(inputb, inBRange, quanType), inputb.CheckedShape);
            return Nncase.IR.K230.F.Tensors.GNNELoad((PrimType)quanType, tensor3);
        }

        throw new NotSupportedException("Incalid in_b_quant_type");
    }

    private byte[] GetQuantInBu8(TensorConst inputB, ValueRange<float> inBRange, DataType quanType)
    {
        Shape checkedShape = inputB.CheckedShape;
        float[] weights = inputB.Value.ToArray<float>();
        byte[] quantWeights = new byte[weights.Length];
        QuantizeParam inBQuantParam = GetInBQuantParam(quanType, inBRange);
        for (int i = 0; i < checkedShape[1].FixedValue; i++)
        {
            for (int j = 0; j < checkedShape[2].FixedValue; j++)
            {
                for (int k = 0; k < checkedShape[3].FixedValue; k++)
                {
                    int flatIndex = i * checkedShape[2].FixedValue * checkedShape[3].FixedValue +
                              j * checkedShape[3].FixedValue + k;
                    quantWeights[flatIndex] = Quantize<byte>(weights[flatIndex], inBQuantParam);
                }
            }
        }

        return quantWeights;
    }

    private sbyte[] GetQuantInBI8(TensorConst inputB, ValueRange<float> inBRange, DataType quanType)
    {
        Shape checkedShape = inputB.CheckedShape;
        float[] weights = inputB.Value.ToArray<float>();
        sbyte[] quantWeights = new sbyte[weights.Length];
        QuantizeParam inBQuantParam = GetInBQuantParam(quanType, inBRange);
        for (int i = 0; i < checkedShape[1].FixedValue; i++)
        {
            for (int j = 0; j < checkedShape[2].FixedValue; j++)
            {
                for (int k = 0; k < checkedShape[3].FixedValue; k++)
                {
                    int flatIndex = i * checkedShape[2].FixedValue * checkedShape[3].FixedValue +
                              j * checkedShape[3].FixedValue + k;
                    quantWeights[flatIndex] = (sbyte)System.Math.Clamp(
                        Convert.ToInt32(Quantize<sbyte>(weights[flatIndex], inBQuantParam)),
                        -127, 127);
                }
            }
        }

        return quantWeights;
    }

    private short[] GetQuantInBi16(TensorConst inputB, ValueRange<float> inBRange, DataType quanType)
    {
        Shape checkedShape = inputB.CheckedShape;
        float[] weights = inputB.Value.ToArray<float>();
        short[] quantWeights = new short[weights.Length];
        QuantizeParam inBQuantParam = GetInBQuantParam(quanType, inBRange);
        for (int i = 0; i < checkedShape[1].FixedValue; i++)
        {
            for (int j = 0; j < checkedShape[2].FixedValue; j++)
            {
                for (int k = 0; k < checkedShape[3].FixedValue; k++)
                {
                    int flatIndex = i * checkedShape[2].FixedValue * checkedShape[3].FixedValue +
                              j * checkedShape[3].FixedValue + k;
                    quantWeights[flatIndex] = Quantize<short>(weights[flatIndex], inBQuantParam);
                }
            }
        }

        return quantWeights;
    }

    private T Quantize<T>(float data, QuantizeParam qp)
    {
        return (T)Convert.ChangeType(
            System.Math.Clamp(
                (int)System.Math.Round(data * qp.Scale + (float)qp.ZeroPoint),
                Convert.ToInt32(typeof(T).GetField("MinValue").GetValue(null)),
                Convert.ToInt32(typeof(T).GetField("MaxValue").GetValue(null))), typeof(T));
    }

    private Call LoadAct(ActParam2 actParam, ValueRange<float> inARange, ValueRange<float> inBRange, DataType quanTypeA,
        DataType quanTypeB)
    {
        DeQuantizeParam inADeqQuantParam = GetInADeqQuantParam(quanTypeA, inARange);
        DeQuantizeParam inBDeqQuantParam = GetInBDeqQuantParam(quanTypeB, inBRange);
        actParam.FusedChannelScale(Enumerable.Repeat(inADeqQuantParam.Scale, actParam.Channels).ToArray());
        actParam.FusedChannelScale(Enumerable.Repeat(inBDeqQuantParam.Scale, actParam.Channels).ToArray());
        return Nncase.IR.K230.F.Tensors.GNNELoadW(DataTypes.Float16, actParam.ToAct0Data());
    }

    private DeQuantizeParam GetInADeqQuantParam(DataType quantType, ValueRange<float> inARange)
    {
        QuantMode quantMode =
            (!(quantType == DataTypes.UInt8)) ? QuantMode.SignedSymmetricMode : QuantMode.UnsignedMode;
        int bits = (quantType == DataTypes.Int16) ? 12 : 8;
        return Nncase.Quantization.K230.Utility.GetDeqParam(QuantUtility.GetQuantParam(inARange, bits, quantMode));
    }

    private QuantizeParam GetInBQuantParam(DataType quantType, ValueRange<float> inBRange)
    {
        return Nncase.Quantization.K230.Utility.GetQuantParamFromDeqParam(GetInBDeqQuantParam(quantType, inBRange));
    }

    private DeQuantizeParam GetInBDeqQuantParam(DataType quantType, ValueRange<float> inBRange)
    {
        QuantMode quantMode =
            (!(quantType == DataTypes.UInt8)) ? QuantMode.SignedSymmetricMode : QuantMode.UnsignedMode;
        int bits = (quantType == DataTypes.Int16) ? 12 : 8;
        return Nncase.Quantization.K230.Utility.GetDeqParam(QuantUtility.GetQuantParam(inBRange, bits, quantMode));
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        FakeMatMul fakeMatmul = (FakeMatMul)result["fakeMatmul"];
        Call call = (Call)result["call"];
        Expr inputA = (Expr)result["inputA"];
        Tensor<float> inputARange = ((TensorConst)result["inputARange"]).Value.Cast<float>();
        Expr inputAMarker = (Expr)result["inputAMarker"];
        Expr inputB = (Expr)result["inputB"];
        Tensor<float> inputBRange = ((TensorConst)result["inputBRange"]).Value.Cast<float>();
        Expr inputBMarker = (Expr)result["inputBMarker"];
        Tensor<float> outputRange = ((TensorConst)result["outputRange"]).Value.Cast<float>();
        Option = context;
        MatchResult = result;
        Init();
        return GetReplace(fakeMatmul, call, inputA, inputARange, inputAMarker, inputB, inputBRange, inputBMarker,
            outputRange);
    }
}
