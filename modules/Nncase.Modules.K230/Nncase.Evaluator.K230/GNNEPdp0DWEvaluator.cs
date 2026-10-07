// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using OrtKISharp;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class GNNEPdp0DWEvaluator : IEvaluator<GNNEPdp0DW>, IEvaluator, ITypeInferencer<GNNEPdp0DW>, ITypeInferencer,
    ICostEvaluator<GNNEPdp0DW>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, GNNEPdp0DW target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    public IValue Visit(IEvaluateContext context, GNNEPdp0DW p)
    {
        Tensor input = context.GetArgumentValueAsTensor(p, GNNEPdp0DW.Input);
        Tensor weights = context.GetArgumentValueAsTensor(p, GNNEPdp0DW.Weights);

        // Per output channel weight zero points.
        byte[] weightsBias = context.GetArgumentValueAsArray<byte>(p, GNNEPdp0DW.WeightsBias);
        Half[] act = context.GetArgumentValueAsArray<Half>(p, GNNEPdp0DW.Act);

        // Input zero point.
        byte deqBias = context.GetArgumentValueAsScalar<byte>(p, GNNEPdp0DW.DeqBias);
        long shiftBits = context.GetArgumentValueAsScalar<long>(p, GNNEPdp0DW.ShiftBits);

        // [top, bottom, left, right].
        long[] padding = context.GetArgumentValueAsArray<long>(p, GNNEPdp0DW.Padding);
        long[] stride = context.GetArgumentValueAsArray<long>(p, GNNEPdp0DW.Stride);
        long[] dilation = context.GetArgumentValueAsArray<long>(p, GNNEPdp0DW.Dilation);
        long groups = context.GetArgumentValueAsScalar<long>(p, GNNEPdp0DW.Groups);

        // Remove the zero points (in place). NOTE: the side effect lives inside Select and relies on
        // ToArray() enumerating every element; kept as is so exceptions stay wrapped like before.
        float[] inputDeq = input.ToArray<float>();
        inputDeq.Select((float _, int i) => inputDeq[i] -= (int)deqBias).AsParallel().ToArray();
        float[] weightsDeq = weights.ToArray<float>();
        int weightsPerOutChannel = weights.Dimensions[1] * weights.Dimensions[2] * weights.Dimensions[3];
        weightsDeq.Select((float _, int i) => weightsDeq[i] -= (int)weightsBias[i / weightsPerOutChannel])
            .AsParallel().ToArray();

        OrtKISharp.Tensor ortInput = OrtKISharp.Tensor.MakeTensor(inputDeq, ToLongArray(input.Dimensions));
        OrtKISharp.Tensor ortWeights = OrtKISharp.Tensor.MakeTensor(weightsDeq, ToLongArray(weights.Dimensions));

        // Padding is [top, bottom, left, right]; ONNX wants [top, left, bottom, right].
        long[] onnxPads = new long[4] { padding[0], padding[2], padding[1], padding[3] };
        Tensor convOutput = OrtKI.Conv(
            ortInput,
            ortWeights,
            K230Kernels.ZeroBias(weights.Dimensions[0]),
            "NOTSET",
            dilation,
            groups,
            new long[2] { weights.Dimensions[2], weights.Dimensions[3] },
            onnxPads,
            stride).ToTensor();

        // Per-channel activation; the output is NCHW so the channel is the flat index / (H * W).
        float[] convValues = convOutput.ToArray<float>();
        float[] activated = new float[K230Kernels.ComputeSize(convOutput.Shape)];
        int channelStride = convOutput.Dimensions[2] * convOutput.Dimensions[3];
        for (int i = 0; i < convValues.Length; i++)
        {
            int channel = i / channelStride;
            activated[i] = K230Kernels.ApplyAct0(convValues[i], act.ToArray(), channel, (sbyte)shiftBits);
        }

        float[] rounded = activated.Select((float x) => (float)System.Math.Round(x)).ToArray();
        Half[] halves = activated.Select((float x) => (Half)x).ToArray();
        Tensor<float> roundedTensor = Tensor.From(rounded, convOutput.Shape);
        Tensor<Half> halfTensor = Tensor.From(halves, convOutput.Shape);
        if (p.DestType == DataTypes.UInt8)
        {
            return Value.FromTensor(roundedTensor.Cast<byte>(CastMode.KDefault));
        }

        if (p.DestType == DataTypes.Int8)
        {
            return Value.FromTensor(roundedTensor.Cast<sbyte>(CastMode.KDefault));
        }

        if (p.DestType == DataTypes.Int16)
        {
            return Value.FromTensor(roundedTensor.Cast<short>(CastMode.KDefault));
        }

        return Value.FromTensor(halfTensor.Cast<Half>(CastMode.KDefault));
    }

    public IRType Visit(ITypeInferenceContext context, GNNEPdp0DW target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, GNNEPdp0DW.Input);
        TensorType weights = context.CheckArgumentType<TensorType>(target, GNNEPdp0DW.Weights);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.Input);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.Weights);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.WeightsBias);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.WeightsBiasQint8);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.Act);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.ActQint8);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.DeqBias);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.ShiftBits);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.ShiftBitsQint8);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.Qint8Qp);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.Padding);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.Stride);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.Dilation);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.Groups);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.Is16Quant);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.PadValue);
        context.CheckArgumentType<IRType>(target, GNNEPdp0DW.WeightsQInt8);
        return Visit(context, target, input, weights);
    }

    private static long[] ToLongArray(ReadOnlySpan<int> values)
    {
        long[] result = new long[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = values[i];
        }

        return result;
    }

    private IRType Visit(ITypeInferenceContext context, GNNEPdp0DW target, TensorType input, TensorType weights)
    {
        if (input.DType != DataTypes.Int8 && input.DType != DataTypes.UInt8 && input.DType != DataTypes.Int16)
        {
            return new InvalidType("Unsupported input_type, should be one of [int8, uint8, int16]");
        }

        if (weights.DType != DataTypes.Int8 && weights.DType != DataTypes.UInt8 && weights.DType != DataTypes.Int16)
        {
            return new InvalidType("Unsupported w_type, should be one of [int8, uint8, int16]");
        }

        if (input.DType == DataTypes.Int16 && weights.DType == DataTypes.Int16)
        {
            return new InvalidType("int16 for both of input_type and w_type is not supported");
        }

        PrimType destType = target.DestType;
        Expr[] arguments = context.GetArguments(target, GNNEPdp0DW.Stride, GNNEPdp0DW.Padding, GNNEPdp0DW.Dilation,
            GNNEPdp0DW.Groups);
        if (input.Shape.IsUnranked)
        {
            return input with { Shape = Shape.Unknown(4) };
        }

        List<Dimension> outputShape = input.Shape.ToList();
        if (arguments[0] is TensorConst strideConst && arguments[1] is TensorConst paddingConst &&
            arguments[2] is TensorConst dilationConst && arguments[3] is TensorConst groupsConst &&
            input.Shape[2].IsFixed && input.Shape[3].IsFixed && weights.Shape[2].IsFixed &&
            weights.Shape[3].IsFixed)
        {
            // stride / dilation: [h, w]; padding: [[top, bottom], [left, right]].
            Tensor<int> strideValue = strideConst.Value.Cast<int>();
            Tensor<int> paddingValue = paddingConst.Value.Cast<int>();
            Tensor<int> dilationValue = dilationConst.Value.Cast<int>();
            int groupCount = groupsConst.Value.ToScalar<int>();
            if (input.Shape[1].FixedValue < groupCount || input.Shape[1].FixedValue % groupCount != 0)
            {
                return new InvalidType(
                    $"The Input Channel / Groups Error ({input.Shape[1].FixedValue}/{groupCount})");
            }

            outputShape[2] = TypePatternUtility.GetWindowedOutputSize(
                input.Shape[2].FixedValue + paddingValue[new int[2] { 0, 0 }] + paddingValue[new int[2] { 0, 1 }],
                weights.Shape[2].FixedValue, strideValue[new int[1] { 0 }], dilationValue[new int[1] { 0 }],
                same: false);
            outputShape[3] = TypePatternUtility.GetWindowedOutputSize(
                input.Shape[3].FixedValue + paddingValue[new int[2] { 1, 0 }] + paddingValue[new int[2] { 1, 1 }],
                weights.Shape[3].FixedValue, strideValue[new int[1] { 1 }], dilationValue[new int[1] { 1 }],
                same: false);
        }
        else
        {
            outputShape[3] = Dimension.Unknown;
            outputShape[2] = Dimension.Unknown;
        }

        // NOTE: the channel dimension (index 1) is copied from the input, not from the weights.
        return new TensorType(destType, new Shape(outputShape));
    }
}
