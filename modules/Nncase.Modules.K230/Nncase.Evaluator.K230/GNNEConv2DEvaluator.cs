using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using OrtKISharp;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class GNNEConv2DEvaluator : IEvaluator<GNNEConv2D>, IEvaluator, ITypeInferencer<GNNEConv2D>, ITypeInferencer,
    ICostEvaluator<GNNEConv2D>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, GNNEConv2D target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    public IValue Visit(IEvaluateContext context, GNNEConv2D conv)
    {
        Tensor input = context.GetArgumentValueAsTensor(conv, GNNEConv2D.Input);
        Tensor weights = context.GetArgumentValueAsTensor(conv, GNNEConv2D.Weights);

        // Per output channel weight zero points.
        byte[] weightsBias = context.GetArgumentValueAsArray<byte>(conv, GNNEConv2D.WeightsBias);
        Half[] act = context.GetArgumentValueAsArray<Half>(conv, GNNEConv2D.Act);

        // Input zero point.
        byte deqBias = context.GetArgumentValueAsScalar<byte>(conv, GNNEConv2D.DeqBias);
        long shiftBits = context.GetArgumentValueAsScalar<long>(conv, GNNEConv2D.ShiftBits);
        long[] padding = context.GetArgumentValueAsArray<long>(conv, GNNEConv2D.Padding);
        long[] stride = context.GetArgumentValueAsArray<long>(conv, GNNEConv2D.Stride);
        long[] dilation = context.GetArgumentValueAsArray<long>(conv, GNNEConv2D.Dilation);
        long groups = context.GetArgumentValueAsScalar<long>(conv, GNNEConv2D.Groups);

        // Re-pack the weights so that the input channel dimension is C / groups: every output channel keeps
        // only its first (C / groups * kh * kw) elements.
        long bytesPerOutputChannel = input.Shape[1].FixedValue / groups * weights.Shape[2].FixedValue *
                                     weights.Shape[3].FixedValue * weights.ElementType.SizeInBytes;
        byte[] packedWeightsBytes = new byte[weights.Shape[0].FixedValue * bytesPerOutputChannel];
        for (int outChannel = 0; outChannel < weights.Shape[0].FixedValue; outChannel++)
        {
            Array.Copy(weights.BytesBuffer.ToArray(),
                (long)outChannel * (long)weights.BytesBuffer.Length / weights.Shape[0].FixedValue,
                packedWeightsBytes, outChannel * bytesPerOutputChannel, bytesPerOutputChannel);
        }

        Dimension[] packedWeightsDims = weights.Shape.ToArray();
        packedWeightsDims[1] = (int)(input.Shape[1].FixedValue / groups);
        Tensor packedWeights =
            Tensor.FromBytes(new TensorType(weights.ElementType, packedWeightsDims), packedWeightsBytes);

        // Remove the zero points (in place). NOTE: the side effect lives inside Select and relies on
        // ToArray() enumerating every element; kept as is so exceptions stay wrapped like before.
        float[] inputDeq = input.ToArray<float>();
        inputDeq.Select((float _, int i) => inputDeq[i] -= (int)deqBias).AsParallel().ToArray();
        float[] weightsDeq = packedWeights.ToArray<float>();
        int weightsPerOutChannel = packedWeights.Dimensions[1] * packedWeights.Dimensions[2] *
                                   packedWeights.Dimensions[3];
        weightsDeq.Select((float _, int i) => weightsDeq[i] -= (int)weightsBias[i / weightsPerOutChannel])
            .AsParallel().ToArray();

        OrtKISharp.Tensor ortInput = OrtKISharp.Tensor.MakeTensor(inputDeq, ToLongArray(input.Dimensions));
        OrtKISharp.Tensor ortWeights =
            OrtKISharp.Tensor.MakeTensor(weightsDeq, ToLongArray(packedWeights.Dimensions));

        // Padding is [top, bottom, left, right]; ONNX wants [top, left, bottom, right].
        long[] onnxPads = new long[4] { padding[0], padding[2], padding[1], padding[3] };
        Tensor convOutput = OrtKI.Conv(
            ortInput,
            ortWeights,
            K230Kernels.ZeroBias(packedWeights.Dimensions[0]),
            "NOTSET",
            dilation,
            groups,
            new long[2] { packedWeights.Dimensions[2], packedWeights.Dimensions[3] },
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

        return ToDestTypeValue(conv.DestType, activated, convOutput.Shape);
    }

    public IRType Visit(ITypeInferenceContext context, GNNEConv2D target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, GNNEConv2D.Input);
        TensorType weights = context.CheckArgumentType<TensorType>(target, GNNEConv2D.Weights);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.Input);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.Weights);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.WeightsBias);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.WeightsBiasQint8);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.Act);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.ActQint8);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.DeqBias);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.ShiftBits);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.ShiftBitsQint8);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.Qint8Qp);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.Padding);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.Stride);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.Dilation);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.Groups);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.Is16Quant);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.PadValue);
        context.CheckArgumentType<IRType>(target, GNNEConv2D.WeightsQInt8);
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

    /// <summary>
    /// Converts the activated float values to the destination type: integer types are rounded first,
    /// every other destination type (float16) is a plain float to Half conversion.
    /// </summary>
    private static IValue ToDestTypeValue(PrimType destType, float[] activated, Shape shape)
    {
        float[] rounded = activated.Select((float x) => (float)System.Math.Round(x)).ToArray();
        Half[] halves = activated.Select((float x) => (Half)x).ToArray();
        Tensor<float> roundedTensor = Tensor.From(rounded, shape);
        Tensor<Half> halfTensor = Tensor.From(halves, shape);
        if (destType == DataTypes.UInt8)
        {
            return Value.FromTensor(roundedTensor.Cast<byte>(CastMode.KDefault));
        }

        if (destType == DataTypes.Int8)
        {
            return Value.FromTensor(roundedTensor.Cast<sbyte>(CastMode.KDefault));
        }

        if (destType == DataTypes.Int16)
        {
            return Value.FromTensor(roundedTensor.Cast<short>(CastMode.KDefault));
        }

        return Value.FromTensor(halfTensor.Cast<Half>(CastMode.KDefault));
    }

    private static IRType Conv2DTypeFp16(TensorType input, TensorType weights, Expr stride, Expr padding,
        Expr dilation, Expr groups, DataType target)
    {
        List<Dimension> outputShape = input.Shape.ToList();
        outputShape[1] = weights.Shape[0];
        if (stride is TensorConst strideConst && padding is TensorConst paddingConst &&
            dilation is TensorConst dilationConst && groups is TensorConst groupsConst && input.Shape[2].IsFixed &&
            input.Shape[3].IsFixed && weights.Shape[2].IsFixed && weights.Shape[3].IsFixed)
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

        return new TensorType(target, new Shape(outputShape));
    }

    private IRType Visit(ITypeInferenceContext context, GNNEConv2D target, TensorType input, TensorType weights)
    {
        int groups = ((TensorConst)context.GetArgument(target, GNNEConv2D.Groups)).Value.ToScalar<int>();

        // The weights hold C / groups input channels; fix the shape up if it does not.
        TensorType effectiveWeights = weights;
        if (input.Shape[1] / groups != weights.Shape[1])
        {
            effectiveWeights = weights with
            {
                Shape = new Shape(weights.Shape[0].FixedValue, input.Shape[1].FixedValue / groups,
                    weights.Shape[2].FixedValue, weights.Shape[3].FixedValue)
            };
        }

        if (input.DType != DataTypes.Int8 && input.DType != DataTypes.UInt8 && input.DType != DataTypes.Int16)
        {
            return new InvalidType("Unsupported input_type, should be one of [int8, uint8, int16]");
        }

        if (effectiveWeights.DType != DataTypes.Int8 && effectiveWeights.DType != DataTypes.UInt8 &&
            effectiveWeights.DType != DataTypes.Int16)
        {
            return new InvalidType("Unsupported w_type, should be one of [int8, uint8, int16]");
        }

        if (input.DType == DataTypes.Int16 && effectiveWeights.DType == DataTypes.Int16)
        {
            return new InvalidType("int16 for both of input_type and w_type is not supported");
        }

        PrimType destType = target.DestType;
        Expr[] arguments = context.GetArguments(target, GNNEConv2D.Stride, GNNEConv2D.Padding, GNNEConv2D.Dilation,
            GNNEConv2D.Groups);
        if (destType == DataTypes.Int8 || destType == DataTypes.UInt8 || destType == DataTypes.Int16)
        {
            IRType convType = TypeInference.Conv2DType(input, effectiveWeights, arguments[0], arguments[1],
                arguments[2], arguments[3]);
            if (convType is TensorType convTensorType)
            {
                return convTensorType with { DType = destType };
            }

            return convType;
        }

        if (destType == DataTypes.Float16)
        {
            return Conv2DTypeFp16(input, effectiveWeights, arguments[0], arguments[1], arguments[2], arguments[3],
                destType);
        }

        return new InvalidType("Conv2d output type should be one of [int8, int16, float16, uint8]");
    }
}
