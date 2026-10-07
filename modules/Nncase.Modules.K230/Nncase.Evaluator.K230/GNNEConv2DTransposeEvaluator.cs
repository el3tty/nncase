// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class GNNEConv2DTransposeEvaluator : IEvaluator<GNNEConv2DTranspose>, IEvaluator,
    ITypeInferencer<GNNEConv2DTranspose>, ITypeInferencer, ICostEvaluator<GNNEConv2DTranspose>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, GNNEConv2DTranspose target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    public IValue Visit(IEvaluateContext context, GNNEConv2DTranspose conv2DTranspose)
    {
        Tensor input = context.GetArgumentValueAsTensor(conv2DTranspose, GNNEConv2DTranspose.Input);
        Tensor weights = context.GetArgumentValueAsTensor(conv2DTranspose, GNNEConv2DTranspose.Weights);

        // Per output channel weight zero points.
        byte[] weightsBias = context.GetArgumentValueAsArray<byte>(conv2DTranspose, GNNEConv2DTranspose.WeightsBias);
        Half[] act = context.GetArgumentValueAsArray<Half>(conv2DTranspose, GNNEConv2DTranspose.Act);

        // Input zero point.
        byte deqBias = context.GetArgumentValueAsScalar<byte>(conv2DTranspose, GNNEConv2DTranspose.DeqBias);
        long shiftBits = context.GetArgumentValueAsScalar<long>(conv2DTranspose, GNNEConv2DTranspose.ShiftBits);

        // [top, bottom, left, right].
        long[] padding = context.GetArgumentValueAsArray<long>(conv2DTranspose, GNNEConv2DTranspose.Padding);
        long[] stride = context.GetArgumentValueAsArray<long>(conv2DTranspose, GNNEConv2DTranspose.Stride);
        long[] dilation = context.GetArgumentValueAsArray<long>(conv2DTranspose, GNNEConv2DTranspose.Dilation);
        long groups = context.GetArgumentValueAsScalar<long>(conv2DTranspose, GNNEConv2DTranspose.Groups);

        // [h, w].
        long[] outputPadding =
            context.GetArgumentValueAsArray<long>(conv2DTranspose, GNNEConv2DTranspose.OutputPadding);

        // [N, C, H, W] of the result.
        long[] outputShape = context.GetArgumentValueAsArray<long>(conv2DTranspose, GNNEConv2DTranspose.OutputShape);
        int[] inputShape = input.Shape.ToValueArray();
        int[] weightsShape = weights.Shape.ToValueArray();

        // The input must be what a regular conv over the (padded) output would produce.
        // NOTE: the exception type is odd (InvalidOleVariantTypeException) but kept as is.
        if (K230Kernels.GetWindowedOutputSize(
                (int)outputShape[2] + (int)padding[0] + (int)padding[1] - (int)outputPadding[0], weightsShape[2],
                (int)stride[0], (int)dilation[0], same: false) != inputShape[2] ||
            K230Kernels.GetWindowedOutputSize(
                (int)outputShape[3] + (int)padding[2] + (int)padding[3] - (int)outputPadding[1], weightsShape[3],
                (int)stride[1], (int)dilation[1], same: false) != inputShape[3])
        {
            throw new InvalidOleVariantTypeException("Invalid conv2d transpose shape");
        }

        // Remove the zero points (in place). NOTE: the side effect lives inside Select and relies on
        // ToArray() enumerating every element; kept as is so exceptions stay wrapped like before.
        float[] inputDeq = input.ToArray<float>();
        inputDeq.Select((float _, int i) => inputDeq[i] -= (int)deqBias).AsParallel().ToArray();
        float[] weightsDeq = weights.ToArray<float>();
        int weightsPerOutChannel = weights.Dimensions[1] * weights.Dimensions[2] * weights.Dimensions[3];
        weightsDeq.Select((float _, int i) => weightsDeq[i] -= (int)weightsBias[i / weightsPerOutChannel])
            .AsParallel().ToArray();

        long outputElementCount = outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3];
        float[] outputValues = new float[outputElementCount];
        Array.Clear(outputValues, 0, (int)outputElementCount);
        long inChannelsPerGroup = inputShape[1] / groups;
        long outChannelsPerGroup = outputShape[1] / groups;

        // Flat (n, c, y, x) read position in the input.
        int inputIndex = 0;

        // Scatter every input pixel, scaled by the kernel, into the output (the transpose of a convolution).
        // NOTE: the weights are indexed as [outChannel, inChannelInGroup, kh, kw] inside each group.
        for (int batch = 0; batch < inputShape[0]; batch++)
        {
            Span<float> outputSpan = outputValues.AsSpan();
            Span<float> batchOutput = outputSpan.Slice(batch * (int)outputShape[1] * (int)outputShape[2] *
                                                       (int)outputShape[3]);
            for (int groupIndex = 0; groupIndex < groups; groupIndex++)
            {
                Span<float> groupOutput =
                    batchOutput.Slice(groupIndex * (int)outChannelsPerGroup * (int)outputShape[2] *
                                      (int)outputShape[3]);
                Span<float> weightsSpan = weightsDeq.AsSpan();
                Span<float> groupWeights = weightsSpan.Slice(groupIndex * (int)outChannelsPerGroup *
                                                             (int)inChannelsPerGroup * weightsShape[2] *
                                                             weightsShape[3]);
                for (int inChannel = 0; inChannel < inChannelsPerGroup; inChannel++)
                {
                    for (int inY = 0; inY < inputShape[2]; inY++)
                    {
                        for (int inX = 0; inX < inputShape[3]; inX++)
                        {
                            // Output position of kernel tap (0, 0) and the range of taps that land inside
                            // the output.
                            int originY = (int)(inY * stride[0] - padding[0]);
                            int originX = (int)(inX * stride[1] - padding[2]);
                            int kernelYBegin = System.Math.Max(0, (int)((-originY + dilation[0] - 1) / dilation[0]));
                            int kernelYEnd = (int)System.Math.Min(weightsShape[2],
                                ((int)outputShape[2] - originY + dilation[0] - 1) / dilation[0]);
                            int kernelXBegin = (int)System.Math.Max(0L, (-originX + dilation[1] - 1) / dilation[1]);
                            int kernelXEnd = (int)System.Math.Min(weightsShape[3],
                                ((int)outputShape[3] - originX + dilation[1] - 1) / dilation[1]);

                            // NOTE: the bounds check is always true here.
                            float inputValue = (inX >= 0 && inX < inputShape[3] && inY >= 0 && inY < inputShape[2])
                                ? inputDeq[inputIndex]
                                : 0f;
                            inputIndex++;
                            for (int outChannel = 0; outChannel < outChannelsPerGroup; outChannel++)
                            {
                                Span<float> channelOutput =
                                    groupOutput.Slice((int)(outChannel * outputShape[2] * outputShape[3]));
                                Span<float> kernel = groupWeights
                                    .Slice((int)(outChannel * inChannelsPerGroup * weightsShape[2] *
                                                 weightsShape[3]))
                                    .Slice(inChannel * weightsShape[2] * weightsShape[3]);
                                for (int kernelY = kernelYBegin; kernelY < kernelYEnd; kernelY++)
                                {
                                    for (int kernelX = kernelXBegin; kernelX < kernelXEnd; kernelX++)
                                    {
                                        int outY = (int)(originY + dilation[0] * kernelY);
                                        int outX = (int)(originX + dilation[1] * kernelX);
                                        float weight = kernel[kernelY * weightsShape[3] + kernelX];
                                        channelOutput[(int)(outY * outputShape[3] + outX)] += inputValue * weight;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor<float> convOutput = Tensor.From(outputValues, ToIntArray(outputShape));

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
        if (conv2DTranspose.DestType == DataTypes.UInt8)
        {
            return Value.FromTensor(roundedTensor.Cast<byte>(CastMode.KDefault));
        }

        if (conv2DTranspose.DestType == DataTypes.Int8)
        {
            return Value.FromTensor(roundedTensor.Cast<sbyte>(CastMode.KDefault));
        }

        if (conv2DTranspose.DestType == DataTypes.Int16)
        {
            return Value.FromTensor(roundedTensor.Cast<short>(CastMode.KDefault));
        }

        return Value.FromTensor(halfTensor.Cast<Half>(CastMode.KDefault));
    }

    public IRType Visit(ITypeInferenceContext context, GNNEConv2DTranspose target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, GNNEConv2DTranspose.Input);
        TensorType weights = context.CheckArgumentType<TensorType>(target, GNNEConv2DTranspose.Weights);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Input);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Weights);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.WeightsBias);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.WeightsBiasQint8);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Act);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.ActQint8);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.DeqBias);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.ShiftBits);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.ShiftBitsQint8);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Qint8Qp);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Padding);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Stride);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Dilation);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Groups);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Is16Quant);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.PadValue);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.WeightsQInt8);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.OutputPadding);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.OutputShape);
        return Visit(context, target, input, weights);
    }

    private static int[] ToIntArray(long[] values)
    {
        return (from v in values
                select (int)v).ToArray();
    }

    private IRType Visit(ITypeInferenceContext context, GNNEConv2DTranspose target, TensorType input,
        TensorType weights)
    {
        // NOTE: the three InvalidType objects below are created and dropped (there is no `return`), so the
        // dtype checks have no effect; kept as is.
        if (input.DType != DataTypes.Int8 && input.DType != DataTypes.UInt8 && input.DType != DataTypes.Int16)
        {
            new InvalidType("Unsupported input_type, should be one of [int8, uint8, int16]");
        }

        if (weights.DType != DataTypes.Int8 && weights.DType != DataTypes.UInt8 && weights.DType != DataTypes.Int16)
        {
            new InvalidType("Unsupported w_type, should be one of [int8, uint8, int16]");
        }

        if (input.DType == DataTypes.Int16 && weights.DType == DataTypes.Int16)
        {
            new InvalidType("int16 for both of input_type and w_type is not supported");
        }

        if (context.GetArgument(target, GNNEConv2DTranspose.OutputShape) is Const outputShape)
        {
            return new TensorType(target.DestType, new Shape(Value.FromConst(outputShape).AsTensor().ToArray<int>()));
        }

        return new InvalidType("Conv2dTranspose can't infer shape with dynamic outputShape");
    }
}
