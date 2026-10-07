#define TRACE
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.PatternMatch;
using OrtKISharp;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class FakeConv2DTransposeEvaluator : IEvaluator<FakeConv2DTranspose>, IEvaluator,
    ITypeInferencer<FakeConv2DTranspose>, ITypeInferencer, ICostEvaluator<FakeConv2DTranspose>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, FakeConv2DTranspose target)
    {
        TensorType inputType = context.GetArgumentType<TensorType>(target, FakeConv2DTranspose.Input);
        TensorType weightsType = context.GetArgumentType<TensorType>(target, FakeConv2DTranspose.Weights);
        Shape weightsShape = weightsType.Shape;
        TensorType returnType = context.GetReturnType<TensorType>();
        Dimension macsPerOutput = weightsShape[1] * weightsShape[2] * weightsShape[3];
        return new Cost
        {
            [CostFactorNames.MemoryLoad] =
                CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightsType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.CPUCycles] =
                CostUtility.GetCPUCycles(returnType, (float)(macsPerOutput.FixedValue * 2) / 768f)
        };
    }

    public IValue Visit(IEvaluateContext context, FakeConv2DTranspose conv)
    {
        OrtKISharp.Tensor input = context.GetOrtArgumentValue(conv, FakeConv2DTranspose.Input);
        OrtKISharp.Tensor weights = context.GetOrtArgumentValue(conv, FakeConv2DTranspose.Weights);
        long[] stride = context.GetArgumentValueAsArray<long>(conv, FakeConv2DTranspose.Stride);

        // NOTE: the value is fetched and dropped (padding is read again as a plain array below); kept so
        // that any exception from this conversion is preserved.
        context.GetOrtArgumentValue(conv, FakeConv2DTranspose.Padding);
        long[] dilation = context.GetArgumentValueAsArray<long>(conv, FakeConv2DTranspose.Dilation);
        long groups = context.GetArgumentValueAsScalar<long>(conv, FakeConv2DTranspose.Groups);

        // [N, C, H, W] of the result.
        long[] outputShape = context.GetArgumentValueAsArray<long>(conv, FakeConv2DTranspose.OutputShape);
        long[] weightsShape = weights.Shape;

        // [top, bottom, left, right].
        long[] padding = context.GetArgumentValueAsArray<long>(conv, FakeConv2DTranspose.Padding);
        long[] inputShape = input.Shape;

        // [h, w].
        long[] outputPadding = context.GetArgumentValueAsArray<long>(conv, FakeConv2DTranspose.OutputPadding);

        // The input must be what a regular conv over the (padded) output would produce.
        // NOTE: the exception type is odd (InvalidOleVariantTypeException) but kept as is.
        if (K230Kernels.GetWindowedOutputSize(
                (int)outputShape[2] + (int)padding[0] + (int)padding[1] - (int)outputPadding[0],
                (int)weightsShape[2], (int)stride[0], (int)dilation[0], same: false) !=
            inputShape[2] || K230Kernels.GetWindowedOutputSize(
                (int)outputShape[3] + (int)padding[2] + (int)padding[3] - (int)outputPadding[1],
                (int)weightsShape[3], (int)stride[1], (int)dilation[1], same: false) != inputShape[3])
        {
            throw new InvalidOleVariantTypeException("Invalid conv2d transpose shape");
        }

        Tensor act = context.GetArgumentValueAsTensor(conv, FakeConv2DTranspose.Act);

        // Mixed-precision search: simulate the quantization of the (marker wrapped) input and weights.
        if (context.CurrentCall.EnodeBestQuantConfigWithCosine != null)
        {
            MarkerPattern markerPattern = Utility.IsRangeOfMarker(Utility.IsWildcard(), Utility.IsWildcard());
            if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[0]))
            {
                MixQuantInfo? inputMixQuantInfo = ((Marker)context.CurrentCall.Arguments[0]).MixQuantInfo;
                if (inputMixQuantInfo != null && inputMixQuantInfo.HasBindedMixQuantInfo)
                {
                    List<QuantParam> inputQuantParams = inputMixQuantInfo.QuantParameter;
                    Trace.Assert(inputQuantParams.Count == 1);
                    input = FakeQuantizePerTensor(input, inputQuantParams[0]);
                }
            }

            if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[1]))
            {
                MixQuantInfo? weightsMixQuantInfo = ((Marker)context.CurrentCall.Arguments[1]).MixQuantInfo;
                if (weightsMixQuantInfo != null && weightsMixQuantInfo.HasBindedMixQuantInfo)
                {
                    List<QuantParam> weightsQuantParams = weightsMixQuantInfo.QuantParameter;
                    weights = FakeQuantizePerChannel(weights, weightsQuantParams);
                }
            }
        }

        long outputElementCount = outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3];
        float[] outputValues = new float[outputElementCount];
        Array.Clear(outputValues, 0, (int)outputElementCount);
        long inChannelsPerGroup = inputShape[1] / groups;
        long outChannelsPerGroup = outputShape[1] / groups;
        float[] weightsValues = weights.ToArray<float>();
        float[] inputValues = input.ToArray<float>();

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
                    batchOutput.Slice(groupIndex * (int)outChannelsPerGroup * (int)outputShape[2] * (int)outputShape[3]);
                Span<float> weightsSpan = weightsValues.AsSpan();
                Span<float> groupWeights = weightsSpan.Slice(groupIndex * (int)outChannelsPerGroup *
                                                             (int)inChannelsPerGroup * (int)weightsShape[2] *
                                                             (int)weightsShape[3]);
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
                                ? inputValues[inputIndex]
                                : 0f;
                            inputIndex++;
                            for (int outChannel = 0; outChannel < outChannelsPerGroup; outChannel++)
                            {
                                Span<float> channelOutput =
                                    groupOutput.Slice((int)(outChannel * outputShape[2] * outputShape[3]));
                                Span<float> kernel = groupWeights
                                    .Slice((int)(outChannel * inChannelsPerGroup * weightsShape[2] * weightsShape[3]))
                                    .Slice((int)(inChannel * weightsShape[2] * weightsShape[3]));
                                for (int kernelY = kernelYBegin; kernelY < kernelYEnd; kernelY++)
                                {
                                    for (int kernelX = kernelXBegin; kernelX < kernelXEnd; kernelX++)
                                    {
                                        int outY = (int)(originY + dilation[0] * kernelY);
                                        int outX = (int)(originX + dilation[1] * kernelX);
                                        float weight = kernel[(int)(kernelY * weightsShape[3] + kernelX)];
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
            activated[i] = K230Kernels.FakeApplyAct0(convValues[i], act.ToArray<float>(), channel, 0);
        }

        // NOTE: the original rounded the activated values into a throw-away copy, so the result is NOT rounded.
        return Value.FromTensor(Tensor.From(activated, ToIntArray(outputShape)));
    }

    public IRType Visit(ITypeInferenceContext context, FakeConv2DTranspose target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, FakeConv2DTranspose.Input);
        context.CheckArgumentType<IRType>(target, FakeConv2DTranspose.Input);
        context.CheckArgumentType<IRType>(target, FakeConv2DTranspose.Weights);
        context.CheckArgumentType<IRType>(target, FakeConv2DTranspose.Act);
        context.CheckArgumentType<IRType>(target, FakeConv2DTranspose.OutputShape);
        context.CheckArgumentType<IRType>(target, FakeConv2DTranspose.Padding);
        context.CheckArgumentType<IRType>(target, FakeConv2DTranspose.OutputPadding);
        context.CheckArgumentType<IRType>(target, FakeConv2DTranspose.Stride);
        context.CheckArgumentType<IRType>(target, FakeConv2DTranspose.Dilation);
        context.CheckArgumentType<IRType>(target, FakeConv2DTranspose.Groups);
        context.CheckArgumentType<IRType>(target, FakeConv2DTranspose.Value);
        return Visit(context, target, input);
    }

    private static int[] ToIntArray(long[] values)
    {
        return (from v in values
                select (int)v).ToArray();
    }

    /// <summary>
    /// Fake-quantizes one value: quantize with <paramref name="param"/>, round (unless the parameter is the
    /// identity scale 1 / zero point 0) and de-quantize back to float.
    /// </summary>
    private static float FakeQuantize(float value, QuantParam param)
    {
        double quantized = (double)value / (double)param.Scale + (double)param.ZeroPoint;
        if (!param.Scale.Equals(1f) || param.ZeroPoint != 0)
        {
            quantized = System.Math.Round(quantized);
        }

        double dequantized = (quantized - (double)param.ZeroPoint) * (double)param.Scale;
        return (float)dequantized;
    }

    /// <summary>Fake-quantizes a whole tensor with a single (per-tensor) quant parameter.</summary>
    private static OrtKISharp.Tensor FakeQuantizePerTensor(OrtKISharp.Tensor tensor, QuantParam param)
    {
        float[] values = tensor.ToArray<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = FakeQuantize(values[i], param);
        }

        return OrtKISharp.Tensor.MakeTensor(values, tensor.Shape);
    }

    /// <summary>
    /// Fake-quantizes a tensor with one quant parameter per slice along the first axis
    /// (the flat buffer is split into equally sized chunks).
    /// </summary>
    private static OrtKISharp.Tensor FakeQuantizePerChannel(OrtKISharp.Tensor tensor, List<QuantParam> quantParams)
    {
        float[] values = tensor.ToArray<float>();
        int paramCount = quantParams.Count;
        int elementsPerParam = values.Length / paramCount;
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = FakeQuantize(values[i], quantParams[i / elementsPerParam]);
        }

        return OrtKISharp.Tensor.MakeTensor(values, tensor.Shape);
    }

    private IRType Visit(ITypeInferenceContext context, FakeConv2DTranspose target, TensorType input)
    {
        if (context.GetArgument(target, FakeConv2DTranspose.OutputShape) is TensorConst outputShape)
        {
            return new TensorType(input.DType, new Shape(outputShape.Value.ToArray<int>()));
        }

        return new InvalidType("Conv2dTranspose can't infer shape with dynamic outputShape");
    }
}
