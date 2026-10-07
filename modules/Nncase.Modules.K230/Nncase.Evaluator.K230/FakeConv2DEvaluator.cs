#define TRACE
using System;
using System.Collections.Generic;
using System.Diagnostics;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.PatternMatch;
using OrtKISharp;

namespace Nncase.Evaluator.K230;

[EvaluatorGenerator]
[TypeInferGenerator]
public class FakeConv2DEvaluator : IEvaluator<FakeConv2D>, IEvaluator, ITypeInferencer<FakeConv2D>, ITypeInferencer,
    ICostEvaluator<FakeConv2D>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, FakeConv2D target)
    {
        TensorType inputType = context.GetArgumentType<TensorType>(target, FakeConv2D.Input);
        TensorType weightsType = context.GetArgumentType<TensorType>(target, FakeConv2D.Weights);
        TensorType returnType = context.GetReturnType<TensorType>();
        Shape weightsShape = weightsType.Shape;

        // Multiply-accumulate count per output element: 2 * (C/g * kh * kw) - 1.
        Dimension opsPerOutput = 2 * weightsShape[1] * weightsShape[2] * weightsShape[3] - 1;
        return new Cost
        {
            [CostFactorNames.MemoryLoad] =
                CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightsType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(returnType, (float)opsPerOutput.FixedValue / 768f)
        };
    }

    public IValue Visit(IEvaluateContext context, FakeConv2D target)
    {
        OrtKISharp.Tensor input = context.GetOrtArgumentValue(target, FakeConv2D.Input);
        OrtKISharp.Tensor weights = context.GetOrtArgumentValue(target, FakeConv2D.Weights);
        Tensor act = context.GetArgumentValueAsTensor(target, FakeConv2D.Act);
        Tensor<long> stride = context.GetArgumentValueAsTensor<long>(target, FakeConv2D.Stride);
        OrtKISharp.Tensor padding = context.GetOrtArgumentValue(target, FakeConv2D.Padding);
        Tensor<long> dilation = context.GetArgumentValueAsTensor<long>(target, FakeConv2D.Dilation);
        long groups = context.GetArgumentValueAsScalar<long>(target, FakeConv2D.Groups);
        return Value.FromConst(Visit(context, input, weights, act, stride, padding, dilation, groups));
    }

    public IRType Visit(ITypeInferenceContext context, FakeConv2D target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, FakeConv2D.Input);
        TensorType weights = context.CheckArgumentType<TensorType>(target, FakeConv2D.Weights);
        context.CheckArgumentType<IRType>(target, FakeConv2D.Input);
        context.CheckArgumentType<IRType>(target, FakeConv2D.Weights);
        context.CheckArgumentType<IRType>(target, FakeConv2D.Act);
        context.CheckArgumentType<IRType>(target, FakeConv2D.Padding);
        context.CheckArgumentType<IRType>(target, FakeConv2D.Stride);
        context.CheckArgumentType<IRType>(target, FakeConv2D.Dilation);
        context.CheckArgumentType<IRType>(target, FakeConv2D.Groups);
        context.CheckArgumentType<IRType>(target, FakeConv2D.Value);
        return Visit(context, target, input, weights);
    }

    /// <summary>
    /// Fake-quantizes one value: quantize with <paramref name="param"/>, round (unless the parameter is the
    /// identity scale 1 / zero point 0) and de-quantize back to float.
    /// </summary>
    private static float FakeQuantize(float value, QuantParam param)
    {
        double quantized = (double)value / (double)param.Scale + (double)param.ZeroPoint;
        if (param.Scale != 1f || param.ZeroPoint != 0)
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

    private Const Visit(IEvaluateContext context, OrtKISharp.Tensor input, OrtKISharp.Tensor weights, Tensor act,
        Tensor<long> stride, OrtKISharp.Tensor padding, Tensor<long> dilation, long groups)
    {
        MarkerPattern markerPattern = Utility.IsRangeOfMarker(Utility.IsWildcard(), Utility.IsWildcard());

        // Mixed-precision search: simulate the quantization of the (marker wrapped) input and weights.
        if (context.CurrentCall.EnodeBestQuantConfigWithCosine != null)
        {
            if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[0]))
            {
                MixQuantInfo? inputMixQuantInfo = ((Marker)context.CurrentCall.Arguments[0]).MixQuantInfo;
                if (inputMixQuantInfo != null && inputMixQuantInfo.HasBindedMixQuantInfo)
                {
                    List<QuantParam> inputQuantParams =
                        ((Marker)context.CurrentCall.Arguments[0]).MixQuantInfo.QuantParameter;
                    Trace.Assert(inputQuantParams.Count == 1);
                    input = FakeQuantizePerTensor(input, inputQuantParams[0]);
                }
            }

            if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[1]))
            {
                MixQuantInfo? weightsMixQuantInfo = ((Marker)context.CurrentCall.Arguments[1]).MixQuantInfo;
                if (weightsMixQuantInfo != null && weightsMixQuantInfo.HasBindedMixQuantInfo)
                {
                    List<QuantParam> weightsQuantParams =
                        ((Marker)context.CurrentCall.Arguments[1]).MixQuantInfo.QuantParameter;
                    weights = FakeQuantizePerChannel(weights, weightsQuantParams);
                }
            }
        }

        // Adaptive-round quantization: fake-quantize the input with the marker's input quant parameter.
        if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[0]))
        {
            AdaQuantInfo? adaQuantInfo = ((Marker)context.CurrentCall.Arguments[0]).AdaQuantInfo;
            if (adaQuantInfo != null)
            {
                QuantParam inputQuantParam =
                    ((Marker)context.CurrentCall.Arguments[0]).AdaQuantInfo.InputQuantParameter;
                input = FakeQuantizePerTensor(input, inputQuantParam);
            }
        }

        // The bias is always zero; the fake activation below applies the real per-channel bias.
        Tensor convOutput = OrtKI.Conv(input, weights, K230Kernels.ZeroBias((int)weights.Shape[0]), "NOTSET",
            dilation.ToArray(), groups, new long[2] { weights.Shape[2], weights.Shape[3] },
            EvaluatorUtil.ToOnnxPadFormat(padding), stride.ToArray()).ToTensor();

        // Remember the float conv result as the reference tensor for adaptive rounding.
        if (context.CurrentCall.Arguments[0] is Marker)
        {
            if (((Marker)context.CurrentCall.Arguments[0]).AdaQuantInfo == null)
            {
                ((Marker)context.CurrentCall.Arguments[0]).AdaQuantInfo = new AdaQuantInfo();
            }

            ((Marker)context.CurrentCall.Arguments[0]).AdaQuantInfo.AdaRoundRefTensor = convOutput;
        }

        // Per-channel activation; the output is NCHW so the channel is the flat index / (H * W).
        float[] convValues = convOutput.ToArray<float>();
        float[] activated = new float[K230Kernels.ComputeSize(convOutput.Shape)];
        int channelStride = convOutput.Shape[2].FixedValue * convOutput.Shape[3].FixedValue;
        for (int i = 0; i < convValues.Length; i++)
        {
            int channel = i / channelStride;
            activated[i] = K230Kernels.FakeApplyAct0(convValues[i], act.ToArray<float>(), channel, 0);
        }

        return Const.FromTensor(Tensor.From(activated, convOutput.Shape));
    }

    private IRType Visit(ITypeInferenceContext context, FakeConv2D target, TensorType input, TensorType weights)
    {
        Expr[] arguments = context.GetArguments(target, FakeConv2D.Stride, FakeConv2D.Padding, FakeConv2D.Dilation,
            FakeConv2D.Groups);
        return TypeInference.Conv2DType(input, weights, arguments[0], arguments[1], arguments[2], arguments[3]);
    }
}
