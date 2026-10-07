// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#define TRACE
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using DryIoc;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.PatternMatch;
using OrtKISharp;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class FakePdpEvaluator : IEvaluator<FakePdp>, IEvaluator, ITypeInferencer<FakePdp>, ITypeInferencer,
    ICostEvaluator<FakePdp>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, FakePdp target)
    {
        TensorType inputType = context.GetArgumentType<TensorType>(target, FakePdp.Input);
        TensorType returnType = context.GetReturnType<TensorType>();
        float cpuCycleNumerator = 1f;
        float cpuCycleDenominator = 5f;
        return new Cost
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(returnType, cpuCycleNumerator / cpuCycleDenominator)
        };
    }

    public IValue Visit(IEvaluateContext context, FakePdp r)
    {
        OrtKISharp.Tensor input = context.GetOrtArgumentValue(r, FakePdp.Input);
        long[] filter = context.GetArgumentValueAsArray<long>(r, FakePdp.Filter);
        long[] stride = context.GetArgumentValueAsArray<long>(r, FakePdp.Stride);
        long[] padding = context.GetArgumentValueAsArray<long>(r, FakePdp.Padding);
        input = FakeQuantizeInput(context, input);

        long[] dilations = Enumerable.Repeat(1L, filter.Length).ToArray();

        // NOTE: the count-include-pad and pad value arguments are ignored; MIN and SUM are not supported.
        return (r.ReduceOp switch
        {
            ReduceOp.Min => throw new NotSupportedException("Unsupported MFU_PDP_OP MIN"),

            // MaxPool(x, auto_pad, ceil_mode, dilations, kernel_shape, pads, storage_order, strides)
            ReduceOp.Max => OrtKI.MaxPool(input, "NOTSET", 0L, new long[2] { 1L, 1L }, filter, padding, 0L,
                stride)[0],

            // AveragePool(x, auto_pad, ceil_mode, count_include_pad, dilations, kernel_shape, pads, strides)
            ReduceOp.Mean => OrtKI.AveragePool(input, "NOTSET", 0L, 0L, dilations, filter, padding, stride),
            ReduceOp.Sum => throw new NotSupportedException("Unsupported MFU_PDP_OP SUM"),
            _ => throw new ArgumentOutOfRangeException("r"),
        }).ToValue();
    }

    public IRType Visit(ITypeInferenceContext context, FakePdp target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, FakePdp.Input);
        context.CheckArgumentType<IRType>(target, FakePdp.Input);
        context.CheckArgumentType<IRType>(target, FakePdp.PadValue);
        context.CheckArgumentType<IRType>(target, FakePdp.Filter);
        context.CheckArgumentType<IRType>(target, FakePdp.Stride);
        context.CheckArgumentType<IRType>(target, FakePdp.Padding);
        context.CheckArgumentType<IRType>(target, FakePdp.CountIncludePad);
        return Visit(context, target, input);
    }

    /// <summary>
    /// Replaces the input by its quantize-dequantize round trip when the producing marker carries bound mixed
    /// quantization info; otherwise returns the input unchanged.
    /// </summary>
    private static OrtKISharp.Tensor FakeQuantizeInput(IEvaluateContext context, OrtKISharp.Tensor input)
    {
        if (context.CurrentCall.EnodeBestQuantConfigWithCosine != null && Utility
                .IsRangeOfMarker(Utility.IsWildcard(), Utility.IsWildcard())
                .MatchLeaf(context.CurrentCall.Arguments[0]))
        {
            MixQuantInfo? mixQuantInfo = ((Marker)context.CurrentCall.Arguments[0]).MixQuantInfo;
            if (mixQuantInfo != null && mixQuantInfo.HasBindedMixQuantInfo)
            {
                List<QuantParam> quantParameter =
                    ((Marker)context.CurrentCall.Arguments[0]).MixQuantInfo.QuantParameter;
                Trace.Assert(quantParameter.Count == 1);
                float[] values = input.ToArray<float>();
                for (int i = 0; i < values.Length; i++)
                {
                    double quantized = (double)values[i] / (double)quantParameter[0].Scale +
                                       (double)quantParameter[0].ZeroPoint;

                    // Rounding is skipped for the identity quant param.
                    if (!quantParameter[0].Scale.Equals(1f) || quantParameter[0].ZeroPoint != 0)
                    {
                        quantized = System.Math.Round(quantized);
                    }

                    double dequantized = (quantized - (double)quantParameter[0].ZeroPoint) *
                                         (double)quantParameter[0].Scale;
                    values[i] = (float)dequantized;
                }

                input = OrtKISharp.Tensor.MakeTensor(values, input.Shape);
            }
        }

        return input;
    }

    private IRType Visit(ITypeInferenceContext context, FakePdp target, TensorType input)
    {
        Expr[] windowArgs = context.GetArguments(target, FakePdp.Filter, FakePdp.Stride, FakePdp.Padding);
        IRType windowedType = TypeInference.ReduceWindow2DType(input, windowArgs[0], windowArgs[1], windowArgs[2],
            false);
        return new TensorType(DataTypes.Float32, ((TensorType)windowedType).Shape);
    }
}
