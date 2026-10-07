// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#define TRACE
using System;
using System.Collections.Generic;
using System.Diagnostics;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.PatternMatch;
using Nncase.TIR.Instructions;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class FakeAi2dResizeEvaluator : IEvaluator<FakeAi2dResize>, IEvaluator, ITypeInferencer<FakeAi2dResize>,
    ITypeInferencer, ICostEvaluator<FakeAi2dResize>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, FakeAi2dResize target)
    {
        TensorType inputType = context.GetArgumentType<TensorType>(target, FakeAi2dResize.Input);
        TensorType returnType = context.GetReturnType<TensorType>();
        return new Cost
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(returnType)
        };
    }

    public IValue Visit(IEvaluateContext context, FakeAi2dResize r)
    {
        Tensor input = context.GetArgumentValueAsTensor(r, FakeAi2dResize.Input);

        // [height, width]
        int[] newSize = context.GetArgumentValueAsArray<int>(r, FakeAi2dResize.NewSize);
        input = FakeQuantizeInput(context, input);

        if (r.ResizeMethod == MFU_CROP_RESIZE.BILINER)
        {
            return Value.FromTensor(K230Kernels.FakeAi2dResizeBilinear(input, newSize, r.AlignCorners,
                r.HalfPixelCenters));
        }

        return Value.FromTensor(K230Kernels.FakeAi2dResizeNearestNeighbor(input, newSize, r.AlignCorners,
            r.HalfPixelCenters));
    }

    public IRType Visit(ITypeInferenceContext context, FakeAi2dResize target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, FakeAi2dResize.Input);
        context.CheckArgumentType<IRType>(target, FakeAi2dResize.Input);
        context.CheckArgumentType<IRType>(target, FakeAi2dResize.NewSize);
        return Visit(context, target, input);
    }

    /// <summary>
    /// Replaces the input by its quantize-dequantize round trip when the producing marker carries bound mixed
    /// quantization info; otherwise returns the input unchanged.
    /// </summary>
    private static Tensor FakeQuantizeInput(IEvaluateContext context, Tensor input)
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

                input = Value.FromTensor(Tensor.From(values, input.Shape)).AsTensor();
            }
        }

        return input;
    }

    private IRType Visit(ITypeInferenceContext context, FakeAi2dResize target, TensorType input)
    {
        if (input.Shape[2] == 1 && input.Shape[3] == 1)
        {
            return new InvalidType("FakeAi2dResize doesn't support 1x1 input");
        }

        Expr newSize = context.GetArgument(target, FakeAi2dResize.NewSize);
        return TypeInference.ResizeType(input, newSize, null);
    }
}
