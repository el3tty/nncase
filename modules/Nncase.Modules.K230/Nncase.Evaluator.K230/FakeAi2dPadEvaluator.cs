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

[TypeInferGenerator]
public class FakeAi2dPadEvaluator : IEvaluator<FakeAi2dPad>, IEvaluator, ITypeInferencer<FakeAi2dPad>, ITypeInferencer,
    ICostEvaluator<FakeAi2dPad>, ICostEvaluator
{
    // ONNX TensorProto data type code of float32, the target of the final Cast.
    private const long OnnxFloat32 = 1L;

    public IValue Visit(IEvaluateContext context, FakeAi2dPad r)
    {
        OrtKISharp.Tensor input = context.GetArgumentValue(r, FakeAi2dPad.Input).AsTensor().Cast<float>()
            .ToOrtTensor();
        OrtKISharp.Tensor padding = context.GetInt64OrtTensorArgumentValue(r, FakeAi2dPad.Padding);
        OrtKISharp.Tensor padValue = context.GetArgumentValue(r, FakeAi2dPad.Value).AsTensor().Cast<float>()
            .ToOrtTensor();
        input = FakeQuantizeInput(context, input);

        if (r.Mode == PadMode.Symmetric)
        {
            throw new NotImplementedException();
        }

        // NOTE: string.ToLower(null) is kept as decompiled (it looks like it should be ToLower()).
        return OrtKI
            .Cast(
                OrtKI.Pad(input, (OrtKISharp.Tensor)EvaluatorUtil.ToOnnxPadFormat(padding), padValue, null,
                    r.Mode.ToString().ToLower(null)), 1, OnnxFloat32).ToValue();
    }

    public Cost Visit(ICostEvaluateContext context, FakeAi2dPad target)
    {
        TensorType inputType = context.GetArgumentType<TensorType>(target, FakeAi2dPad.Input);
        TensorType returnType = context.GetReturnType<TensorType>();
        return new Cost
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) / (byte)2,
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType) / (byte)2
        };
    }

    public IRType Visit(ITypeInferenceContext context, FakeAi2dPad target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, FakeAi2dPad.Input);
        context.CheckArgumentType<IRType>(target, FakeAi2dPad.Input);
        context.CheckArgumentType<IRType>(target, FakeAi2dPad.Padding);
        context.CheckArgumentType<IRType>(target, FakeAi2dPad.Value);
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

    private IRType Visit(ITypeInferenceContext context, FakeAi2dPad target, TensorType input)
    {
        Expr padding = context.GetArgument(target, FakeAi2dPad.Padding);
        Expr padValue = context.GetArgument(target, FakeAi2dPad.Value);
        return TypeInference.PadType(input, padding, padValue);
    }
}
