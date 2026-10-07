#define TRACE
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.PatternMatch;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class FakeActivationEvaluator : IEvaluator<FakeActivation>, IEvaluator, ITypeInferencer<FakeActivation>,
    ITypeInferencer, ICostEvaluator<FakeActivation>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, FakeActivation target)
    {
        TensorType inputAType = context.GetArgumentType<TensorType>(target, FakeActivation.InputA);
        IRType inputBType = context.GetArgumentType<IRType>(target, FakeActivation.InputB);
        TensorType returnType = context.GetReturnType<TensorType>();
        return new Cost
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputAType) +
                                           ((inputBType is TensorType type)
                                               ? CostUtility.GetMemoryAccess(type)
                                               : ((UInt128)(byte)0)),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(returnType, 0.125)
        };
    }

    public IValue Visit(IEvaluateContext context, FakeActivation a)
    {
        Tensor inputA = context.GetArgumentValueAsTensor(a, FakeActivation.InputA);
        IValue inputB = context.GetArgumentValue(a, FakeActivation.InputB);

        // Input B is optional: either absent or an empty tensor.
        bool hasUninitializedInput = inputB is NoneValue || ((TensorType)inputB.Type).Shape.Size == 0;
        Tensor act = context.GetArgumentValueAsTensor(a, FakeActivation.Act);
        bool[] is16Segments = context.GetArgumentValueAsTensor(a, FakeActivation.Is16Segments).ToArray<bool>();
        int[] outChannels = context.GetArgumentValueAsTensor(a, FakeActivation.OutChannels).ToArray<int>();
        Shape outputShape = context.CurrentCall.CheckedShape;
        if (context.CurrentCall.EnodeBestQuantConfigWithCosine != null)
        {
            MarkerPattern markerPattern = Utility.IsRangeOfMarker(Utility.IsWildcard(), Utility.IsWildcard());
            if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[0]))
            {
                MixQuantInfo? mixQuantInfoA = ((Marker)context.CurrentCall.Arguments[0]).MixQuantInfo;
                if (mixQuantInfoA != null && mixQuantInfoA.HasBindedMixQuantInfo)
                {
                    List<QuantParam> quantParameter =
                        ((Marker)context.CurrentCall.Arguments[0]).MixQuantInfo.QuantParameter;
                    Trace.Assert(quantParameter.Count == 1);
                    float[] valuesA = inputA.ToArray<float>();
                    FakeQuantizeInPlace(valuesA, quantParameter);
                    inputA = Value.FromTensor(Tensor.From(valuesA, inputA.Shape)).AsTensor();
                }
            }

            if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[1]))
            {
                MixQuantInfo? mixQuantInfoB = ((Marker)context.CurrentCall.Arguments[1]).MixQuantInfo;
                if (mixQuantInfoB != null && mixQuantInfoB.HasBindedMixQuantInfo)
                {
                    List<QuantParam> quantParameter =
                        ((Marker)context.CurrentCall.Arguments[1]).MixQuantInfo.QuantParameter;
                    Trace.Assert(quantParameter.Count == 1);
                    float[] valuesB = inputB.AsTensor().ToArray<float>();
                    FakeQuantizeInPlace(valuesB, quantParameter);
                    inputB = Value.FromTensor(Tensor.From(valuesB, inputB.AsTensor().Shape));
                }
            }
        }

        return Value.FromConst(K230Kernels.FakeGnneActivation(is16Segments[0], inputA,
            hasUninitializedInput ? new Tensor<float>(0) : inputB.AsTensor(), hasUninitializedInput, outputShape,
            act.ToArray<float>(), outChannels[0], a.Type));
    }

    public IRType Visit(ITypeInferenceContext context, FakeActivation target)
    {
        TensorType inputA = context.CheckArgumentType<TensorType>(target, FakeActivation.InputA);
        context.CheckArgumentType<IRType>(target, FakeActivation.InputA);
        context.CheckArgumentType<IRType>(target, FakeActivation.InputB);
        context.CheckArgumentType<IRType>(target, FakeActivation.Act);
        context.CheckArgumentType<IRType>(target, FakeActivation.OutChannels);
        context.CheckArgumentType<IRType>(target, FakeActivation.InAShiftBits);
        context.CheckArgumentType<IRType>(target, FakeActivation.InBShiftBits);
        context.CheckArgumentType<IRType>(target, FakeActivation.OutShiftBits);
        context.CheckArgumentType<IRType>(target, FakeActivation.Is16Segments);
        return Visit(context, target, inputA);
    }

    /// <summary>Replaces every value by its quantize-dequantize round trip using the first quant param.</summary>
    private static void FakeQuantizeInPlace(float[] values, List<QuantParam> quantParameter)
    {
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
    }

    private IRType Visit(ITypeInferenceContext context, FakeActivation target, TensorType inputA)
    {
        if (!(context.GetArgument(target, FakeActivation.OutChannels) is Const))
        {
            return new InvalidType("FakeActivation out_channels need a constant value");
        }

        return new TensorType(inputA.DType, target.OutputShape.ToArray());
    }
}
