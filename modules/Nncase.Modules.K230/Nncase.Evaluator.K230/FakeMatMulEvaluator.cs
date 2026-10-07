#define TRACE
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.PatternMatch;
using OrtKISharp;

namespace Nncase.Evaluator.K230;

[EvaluatorGenerator]
[TypeInferGenerator]
public sealed class FakeMatMulEvaluator : IEvaluator<FakeMatMul>, IEvaluator, ITypeInferencer<FakeMatMul>,
    ITypeInferencer, ICostEvaluator<FakeMatMul>, ICostEvaluator
{
    /// <summary>Divisor applied to the reduction size to get the CPU cycle factor.</summary>
    private const float CpuCyclesDivisor = 768f;

    // Positions of the marker-wrapped operands in the call's argument list.
    private const int InputAArgumentIndex = 0;
    private const int InputBArgumentIndex = 1;

    public Cost Visit(ICostEvaluateContext context, FakeMatMul target)
    {
        TensorType inputAType = context.GetArgumentType<TensorType>(target, FakeMatMul.InputA);
        TensorType inputBType = context.GetArgumentType<TensorType>(target, FakeMatMul.InputB);
        TensorType actType = context.GetArgumentType<TensorType>(target, FakeMatMul.Act);
        TensorType returnType = context.GetReturnType<TensorType>();

        // Reduction size (columns of input A); 1 when it is not statically known.
        Shape inputAShape = inputAType.Shape;
        int aCols;
        if (!inputAShape[inputAShape.Count - 1].IsFixed)
        {
            aCols = 1;
        }
        else
        {
            Shape shape = inputAType.Shape;
            aCols = shape[shape.Count - 1].FixedValue;
        }

        uint aColsUnsigned = (uint)aCols;
        return new Cost
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputAType) +
                                           CostUtility.GetMemoryAccess(inputBType) +
                                           CostUtility.GetMemoryAccess(actType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(returnType, (float)aColsUnsigned / CpuCyclesDivisor),
        };
    }

    /// <summary>Quantizes (rounding unless the parameter is the identity) and dequantizes one value.</summary>
    private static float FakeQuantize(float value, QuantParam quantParam)
    {
        double quantized = (double)value / (double)quantParam.Scale + (double)quantParam.ZeroPoint;
        if (!quantParam.Scale.Equals(1f) || quantParam.ZeroPoint != 0)
        {
            quantized = System.Math.Round(quantized);
        }

        return (float)((quantized - (double)quantParam.ZeroPoint) * (double)quantParam.Scale);
    }

    private IValue Visit(IEvaluateContext context, OrtKISharp.Tensor inputA, OrtKISharp.Tensor inputB,
        Tensor<float> act)
    {
        if (context.CurrentCall.EnodeBestQuantConfigWithCosine != null)
        {
            // Replace the inputs by their quantize-dequantize round trip when the markers carry mix-quant info.
            MarkerPattern markerPattern = Utility.IsRangeOfMarker(Utility.IsWildcard(), Utility.IsWildcard());
            if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[InputAArgumentIndex]))
            {
                MixQuantInfo? mixQuantInfoA = ((Marker)context.CurrentCall.Arguments[InputAArgumentIndex]).MixQuantInfo;
                if (mixQuantInfoA != null && mixQuantInfoA.HasBindedMixQuantInfo)
                {
                    List<QuantParam> quantParamsA =
                        ((Marker)context.CurrentCall.Arguments[InputAArgumentIndex]).MixQuantInfo.QuantParameter;

                    // Input A has a single per-tensor quant parameter.
                    Trace.Assert(quantParamsA.Count == 1);
                    float[] inputAValues = inputA.ToArray<float>();
                    for (int i = 0; i < inputAValues.Length; i++)
                    {
                        inputAValues[i] = FakeQuantize(inputAValues[i], quantParamsA[0]);
                    }

                    inputA = OrtKISharp.Tensor.MakeTensor(inputAValues, inputA.Shape);
                }
            }

            if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[InputBArgumentIndex]))
            {
                MixQuantInfo? mixQuantInfoB = ((Marker)context.CurrentCall.Arguments[InputBArgumentIndex]).MixQuantInfo;
                if (mixQuantInfoB != null && mixQuantInfoB.HasBindedMixQuantInfo)
                {
                    List<QuantParam> quantParamsB =
                        ((Marker)context.CurrentCall.Arguments[InputBArgumentIndex]).MixQuantInfo?.QuantParameter;

                    // Input B has one quant parameter per equally sized chunk (per-channel quantization).
                    float[] inputBValues = inputB.ToArray<float>();
                    int chunkLength = inputBValues.Length / quantParamsB.Count;
                    for (int i = 0; i < inputBValues.Length; i++)
                    {
                        inputBValues[i] = FakeQuantize(inputBValues[i], quantParamsB[i / chunkLength]);
                    }

                    inputB = OrtKISharp.Tensor.MakeTensor(inputBValues, inputB.Shape);
                }
            }
        }

        // Dimension 1 is the channel dimension; it must match or one side must be 1 (broadcast).
        long inputAChannels = inputA.Shape[1];
        long inputBChannels = inputB.Shape[1];
        Tensor matMulResult = OrtKI.MatMul(inputA, inputB).ToTensor();
        float[] matMulData = matMulResult.ToArray<float>();
        float[] outputData = new float[K230Kernels.ComputeSize(matMulResult.Shape)];
        if (inputAChannels == inputBChannels || (inputAChannels > inputBChannels && inputBChannels == 1) ||
            (inputAChannels < inputBChannels && inputAChannels == 1))
        {
            // Activation parameters are selected per channel: one block of rows * cols results per channel.
            float[] actData = act.ToArray<float>();
            for (int k = 0; k < outputData.Length; k++)
            {
                outputData[k] = K230Kernels.FakeApplyAct0(matMulData[k], actData,
                    k / (matMulResult.Shape[2].FixedValue * matMulResult.Shape[3].FixedValue), 0);
            }

            return Value.FromTensor(Tensor.From(outputData, matMulResult.Shape));
        }

        // NOTE: an unrelated exception type is thrown for incompatible channel counts (kept as is).
        throw new InvalidOleVariantTypeException("Invalid matmul");
    }

    private IRType Visit(TensorType inputA, TensorType inputB)
    {
        if (inputA.Shape.IsUnranked || inputB.Shape.IsUnranked)
        {
            return new InvalidType("Shape InputA or InputB Can't Be Unranked");
        }

        if (inputA.Shape[3] != inputB.Shape[2])
        {
            return new InvalidType("FakeMatMul input a's cols must be equal to input b's rows");
        }

        // [max batch, max channels, rows of A, cols of B].
        Shape shape = new Shape(System.Math.Max(inputA.Shape[0].FixedValue, inputB.Shape[0].FixedValue),
            System.Math.Max(inputA.Shape[1].FixedValue, inputB.Shape[1].FixedValue), inputA.Shape[2], inputB.Shape[3]);
        return new TensorType(inputA.DType, shape);
    }

    public IValue Visit(IEvaluateContext context, FakeMatMul target)
    {
        OrtKISharp.Tensor inputA = context.GetOrtArgumentValue(target, FakeMatMul.InputA);
        OrtKISharp.Tensor inputB = context.GetOrtArgumentValue(target, FakeMatMul.InputB);
        Tensor<float> act = context.GetArgumentValueAsTensor<float>(target, FakeMatMul.Act);
        return Visit(context, inputA, inputB, act);
    }

    public IRType Visit(ITypeInferenceContext context, FakeMatMul target)
    {
        TensorType inputA = context.CheckArgumentType<TensorType>(target, FakeMatMul.InputA);
        TensorType inputB = context.CheckArgumentType<TensorType>(target, FakeMatMul.InputB);
        context.CheckArgumentType<IRType>(target, FakeMatMul.InputA);
        context.CheckArgumentType<IRType>(target, FakeMatMul.InputB);
        context.CheckArgumentType<IRType>(target, FakeMatMul.Act);
        return Visit(inputA, inputB);
    }
}
