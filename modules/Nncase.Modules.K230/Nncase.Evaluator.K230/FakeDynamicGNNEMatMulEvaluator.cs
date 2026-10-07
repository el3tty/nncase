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

[EvaluatorGenerator]
[TypeInferGenerator]
public sealed class FakeDynamicGNNEMatMulEvaluator : IEvaluator<FakeDynamicGNNEMatMul>, IEvaluator,
    ITypeInferencer<FakeDynamicGNNEMatMul>, ITypeInferencer, ICostEvaluator<FakeDynamicGNNEMatMul>, ICostEvaluator
{
    /// <summary>Divisor applied to the reduction size to get the CPU cycle factor.</summary>
    private const float CpuCyclesDivisor = 768f;

    // Positions of the marker-wrapped operands in the call's argument list.
    private const int InputAArgumentIndex = 0;
    private const int InputBArgumentIndex = 1;

    public Cost Visit(ICostEvaluateContext context, FakeDynamicGNNEMatMul target)
    {
        TensorType inputAType = context.GetArgumentType<TensorType>(target, FakeDynamicGNNEMatMul.InputA);
        TensorType inputBType = context.GetArgumentType<TensorType>(target, FakeDynamicGNNEMatMul.InputB);
        TensorType actType = context.GetArgumentType<TensorType>(target, FakeDynamicGNNEMatMul.Act);
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

        return new Cost
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputAType) +
                                           CostUtility.GetMemoryAccess(inputBType) +
                                           CostUtility.GetMemoryAccess(actType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(returnType, (float)aCols / CpuCyclesDivisor),
        };
    }

    /// <summary>Quantizes (rounding unless the parameter is the identity) and dequantizes one value.</summary>
    private static float FakeQuantize(float value, QuantParam quantParam)
    {
        double quantized = (double)value / (double)quantParam.Scale + (double)quantParam.ZeroPoint;
        if (quantParam.Scale != 1f || quantParam.ZeroPoint != 0)
        {
            quantized = System.Math.Round(quantized);
        }

        return (float)((quantized - (double)quantParam.ZeroPoint) * (double)quantParam.Scale);
    }

    private IValue Visit(IEvaluateContext context, FakeDynamicGNNEMatMul op, Tensor<float> inputA, Tensor<float> inputB,
        Tensor<float> act)
    {
        // Batch dimensions of each input, padded at the end with 1 up to two entries.
        // NOTE: only the first two batch dimensions are passed to the kernel (and they are not left-aligned for
        // broadcasting here, unlike the output shape computed below).
        int[] aBatchDims = inputA.Dimensions.ToArray().SkipLast(2).TakeOrDefault(2, 1)
            .ToArray();
        ReadOnlySpan<int> inputADims = inputA.Dimensions;
        int aRows = inputADims[inputADims.Length - 2];
        int aCols = inputADims[inputADims.Length - 1];
        int[] bBatchDims = inputB.Dimensions.ToArray().SkipLast(2).TakeOrDefault(2, 1)
            .ToArray();
        ReadOnlySpan<int> inputBDims = inputB.Dimensions;
        int bCols = inputBDims[inputBDims.Length - 1];

        // Output shape: broadcast batch dimensions (left-padded with 1) followed by [aRows, bCols].
        List<int> aBatchShape = inputA.Dimensions.ToArray().SkipLast(2).ToList();
        List<int> bBatchShape = inputB.Dimensions.ToArray().SkipLast(2).ToList();
        while (aBatchShape.Count < bBatchShape.Count)
        {
            aBatchShape.Insert(0, 1);
        }

        while (bBatchShape.Count < aBatchShape.Count)
        {
            bBatchShape.Insert(0, 1);
        }

        Tensor<float> output = new Tensor<float>((from p in aBatchShape.Zip(bBatchShape)
            select System.Math.Max(p.First, p.Second)).Concat(new int[2] { aRows, bCols }).ToArray());

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
                    Span<float> inputAValues = inputA.Buffer.Span;
                    for (int i = 0; i < inputA.Length; i++)
                    {
                        inputAValues[i] = FakeQuantize(inputAValues[i], quantParamsA[0]);
                    }
                }
            }

            if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[InputBArgumentIndex]))
            {
                MixQuantInfo? mixQuantInfoB = ((Marker)context.CurrentCall.Arguments[InputBArgumentIndex]).MixQuantInfo;
                if (mixQuantInfoB != null && mixQuantInfoB.HasBindedMixQuantInfo)
                {
                    List<QuantParam> quantParamsB =
                        ((Marker)context.CurrentCall.Arguments[InputBArgumentIndex]).MixQuantInfo.QuantParameter;

                    // Input B has one quant parameter per equally sized chunk (per-channel quantization).
                    int chunkLength = inputB.Length / quantParamsB.Count;
                    Span<float> inputBValues = inputB.Buffer.Span;
                    for (int i = 0; i < inputB.Length; i++)
                    {
                        inputBValues[i] = FakeQuantize(inputBValues[i], quantParamsB[i / chunkLength]);
                    }
                }
            }
        }

        Span<float> inputAData = inputA.Buffer.Span;
        Span<float> inputBData = inputB.Buffer.Span;
        Span<float> outputData = output.Buffer.Span;
        K230Kernels.FakeDynamicGnneMatmul(context, inputAData, inputBData, outputData, act.Buffer.Span,
            aBatchDims[0], aBatchDims[1], aRows, aCols, bBatchDims[0], bBatchDims[1], bCols, op.DynamicChannel);
        return Value.FromTensor(output);
    }

    private IRType Visit(TensorType inputA, TensorType inputB, TensorType act)
    {
        if (inputA.Shape.IsUnranked || inputB.Shape.IsUnranked)
        {
            return new InvalidType("Shape InputA or InputB Can't Be Unranked");
        }

        if (inputA.Shape.Rank < 2 || inputB.Shape.Rank < 2)
        {
            return new InvalidType("Rank InputA and InputB Must >= 2!");
        }

        // Batch dimensions are left-padded with 1 and broadcast; unknown dimensions stay unknown.
        List<Dimension> aBatchShape = inputA.Shape.SkipLast(2).ToList();
        List<Dimension> bBatchShape = inputB.Shape.SkipLast(2).ToList();
        while (bBatchShape.Count < aBatchShape.Count)
        {
            bBatchShape.Insert(0, 1);
        }

        while (aBatchShape.Count < bBatchShape.Count)
        {
            aBatchShape.Insert(0, 1);
        }

        IEnumerable<Dimension> batchDims = aBatchShape.Zip(bBatchShape).Select(delegate((Dimension First, Dimension Second) p)
        {
            var (aDim, bDim) = p;
            return (aDim.Kind == DimensionKind.Fixed && bDim.Kind == DimensionKind.Fixed)
                ? ((Dimension)System.Math.Max(aDim.FixedValue, bDim.FixedValue))
                : Dimension.Unknown;
        });

        // Trailing [rows of A, cols of B].
        Dimension[] matrixDims = new Dimension[2];
        Shape inputAShape = inputA.Shape;
        matrixDims[0] = inputAShape[inputAShape.Count - 2];
        Shape inputBShape = inputB.Shape;
        matrixDims[1] = inputBShape[inputBShape.Count - 1];
        return new TensorType(inputA.DType, batchDims.Concat(matrixDims).ToArray());
    }

    public IValue Visit(IEvaluateContext context, FakeDynamicGNNEMatMul target)
    {
        Tensor<float> inputA = context.GetArgumentValueAsTensor<float>(target, FakeDynamicGNNEMatMul.InputA);
        Tensor<float> inputB = context.GetArgumentValueAsTensor<float>(target, FakeDynamicGNNEMatMul.InputB);
        Tensor<float> act = context.GetArgumentValueAsTensor<float>(target, FakeDynamicGNNEMatMul.Act);
        return Visit(context, target, inputA, inputB, act);
    }

    public IRType Visit(ITypeInferenceContext context, FakeDynamicGNNEMatMul target)
    {
        TensorType inputA = context.CheckArgumentType<TensorType>(target, FakeDynamicGNNEMatMul.InputA);
        TensorType inputB = context.CheckArgumentType<TensorType>(target, FakeDynamicGNNEMatMul.InputB);
        TensorType act = context.CheckArgumentType<TensorType>(target, FakeDynamicGNNEMatMul.Act);
        context.CheckArgumentType<IRType>(target, FakeDynamicGNNEMatMul.InputA);
        context.CheckArgumentType<IRType>(target, FakeDynamicGNNEMatMul.InputB);
        context.CheckArgumentType<IRType>(target, FakeDynamicGNNEMatMul.Act);
        return Visit(inputA, inputB, act);
    }
}
