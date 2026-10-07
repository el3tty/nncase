// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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
public class FakeLSTMEvaluator : IEvaluator<FakeLSTM>, IEvaluator, ITypeInferencer<FakeLSTM>, ITypeInferencer,
    ICostEvaluator<FakeLSTM>, ICostEvaluator
{
    // Positions of the marker-wrapped operands in the call's argument list.
    private const int InputArgumentIndex = 0;
    private const int InitialHArgumentIndex = 5;

    public Cost Visit(ICostEvaluateContext context, FakeLSTM target)
    {
        TensorType inputType = context.GetArgumentType<TensorType>(target, FakeLSTM.Input);
        TensorType inputWeightsType = context.GetArgumentType<TensorType>(target, FakeLSTM.WXc);
        TensorType recurrentWeightsType = context.GetArgumentType<TensorType>(target, FakeLSTM.WRc);
        TupleType returnType = context.GetReturnType<TupleType>();
        return new Cost
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) +
                                           CostUtility.GetMemoryAccess(inputWeightsType) +
                                           CostUtility.GetMemoryAccess(recurrentWeightsType),
            [CostFactorNames.MemoryStore] = returnType.Select((IRType t) =>
                (t is TensorType type) ? CostUtility.GetMemoryAccess(type) : UInt128.One).Sum(),
        };
    }

    public IValue Visit(IEvaluateContext context, FakeLSTM l)
    {
        Tensor input = context.GetArgumentValueAsTensor(l, FakeLSTM.Input);
        Tensor<float> inputWeights = context.GetArgumentValueAsTensor<float>(l, FakeLSTM.WXc);
        Tensor<Half> inputActivation = context.GetArgumentValueAsTensor<Half>(l, FakeLSTM.ActXc);
        Tensor<float> recurrentWeights = context.GetArgumentValueAsTensor<float>(l, FakeLSTM.WRc);
        Tensor<Half> recurrentActivation = context.GetArgumentValueAsTensor<Half>(l, FakeLSTM.ActRc);
        Tensor initialHidden = context.GetArgumentValueAsTensor(l, FakeLSTM.InitialH);
        Tensor<float> initialCell = context.GetArgumentValueAsTensor<float>(l, FakeLSTM.InitialC);
        Tensor<Half> sigmoidFit = context.GetArgumentValueAsTensor<Half>(l, FakeLSTM.SegFittingParamFt);
        Tensor<Half> tanhFit = context.GetArgumentValueAsTensor<Half>(l, FakeLSTM.SegFittingParamGt);
        int outputSize = context.GetArgumentValueAsTensor<int>(l, FakeLSTM.OutputSize).ToArray()[0];

        // Output buffers (Y, last hidden, last cell) are shaped after the call's tuple type; the hidden / cell
        // buffers fall back to the Y shape when that output is not requested.
        Tensor<float> output =
            new Tensor<float>(((TensorType)((TupleType)context.CurrentCall.CheckedType)[0]).Shape.ToValueArray());
        Tensor<float> outputH =
            new Tensor<float>(((TensorType)((TupleType)context.CurrentCall.CheckedType)[0]).Shape.ToValueArray());
        Tensor<float> outputC =
            new Tensor<float>(((TensorType)((TupleType)context.CurrentCall.CheckedType)[0]).Shape.ToValueArray());
        if (outputSize >= 2)
        {
            outputH = new Tensor<float>(
                ((TensorType)((TupleType)context.CurrentCall.CheckedType)[1]).Shape.ToValueArray());
        }

        if (outputSize >= 3)
        {
            outputC = new Tensor<float>(
                ((TensorType)((TupleType)context.CurrentCall.CheckedType)[2]).Shape.ToValueArray());
        }

        if (context.CurrentCall.EnodeBestQuantConfigWithCosine != null)
        {
            // Replace the input / initial hidden state by their quantize-dequantize round trip when the markers
            // carry mix-quant info.
            MarkerPattern markerPattern = Utility.IsRangeOfMarker(Utility.IsWildcard(), Utility.IsWildcard());
            if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[InputArgumentIndex]))
            {
                MixQuantInfo? inputMixQuantInfo = ((Marker)context.CurrentCall.Arguments[InputArgumentIndex]).MixQuantInfo;
                if (inputMixQuantInfo != null && inputMixQuantInfo.HasBindedMixQuantInfo)
                {
                    List<QuantParam> inputQuantParams =
                        ((Marker)context.CurrentCall.Arguments[InputArgumentIndex]).MixQuantInfo?.QuantParameter;
                    Trace.Assert(inputQuantParams.Count == 1);
                    float[] inputValues = input.ToArray<float>();
                    for (int i = 0; i < inputValues.Length; i++)
                    {
                        inputValues[i] = FakeQuantize(inputValues[i], inputQuantParams[0]);
                    }

                    input = Tensor.From(inputValues, input.Shape);
                }
            }

            if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[InitialHArgumentIndex]))
            {
                MixQuantInfo? initialHMixQuantInfo =
                    ((Marker)context.CurrentCall.Arguments[InitialHArgumentIndex]).MixQuantInfo;

                // Only a (scalar) zero initial state is quantized.
                if (initialHMixQuantInfo != null && initialHMixQuantInfo.HasBindedMixQuantInfo &&
                    ((Tensor<float>)initialHidden).ToScalar() == 0f)
                {
                    List<QuantParam> initialHQuantParams =
                        ((Marker)context.CurrentCall.Arguments[InitialHArgumentIndex]).MixQuantInfo?.QuantParameter;
                    Trace.Assert(initialHQuantParams.Count == 1);
                    float[] initialHValues = initialHidden.ToArray<float>();
                    for (int i = 0; i < initialHValues.Length; i++)
                    {
                        initialHValues[i] = FakeQuantize(initialHValues[i], initialHQuantParams[0]);
                    }

                    initialHidden = Tensor.From(initialHValues, initialHidden.Shape);
                }
            }
        }

        Tensor[] outputs = K230Kernels.FakeGnneLstm((Tensor<float>)input, inputWeights,
            inputActivation, recurrentWeights, recurrentActivation, (Tensor<float>)initialHidden,
            initialCell, sigmoidFit, tanhFit, output, outputH, outputC,
            l.Direction, outputSize).ToArray();
        return Value.FromTensors(outputs);
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

    private IRType Visit(ITypeInferenceContext context, FakeLSTM target, TensorType input, TensorType initialH,
        TensorType initialC)
    {
        int numDirections = (target.Direction != LSTMDirection.Bidirectional) ? 1 : 2;
        int seqLenIndex = 1;
        if (context.GetArgument(target, FakeLSTM.OutputSize) is TensorConst outputSizeConst)
        {
            TensorType yType = InferYType(context, target, input, seqLenIndex, numDirections);

            // Outputs: Y, last hidden, last cell; keep the first OutputSize of them.
            IRType[] outputTypes =
                (new TensorType[3] { yType, initialH, initialC })[..outputSizeConst.Value.ToScalar<int>()];
            return new TupleType(outputTypes);
        }

        return new InvalidType("LSTM OutputSize Must be known");
    }

    /// <summary>
    /// Y type: the input shape with the sequence dimension split off and the direction count inserted; the last
    /// dimension becomes the hidden size of the recurrent weights.
    /// </summary>
    private TensorType InferYType(ITypeInferenceContext context, FakeLSTM target, TensorType x, int seqLenIndex,
        int numDirections)
    {
        List<Dimension> dims = x.Shape.ToList();
        dims.Insert(seqLenIndex + 1, numDirections);
        int hiddenSize = context.GetArgument(target, FakeLSTM.WRc).CheckedShape[3].FixedValue;
        dims.RemoveAt(0);
        dims[dims.Count - 1] = hiddenSize;
        return x with { Shape = dims.ToArray() };
    }

    public IRType Visit(ITypeInferenceContext context, FakeLSTM target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, FakeLSTM.Input);
        TensorType initialH = context.CheckArgumentType<TensorType>(target, FakeLSTM.InitialH);
        TensorType initialC = context.CheckArgumentType<TensorType>(target, FakeLSTM.InitialC);
        context.CheckArgumentType<IRType>(target, FakeLSTM.Input);
        context.CheckArgumentType<IRType>(target, FakeLSTM.WXc);
        context.CheckArgumentType<IRType>(target, FakeLSTM.ActXc);
        context.CheckArgumentType<IRType>(target, FakeLSTM.WRc);
        context.CheckArgumentType<IRType>(target, FakeLSTM.ActRc);
        context.CheckArgumentType<IRType>(target, FakeLSTM.InitialH);
        context.CheckArgumentType<IRType>(target, FakeLSTM.InitialC);
        context.CheckArgumentType<IRType>(target, FakeLSTM.SegFittingParamFt);
        context.CheckArgumentType<IRType>(target, FakeLSTM.SegFittingParamGt);
        context.CheckArgumentType<IRType>(target, FakeLSTM.HasStatic);
        context.CheckArgumentType<IRType>(target, FakeLSTM.OutputSize);
        return Visit(context, target, input, initialH, initialC);
    }
}
