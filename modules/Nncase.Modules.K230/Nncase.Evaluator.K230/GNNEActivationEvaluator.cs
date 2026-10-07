// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class GNNEActivationEvaluator : IEvaluator<GNNEActivation>, IEvaluator, ITypeInferencer<GNNEActivation>,
    ITypeInferencer, ICostEvaluator<GNNEActivation>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, GNNEActivation target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    public IValue Visit(IEvaluateContext context, GNNEActivation a)
    {
        Tensor inputA = context.GetArgumentValueAsTensor(a, GNNEActivation.InputA);
        IValue inputB = context.GetArgumentValue(a, GNNEActivation.InputB);
        bool hasUninitializedInput = inputB is NoneValue;

        // Output dims: element-wise max of the A and B dims (B may be absent).
        int[] outputDims = inputA.Shape.ToValueArray();
        if (!hasUninitializedInput)
        {
            for (int i = 0; i < outputDims.Length; i++)
            {
                outputDims[i] = System.Math.Max(outputDims[i], inputB.AsTensor().Shape.ToValueArray()[i]);
            }
        }

        Tensor act = context.GetArgumentValueAsTensor(a, GNNEActivation.Act);
        bool[] is16Segments = context.GetArgumentValueAsTensor(a, GNNEActivation.Is16Segments).ToArray<bool>();
        int[] outChannels = context.GetArgumentValueAsTensor(a, GNNEActivation.OutChannels).ToArray<int>();
        IValue deqAParams = context.GetArgumentValue(a, GNNEActivation.DeqAParams);
        IValue deqBParams = context.GetArgumentValue(a, GNNEActivation.DeqBParams);
        return Value.FromConst(K230Kernels.GnneActivation(outputShape: context.CurrentCall.CheckedShape,
            is16Segments: is16Segments[0], inputA: inputA,
            inputB: hasUninitializedInput ? new Tensor<Half>(0) : inputB.AsTensor(),
            hasUninitializedInput: hasUninitializedInput, actData: act.ToArray<float>(), outChannels: outChannels[0],
            deQuantizeParamA: (deqAParams is NoneValue)
                ? new DeQuantizeParam(0, 1f)
                : ((DeQuantizeParam)deqAParams.AsTensor()[new int[1]]),
            deQuantizeParamB: (deqBParams is NoneValue)
                ? new DeQuantizeParam(0, 1f)
                : ((DeQuantizeParam)deqBParams.AsTensor()[new int[1]]), actType: a.Type,
            outputType: new TensorType(a.OutputDType, outputDims)));
    }

    public IRType Visit(ITypeInferenceContext context, GNNEActivation target)
    {
        TensorType inputA = context.CheckArgumentType<TensorType>(target, GNNEActivation.InputA);
        IRType inputB = context.CheckArgumentType<IRType>(target, GNNEActivation.InputB);
        context.CheckArgumentType<IRType>(target, GNNEActivation.InputA);
        context.CheckArgumentType<IRType>(target, GNNEActivation.InputB);
        context.CheckArgumentType<IRType>(target, GNNEActivation.Act);
        context.CheckArgumentType<IRType>(target, GNNEActivation.InAShiftBits);
        context.CheckArgumentType<IRType>(target, GNNEActivation.InBShiftBits);
        context.CheckArgumentType<IRType>(target, GNNEActivation.OutShiftBits);
        context.CheckArgumentType<IRType>(target, GNNEActivation.DeqAParams);
        context.CheckArgumentType<IRType>(target, GNNEActivation.DeqBParams);
        context.CheckArgumentType<IRType>(target, GNNEActivation.OutChannels);
        context.CheckArgumentType<IRType>(target, GNNEActivation.Is16Segments);
        return Visit(target, inputA, inputB);
    }

    private IRType Visit(GNNEActivation target, TensorType inputA, IRType inputB)
    {
        TensorType result = new TensorType(target.OutputDType, target.OutputShape.ToArray());
        if (!(inputB is NoneType))
        {
            // Output dims: element-wise max of the A and B dims.
            int[] outputDims = inputA.Shape.ToValueArray();
            TensorType inputBType = (TensorType)inputB;
            for (int i = 0; i < outputDims.Length; i++)
            {
                outputDims[i] = System.Math.Max(outputDims[i], inputBType.Shape.ToValueArray()[i]);
            }

            result = new TensorType(target.OutputDType, outputDims);
        }

        bool inputASupported = inputA.DType == DataTypes.UInt8 || inputA.DType == DataTypes.Int8 ||
                               inputA.DType == DataTypes.Float16 || inputA.DType == DataTypes.Int16;
        if (inputB is NoneType)
        {
            if (inputASupported)
            {
                return result;
            }

            return new InvalidType("No support the act's datatype");
        }

        // NOTE: input B does not accept int16, unlike input A.
        TensorType inputBTensorType = inputB as TensorType;
        if (inputASupported && (inputBTensorType.DType == DataTypes.UInt8 || inputBTensorType.DType == DataTypes.Int8 ||
                                inputBTensorType.DType == DataTypes.Float16))
        {
            return result;
        }

        return new InvalidType("No support the act's datatype");
    }
}
