// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class GNNELSTMEvaluator : IEvaluator<GNNELSTM>, IEvaluator, ITypeInferencer<GNNELSTM>, ITypeInferencer,
    ICostEvaluator<GNNELSTM>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, GNNELSTM target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    public IValue Visit(IEvaluateContext context, GNNELSTM l)
    {
        Tensor<float> input = context.GetArgumentValueAsTensor<float>(l, GNNELSTM.Input);
        Tensor<float> inputWeights = context.GetArgumentValueAsTensor<float>(l, GNNELSTM.WXc);
        Tensor<Half> inputActivation = context.GetArgumentValueAsTensor<Half>(l, GNNELSTM.ActXc);
        Tensor<float> recurrentWeights = context.GetArgumentValueAsTensor<float>(l, GNNELSTM.WRc);
        Tensor<Half> recurrentActivationFirstStep = context.GetArgumentValueAsTensor<Half>(l, GNNELSTM.ActRc0);
        Tensor<Half> recurrentActivationOtherSteps = context.GetArgumentValueAsTensor<Half>(l, GNNELSTM.ActRc1);
        Tensor<float> initialHidden = context.GetArgumentValueAsTensor<float>(l, GNNELSTM.InitialH);
        Tensor<float> initialCell = context.GetArgumentValueAsTensor<float>(l, GNNELSTM.InitialC);
        Tensor<Half> sigmoidFit = context.GetArgumentValueAsTensor<Half>(l, GNNELSTM.SegFittingParamFt);
        Tensor<Half> tanhFit = context.GetArgumentValueAsTensor<Half>(l, GNNELSTM.SegFittingParamGt);
        Tensor<Half> inputWeightsQuantArgs = context.GetArgumentValueAsTensor<Half>(l, GNNELSTM.WXcQarg);
        Tensor<Half> recurrentWeightsQuantArgs = context.GetArgumentValueAsTensor<Half>(l, GNNELSTM.WRcQarg);
        Tensor<Half> cellBinAct = context.GetArgumentValueAsTensor<Half>(l, GNNELSTM.ActBin);
        Tensor<Half> hiddenBinQuantAct = context.GetArgumentValueAsTensor<Half>(l, GNNELSTM.ActBinQ);
        Tensor<int> inputDeqBias = context.GetArgumentValueAsTensor<int>(l, GNNELSTM.IfDeqBias);
        Tensor<int> inputShiftBits = context.GetArgumentValueAsTensor<int>(l, GNNELSTM.XcShiftBits);
        Tensor<int> hiddenDeqBias0 = context.GetArgumentValueAsTensor<int>(l, GNNELSTM.HDeqBias0);
        Tensor<int> hiddenDeqBias1 = context.GetArgumentValueAsTensor<int>(l, GNNELSTM.HDeqBias1);
        Tensor<int> recurrentShiftBitsFirstStep = context.GetArgumentValueAsTensor<int>(l, GNNELSTM.RcShiftBits0);
        Tensor<int> recurrentShiftBitsOtherSteps = context.GetArgumentValueAsTensor<int>(l, GNNELSTM.RcShiftBits1);
        Tensor<int> outputSizeTensor = context.GetArgumentValueAsTensor<int>(l, GNNELSTM.OutputSize);

        // Output buffers (Y, last hidden, last cell) are shaped after the call's tuple type.
        Tensor outputC = new Tensor<Half>(GetOutputShape(context, 0));
        if (outputSizeTensor.ToArray()[0] >= 2)
        {
            outputC = new Tensor<Half>(GetOutputShape(context, 1));
        }

        // NOTE: the original also had an 'else if (outputSize >= 3)' branch that could never run, so the cell
        // buffer is never shaped after the third tuple element.
        Tensor output = CreateOutputBuffer(context, l.DestTypeO, outputSizeTensor);
        Tensor outputH = CreateOutputBuffer(context, l.DestTypeOH, outputSizeTensor);

        return Value.FromTensors(K230Kernels.GnneLstmImpl(l.DestTypeO, l.DestTypeOH, input,
            inputWeights, inputActivation, recurrentWeights, recurrentActivationFirstStep,
            recurrentActivationOtherSteps, initialHidden, initialCell, sigmoidFit,
            tanhFit, output, outputH, outputC, l.Direction, inputWeightsQuantArgs,
            recurrentWeightsQuantArgs, cellBinAct, hiddenBinQuantAct,
            inputDeqBias[new int[1]], hiddenDeqBias0[new int[1]],
            hiddenDeqBias1[new int[1]], inputShiftBits[new int[1]],
            recurrentShiftBitsFirstStep[new int[1]], recurrentShiftBitsOtherSteps[new int[1]],
            outputSizeTensor.ToArray()[0]).ToArray());
    }

    /// <summary>Shape of the <paramref name="index"/>-th element of the call's output tuple type.</summary>
    private static int[] GetOutputShape(IEvaluateContext context, int index)
    {
        return ((TensorType)((TupleType)context.CurrentCall.CheckedType)[index]).Shape.ToValueArray();
    }

    /// <summary>
    /// Creates the (zero-filled) buffer the kernel writes the Y / hidden output into: element type
    /// <paramref name="destType"/> when at least one output is requested.
    /// </summary>
    private static Tensor CreateOutputBuffer(IEvaluateContext context, DataType destType, Tensor<int> outputSizeTensor)
    {
        // Default (also used when no output is requested).
        Tensor buffer = new Tensor<byte>(GetOutputShape(context, 0));

        // NOTE: the original also had 'else if (outputSize >= 2/3)' branches that could never run, so the
        // buffer is always shaped after the first tuple element.
        if (destType == DataTypes.UInt8)
        {
            if (outputSizeTensor.ToArray()[0] >= 1)
            {
                buffer = new Tensor<byte>(GetOutputShape(context, 0));
            }
        }
        else if (destType == DataTypes.Int8)
        {
            if (outputSizeTensor.ToArray()[0] >= 1)
            {
                buffer = new Tensor<sbyte>(GetOutputShape(context, 0));
            }
        }
        else if (destType == DataTypes.Int16)
        {
            if (outputSizeTensor.ToArray()[0] >= 1)
            {
                buffer = new Tensor<short>(GetOutputShape(context, 0));
            }
        }
        else if (outputSizeTensor.ToArray()[0] >= 1)
        {
            buffer = new Tensor<Half>(GetOutputShape(context, 0));
        }

        return buffer;
    }

    private IRType Visit(ITypeInferenceContext context, GNNELSTM target, TensorType input, TensorType initialH,
        TensorType initialC)
    {
        int numDirections = (target.Direction != LSTMDirection.Bidirectional) ? 1 : 2;
        int seqLenIndex = 1;
        if (context.GetArgument(target, GNNELSTM.OutputSize) is TensorConst outputSizeConst)
        {
            // NOTE: the result is discarded (only its argument checks can have an effect); the Y type below is
            // built by hand instead.
            InferYType(context, target, input, seqLenIndex, numDirections);
            Shape outputShape = new Dimension[4] { input.Shape[1], initialH.Shape[1], initialH.Shape[2], initialH.Shape[3] };

            // Outputs: Y, last hidden, last cell (cell is always float16); keep the first OutputSize of them.
            IRType[] outputTypes = (new TensorType[3]
            {
                new TensorType(target.DestTypeO, outputShape),
                new TensorType(target.DestTypeOH,
                    new Dimension[4] { 1, initialH.Shape[1], initialH.Shape[2], initialH.Shape[3] }),
                new TensorType(DataTypes.Float16,
                    new Dimension[4] { 1, initialC.Shape[1], initialC.Shape[2], initialC.Shape[3] })
            })[..outputSizeConst.Value.ToScalar<int>()];
            return new TupleType(outputTypes);
        }

        return new InvalidType("LSTM OutputSize Must be known");
    }

    /// <summary>
    /// Y type: the input shape with the sequence dimension split off and the direction count inserted; the last
    /// dimension becomes the hidden size of the recurrent weights.
    /// </summary>
    private TensorType InferYType(ITypeInferenceContext context, GNNELSTM target, TensorType x, int seqLenIndex,
        int numDirections)
    {
        List<Dimension> dims = x.Shape.ToList();
        dims.Insert(seqLenIndex + 1, numDirections);
        int hiddenSize = context.GetArgument(target, GNNELSTM.WRc).CheckedShape[3].FixedValue;
        dims.RemoveAt(0);
        dims[dims.Count - 1] = hiddenSize;
        return x with { Shape = dims.ToArray() };
    }

    public IRType Visit(ITypeInferenceContext context, GNNELSTM target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, GNNELSTM.Input);
        TensorType initialH = context.CheckArgumentType<TensorType>(target, GNNELSTM.InitialH);
        TensorType initialC = context.CheckArgumentType<TensorType>(target, GNNELSTM.InitialC);
        context.CheckArgumentType<IRType>(target, GNNELSTM.Input);
        context.CheckArgumentType<IRType>(target, GNNELSTM.WXc);
        context.CheckArgumentType<IRType>(target, GNNELSTM.ActXc);
        context.CheckArgumentType<IRType>(target, GNNELSTM.WRc);
        context.CheckArgumentType<IRType>(target, GNNELSTM.ActRc0);
        context.CheckArgumentType<IRType>(target, GNNELSTM.ActRc1);
        context.CheckArgumentType<IRType>(target, GNNELSTM.InitialH);
        context.CheckArgumentType<IRType>(target, GNNELSTM.InitialC);
        context.CheckArgumentType<IRType>(target, GNNELSTM.SegFittingParamFt);
        context.CheckArgumentType<IRType>(target, GNNELSTM.SegFittingParamGt);
        context.CheckArgumentType<IRType>(target, GNNELSTM.WXcQarg);
        context.CheckArgumentType<IRType>(target, GNNELSTM.WRcQarg);
        context.CheckArgumentType<IRType>(target, GNNELSTM.ActBin);
        context.CheckArgumentType<IRType>(target, GNNELSTM.ActBinQ);
        context.CheckArgumentType<IRType>(target, GNNELSTM.IfDeqBias);
        context.CheckArgumentType<IRType>(target, GNNELSTM.XcShiftBits);
        context.CheckArgumentType<IRType>(target, GNNELSTM.HDeqBias0);
        context.CheckArgumentType<IRType>(target, GNNELSTM.HDeqBias1);
        context.CheckArgumentType<IRType>(target, GNNELSTM.CShiftBits);
        context.CheckArgumentType<IRType>(target, GNNELSTM.RcShiftBits0);
        context.CheckArgumentType<IRType>(target, GNNELSTM.RcShiftBits1);
        context.CheckArgumentType<IRType>(target, GNNELSTM.OutHShiftBits);
        context.CheckArgumentType<IRType>(target, GNNELSTM.OutCShiftBits);
        context.CheckArgumentType<IRType>(target, GNNELSTM.HasStatic);
        context.CheckArgumentType<IRType>(target, GNNELSTM.OutputSize);
        return Visit(context, target, input, initialH, initialC);
    }
}
