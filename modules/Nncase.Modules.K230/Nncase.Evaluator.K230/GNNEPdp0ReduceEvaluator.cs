// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.TIR.Instructions;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class GNNEPdp0ReduceEvaluator : IEvaluator<GNNEPdp0Reduce>, IEvaluator, ITypeInferencer<GNNEPdp0Reduce>,
    ITypeInferencer, ICostEvaluator<GNNEPdp0Reduce>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, GNNEPdp0Reduce target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    public IValue Visit(IEvaluateContext context, GNNEPdp0Reduce p)
    {
        Tensor<Half> input = context.GetArgumentValueAsTensor<Half>(p, GNNEPdp0Reduce.Input);
        int[] filter = context.GetArgumentValueAsArray<int>(p, GNNEPdp0Reduce.Filter);
        int[] stride = context.GetArgumentValueAsArray<int>(p, GNNEPdp0Reduce.Stride);

        // Layout: [[heightBefore, heightAfter], [widthBefore, widthAfter]].
        Tensor<int> padding = context.GetArgumentValueAsTensor<int>(p, GNNEPdp0Reduce.Padding);
        bool[] countIncludePad = context.GetArgumentValueAsArray<bool>(p, GNNEPdp0Reduce.CountIncludePad);
        IValue dequantParams = context.GetArgumentValue(p, GNNEPdp0Reduce.DepuantParams);
        Tensor<Half> act = context.GetArgumentValueAsTensor<Half>(p, GNNEPdp0Reduce.Act);
        int shiftBits = context.GetArgumentValueAsScalar<int>(p, GNNEPdp0Reduce.ShiftBits);
        Tensor<float> padValue = context.GetArgumentValue(p, GNNEPdp0Reduce.Value).AsTensor().Cast<float>();
        Tensor<float> output = new Tensor<float>(context.CurrentCall.CheckedShape.ToValueArray());
        (Func<float, float, float> BinaryOp, Func<float, int, float> WindowOp) reduceFuncs = p.ReduceOp switch
        {
            PU_PDP0_MODE.average => ((float a, float b) => a + b, (float v, int k) => v / (float)k),
            PU_PDP0_MODE.min => ((float a, float b) => System.Math.Min(a, b), (float v, int _) => v),
            PU_PDP0_MODE.max => ((float a, float b) => System.Math.Max(a, b), (float v, int _) => v),
            PU_PDP0_MODE.sum => ((float a, float b) => a + b, (float v, int _) => v),
            _ => throw new ArgumentOutOfRangeException("context"),
        };
        Pdp0Impl(input, act.Buffer.Span, output, filter[0], filter[1], stride[0], stride[1],
            (Before: padding[new int[2]], After: padding[new int[2] { 0, 1 }]),
            (Before: padding[new int[2] { 1, 0 }], After: padding[new int[2] { 1, 1 }]),
            checked((sbyte)shiftBits), reduceFuncs.BinaryOp, reduceFuncs.WindowOp, countIncludePad[0],
            (dequantParams is NoneValue)
                ? new DeQuantizeParam(0, 1f)
                : dequantParams.AsTensor().ToScalar<DeQuantizeParam>(), padValue);
        return CastOutput(output, p.DestType);
    }

    public IRType Visit(ITypeInferenceContext context, GNNEPdp0Reduce target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, GNNEPdp0Reduce.Input);
        context.CheckArgumentType<IRType>(target, GNNEPdp0Reduce.Input);
        context.CheckArgumentType<IRType>(target, GNNEPdp0Reduce.Filter);
        context.CheckArgumentType<IRType>(target, GNNEPdp0Reduce.Stride);
        context.CheckArgumentType<IRType>(target, GNNEPdp0Reduce.Padding);
        context.CheckArgumentType<IRType>(target, GNNEPdp0Reduce.DepuantParams);
        context.CheckArgumentType<IRType>(target, GNNEPdp0Reduce.Value);
        context.CheckArgumentType<IRType>(target, GNNEPdp0Reduce.ShiftBits);
        context.CheckArgumentType<IRType>(target, GNNEPdp0Reduce.CountIncludePad);
        context.CheckArgumentType<IRType>(target, GNNEPdp0Reduce.Act);
        return Visit(context, target, input);
    }

    /// <summary>Converts the float result to the destination element type (anything unlisted becomes float16).</summary>
    private static IValue CastOutput(Tensor<float> output, PrimType destType)
    {
        if (destType == DataTypes.Int8)
        {
            return Value.FromTensor(output.Cast<sbyte>(CastMode.KDefault));
        }

        if (destType == DataTypes.UInt8)
        {
            return Value.FromTensor(output.Cast<byte>(CastMode.KDefault));
        }

        if (destType == DataTypes.Int16)
        {
            return Value.FromTensor(output.Cast<short>(CastMode.KDefault));
        }

        return Value.FromTensor(output.Cast<Half>(CastMode.KDefault));
    }

    private void Pdp0Impl(Tensor<Half> input, ReadOnlySpan<Half> act, Tensor<float> output, int filterH, int filterW,
        int strideH, int strideW, (int Before, int After) paddingH, (int Before, int After) paddingW, sbyte shiftBits,
        Func<float, float, float> binaryOp, Func<float, int, float> windowOp, bool countIncludePad,
        DeQuantizeParam deQuantizeParam, Tensor<float> padValue)
    {
        int[] inputShape = input.Shape.ToValueArray();
        int batch = inputShape[0];
        int channels = inputShape[1];
        int inputH = inputShape[2];
        int inputW = inputShape[3];
        int outputH = TypePatternUtility.GetWindowedOutputSize(inputH + paddingH.Before + paddingH.After, filterH,
            strideH, 1, same: false);
        int outputW = TypePatternUtility.GetWindowedOutputSize(inputW + paddingW.Before + paddingW.After, filterW,
            strideW, 1, same: false);
        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oy = 0; oy < outputH; oy++)
                {
                    for (int ox = 0; ox < outputW; ox++)
                    {
                        // Window origin in input coordinates (may be negative inside the padding).
                        int windowTop = oy * strideH - paddingH.Before;
                        int windowLeft = ox * strideW - paddingW.Before;

                        // Window range clipped to the real (unpadded) input.
                        int firstRow = System.Math.Max(0, -windowTop);
                        int endRow = System.Math.Min(filterH, inputH - windowTop);
                        int firstCol = System.Math.Max(0, -windowLeft);
                        int endCol = System.Math.Min(filterW, inputW - windowLeft);

                        // NOTE: the first valid element seeds the accumulator (not zero-point adjusted); there is
                        // no guard for windows that contain no valid element.
                        float accumulator =
                            (float)input[new int[4] { n, c, windowTop + firstRow, windowLeft + firstCol }];
                        int validCount = 0;
                        for (int row = firstRow; row < endRow; row++)
                        {
                            for (int col = firstCol; col < endCol; col++)
                            {
                                int inputY = windowTop + row;
                                int inputX = windowLeft + col;
                                float value = (float)input[new int[4] { n, c, inputY, inputX }];

                                // The first element is already in the accumulator.
                                if (row != firstRow || col != firstCol)
                                {
                                    // NOTE: the zero point is subtracted from the running accumulator at every
                                    // step, not only from the new value.
                                    accumulator = binaryOp(accumulator - (float)deQuantizeParam.ZeroPoint,
                                        value - (float)deQuantizeParam.ZeroPoint);
                                }

                                validCount++;
                            }
                        }

                        // Padded positions contribute the pad value and count towards the divisor.
                        if (countIncludePad)
                        {
                            for (int padIndex = 0; padIndex < filterH * filterW - validCount; padIndex++)
                            {
                                accumulator = binaryOp(accumulator, padValue.GetValue(0));
                            }

                            validCount = filterH * filterW;
                        }

                        // NOTE: the window size is round-tripped through float16 before being passed to windowOp.
                        Half windowSize = (Half)validCount;
                        float result = windowOp(accumulator * deQuantizeParam.Scale, (int)windowSize);
                        result = K230Kernels.ApplyAct0(result, act, c, shiftBits);
                        output[new int[4] { n, c, oy, ox }] = result;
                    }
                }
            }
        }
    }

    private IRType Visit(ITypeInferenceContext context, GNNEPdp0Reduce target, TensorType input)
    {
        Expr[] windowArgs =
            context.GetArguments(target, GNNEPdp0Reduce.Filter, GNNEPdp0Reduce.Stride, GNNEPdp0Reduce.Padding);
        IRType windowedType = TypeInference.ReduceWindow2DType(input, windowArgs[0], windowArgs[1], windowArgs[2],
            false);
        return new TensorType(target.DestType, ((TensorType)windowedType).Shape);
    }
}
