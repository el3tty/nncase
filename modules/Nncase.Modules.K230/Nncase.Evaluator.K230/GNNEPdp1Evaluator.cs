// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.TIR.Instructions;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class GNNEPdp1Evaluator : IEvaluator<GNNEPdp1>, IEvaluator, ITypeInferencer<GNNEPdp1>, ITypeInferencer,
    ICostEvaluator<GNNEPdp1>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, GNNEPdp1 target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    public IValue Visit(IEvaluateContext context, GNNEPdp1 r)
    {
        Tensor input = context.GetArgumentValueAsTensor(r, GNNEPdp1.Input);
        int[] filter = context.GetArgumentValueAsArray<int>(r, GNNEPdp1.Filter);
        int[] stride = context.GetArgumentValueAsArray<int>(r, GNNEPdp1.Stride);

        // Layout: [[heightBefore, heightAfter], [widthBefore, widthAfter]].
        Tensor<int> padding = context.GetArgumentValueAsTensor<int>(r, GNNEPdp1.Padding);
        bool[] countIncludePad = context.GetArgumentValueAsArray<bool>(r, GNNEPdp1.CountIncludePad);
        IValue quantParams = context.GetArgumentValue(r, GNNEPdp1.QuantParams);
        IValue dequantParams = context.GetArgumentValue(r, GNNEPdp1.DequantParams);
        Tensor<Half> padValue = context.GetArgumentValue(r, GNNEPdp1.Value).AsTensor().Cast<Half>();
        Tensor<float> output = new Tensor<float>(context.CurrentCall.CheckedShape.ToValueArray());

        // Only MAX / MIN / AVERAGE / SUM exist in the enum; any other value is unsupported.
        (Func<float, float, float> BinaryOp, Func<float, int, float> WindowOp) reduceFuncs = r.ReduceOp switch
        {
            MFU_PDP_OP.AVERAGE => ((float a, float b) => a + b, (float v, int k) => v / (float)k),
            MFU_PDP_OP.MIN => ((float a, float b) => System.Math.Min(a, b), (float v, int _) => v),
            MFU_PDP_OP.MAX => ((float a, float b) => System.Math.Max(a, b), (float v, int _) => v),
            MFU_PDP_OP.SUM => ((float a, float b) => a + b, (float v, int _) => v),
            _ => throw new NotSupportedException(),
        };

        Pdp1Impl(input, output, r.DestType, filter[0], filter[1], stride[0], stride[1],
            (Before: padding[new int[2]], After: padding[new int[2] { 0, 1 }]),
            (Before: padding[new int[2] { 1, 0 }], After: padding[new int[2] { 1, 1 }]), reduceFuncs.BinaryOp,
            reduceFuncs.WindowOp, countIncludePad[0], (quantParams is NoneValue) ? null : quantParams.AsTensor(),
            (dequantParams is NoneValue) ? null : dequantParams.AsTensor(), padValue);
        return CastOutput(output, r.DestType);
    }

    public IRType Visit(ITypeInferenceContext context, GNNEPdp1 target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, GNNEPdp1.Input);
        context.CheckArgumentType<IRType>(target, GNNEPdp1.Input);
        context.CheckArgumentType<IRType>(target, GNNEPdp1.Filter);
        context.CheckArgumentType<IRType>(target, GNNEPdp1.Stride);
        context.CheckArgumentType<IRType>(target, GNNEPdp1.Padding);
        context.CheckArgumentType<IRType>(target, GNNEPdp1.QuantParams);
        context.CheckArgumentType<IRType>(target, GNNEPdp1.DequantParams);
        context.CheckArgumentType<IRType>(target, GNNEPdp1.Value);
        context.CheckArgumentType<IRType>(target, GNNEPdp1.ShiftBits);
        context.CheckArgumentType<IRType>(target, GNNEPdp1.CountIncludePad);
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

    /// <summary>Applies the input dequantization (value - zeroPoint) * scale using the first dequant param.</summary>
    private static float Dequantize(Tensor deQuantizeParam, float value)
    {
        DeQuantizeParam param = deQuantizeParam.ToArray<DeQuantizeParam>()[0];
        return (value - (float)param.ZeroPoint) * param.Scale;
    }

    private void Pdp1Impl(Tensor input, Tensor output, DataType destType, int filterH, int filterW, int strideH,
        int strideW, (int Before, int After) paddingH, (int Before, int After) paddingW,
        Func<float, float, float> binaryOp, Func<float, int, float> windowOp, bool countIncludePad,
        Tensor quantizeParam = null, Tensor deQuantizeParam = null, Tensor<Half> padValue = null)
    {
        int[] inputShape = input.Shape.ToValueArray();
        int batch = inputShape[0];
        int channels = inputShape[1];
        int inputH = inputShape[2];
        int inputW = inputShape[3];
        DataType elementType = input.ElementType;
        Tensor floatInput = input.Cast<float>();

        // The dequantization only applies to non-float16 inputs.
        bool applyDequant = elementType != DataTypes.Float16 && deQuantizeParam != null;
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

                        // NOTE: the first valid element seeds the accumulator; there is no guard for windows
                        // that contain no valid element.
                        float accumulator =
                            (float)floatInput[new int[4] { n, c, windowTop + firstRow, windowLeft + firstCol }];
                        if (applyDequant)
                        {
                            accumulator = Dequantize(deQuantizeParam, accumulator);
                        }

                        int validCount = 0;
                        for (int row = firstRow; row < endRow; row++)
                        {
                            for (int col = firstCol; col < endCol; col++)
                            {
                                int inputY = windowTop + row;
                                int inputX = windowLeft + col;
                                float value = (float)floatInput[new int[4] { n, c, inputY, inputX }];
                                if (applyDequant)
                                {
                                    value = Dequantize(deQuantizeParam, value);
                                }

                                // The first element is already in the accumulator.
                                if (row != firstRow || col != firstCol)
                                {
                                    accumulator = binaryOp(accumulator, value);
                                }

                                validCount++;
                            }
                        }

                        // Padded positions contribute the pad value and count towards the divisor.
                        if (countIncludePad)
                        {
                            for (int padIndex = 0; padIndex < filterH * filterW - validCount; padIndex++)
                            {
                                accumulator = binaryOp(accumulator, (float)padValue.GetValue(0));
                            }

                            validCount = filterH * filterW;
                        }

                        float result = windowOp(accumulator, validCount);

                        // NOTE: destination types other than float16 / int8 / int16 / uint8 leave the output
                        // untouched; the clamp lower bound of the signed types excludes the minimum value.
                        if (destType == DataTypes.Float16)
                        {
                            output[new int[4] { n, c, oy, ox }] = result;
                        }
                        else if (destType == DataTypes.Int8)
                        {
                            QuantizeParam quantParam = quantizeParam.ToArray<QuantizeParam>()[0];
                            output[new int[4] { n, c, oy, ox }] = System.Math.Clamp(
                                (sbyte)System.Math.Round(result * quantParam.Scale + (float)quantParam.ZeroPoint),
                                (sbyte)-127, sbyte.MaxValue);
                        }
                        else if (destType == DataTypes.Int16)
                        {
                            QuantizeParam quantParam = quantizeParam.ToArray<QuantizeParam>()[0];
                            output[new int[4] { n, c, oy, ox }] = System.Math.Clamp(
                                (short)System.Math.Round(result * quantParam.Scale + (float)quantParam.ZeroPoint),
                                (short)(-32767), short.MaxValue);
                        }
                        else if (destType == DataTypes.UInt8)
                        {
                            QuantizeParam quantParam = quantizeParam.ToArray<QuantizeParam>()[0];
                            output[new int[4] { n, c, oy, ox }] = System.Math.Clamp(
                                (byte)System.Math.Round(result * quantParam.Scale + (float)quantParam.ZeroPoint),
                                (byte)0, byte.MaxValue);
                        }
                    }
                }
            }
        }
    }

    private IRType Visit(ITypeInferenceContext context, GNNEPdp1 target, TensorType input)
    {
        Expr[] windowArgs = context.GetArguments(target, GNNEPdp1.Filter, GNNEPdp1.Stride, GNNEPdp1.Padding);
        IRType windowedType = TypeInference.ReduceWindow2DType(input, windowArgs[0], windowArgs[1], windowArgs[2],
            false);
        return new TensorType(target.DestType, ((TensorType)windowedType).Shape);
    }
}
