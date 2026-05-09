using System;
using System.Linq;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class GNNEConv2DTransposeEvaluator : IEvaluator<GNNEConv2DTranspose>, IEvaluator,
    ITypeInferencer<GNNEConv2DTranspose>, ITypeInferencer, ICostEvaluator<GNNEConv2DTranspose>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, GNNEConv2DTranspose target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    public IValue Visit(IEvaluateContext context, GNNEConv2DTranspose conv2DTranspose)
    {
        Tensor argumentValueAsTensor = context.GetArgumentValueAsTensor(conv2DTranspose, GNNEConv2DTranspose.Input);
        Tensor argumentValueAsTensor2 = context.GetArgumentValueAsTensor(conv2DTranspose, GNNEConv2DTranspose.Weights);
        byte[] weightsBias = context.GetArgumentValueAsArray<byte>(conv2DTranspose, GNNEConv2DTranspose.WeightsBias);
        Half[] argumentValueAsArray = context.GetArgumentValueAsArray<Half>(conv2DTranspose, GNNEConv2DTranspose.Act);
        byte deqBias = context.GetArgumentValueAsScalar<byte>(conv2DTranspose, GNNEConv2DTranspose.DeqBias);
        long argumentValueAsScalar =
            context.GetArgumentValueAsScalar<long>(conv2DTranspose, GNNEConv2DTranspose.ShiftBits);
        long[] array = context.GetArgumentValueAsArray<long>(conv2DTranspose, GNNEConv2DTranspose.Padding).ToArray();
        long[] argumentValueAsArray2 =
            context.GetArgumentValueAsArray<long>(conv2DTranspose, GNNEConv2DTranspose.Stride);
        long[] argumentValueAsArray3 =
            context.GetArgumentValueAsArray<long>(conv2DTranspose, GNNEConv2DTranspose.Dilation);
        long argumentValueAsScalar2 =
            context.GetArgumentValueAsScalar<long>(conv2DTranspose, GNNEConv2DTranspose.Groups);
        long[] array2 = context.GetArgumentValueAsArray<long>(conv2DTranspose, GNNEConv2DTranspose.OutputPadding)
            .ToArray();
        long[] argumentValueAsArray4 =
            context.GetArgumentValueAsArray<long>(conv2DTranspose, GNNEConv2DTranspose.OutputShape);
        int[] array3 = argumentValueAsTensor.Shape.ToValueArray();
        int[] array4 = argumentValueAsTensor2.Shape.ToValueArray();
        if (K230Kernels.GetWindowedOutputSize(
                (int)argumentValueAsArray4[2] + (int)array[0] + (int)array[1] - (int)array2[0], array4[2],
                (int)argumentValueAsArray2[0], (int)argumentValueAsArray3[0], same: false) != array3[2] ||
            K230Kernels.GetWindowedOutputSize(
                (int)argumentValueAsArray4[3] + (int)array[2] + (int)array[3] - (int)array2[1], array4[3],
                (int)argumentValueAsArray2[1], (int)argumentValueAsArray3[1], same: false) != array3[3])
        {
            throw new InvalidOleVariantTypeException("Invalid conv2d transpose shape");
        }

        float[] inputDeq = argumentValueAsTensor.ToArray<float>();
        inputDeq.Select((float _, int i) => inputDeq[i] -= (int)deqBias).AsParallel().ToArray();
        float[] weightsDeq = argumentValueAsTensor2.ToArray<float>();
        int qArgPerChannel = argumentValueAsTensor2.Dimensions[1] * argumentValueAsTensor2.Dimensions[2] *
                             argumentValueAsTensor2.Dimensions[3];
        weightsDeq.Select((float _, int i) => weightsDeq[i] -= (int)weightsBias[i / qArgPerChannel]).AsParallel()
            .ToArray();
        long num = argumentValueAsArray4[0] * argumentValueAsArray4[1] * argumentValueAsArray4[2] *
                   argumentValueAsArray4[3];
        float[] array5 = new float[num];
        Array.Clear(array5, 0, (int)num);
        long num2 = array3[1] / argumentValueAsScalar2;
        long num3 = argumentValueAsArray4[1] / argumentValueAsScalar2;
        int num4 = 0;
        for (int num5 = 0; num5 < array3[0]; num5++)
        {
            Span<float> span = array5.AsSpan();
            Span<float> span2 = span.Slice(num5 * (int)argumentValueAsArray4[1] * (int)argumentValueAsArray4[2] *
                                           (int)argumentValueAsArray4[3]);
            for (int num6 = 0; num6 < argumentValueAsScalar2; num6++)
            {
                Span<float> span3 =
                    span2.Slice(num6 * (int)num3 * (int)argumentValueAsArray4[2] * (int)argumentValueAsArray4[3]);
                span = weightsDeq.ToArray().AsSpan();
                Span<float> span4 = span.Slice(num6 * (int)num3 * (int)num2 * array4[2] * array4[3]);
                for (int num7 = 0; num7 < num2; num7++)
                {
                    for (int num8 = 0; num8 < array3[2]; num8++)
                    {
                        for (int num9 = 0; num9 < array3[3]; num9++)
                        {
                            int num10 = (int)(num8 * argumentValueAsArray2[0] - array[0]);
                            int num11 = (int)(num9 * argumentValueAsArray2[1] - array[2]);
                            int num12 = System.Math.Max(0,
                                (int)((-num10 + argumentValueAsArray3[0] - 1) / argumentValueAsArray3[0]));
                            int num13 = (int)System.Math.Min(array4[2],
                                ((int)argumentValueAsArray4[2] - num10 + argumentValueAsArray3[0] - 1) /
                                argumentValueAsArray3[0]);
                            int num14 = (int)System.Math.Max(0L,
                                (-num11 + argumentValueAsArray3[1] - 1) / argumentValueAsArray3[1]);
                            int num15 = (int)System.Math.Min(array4[3],
                                ((int)argumentValueAsArray4[3] - num11 + argumentValueAsArray3[1] - 1) /
                                argumentValueAsArray3[1]);
                            float num16 = ((num9 >= 0 && num9 < array3[3] && num8 >= 0 && num8 < array3[2])
                                ? inputDeq.ToArray()[num4]
                                : 0f);
                            num4++;
                            for (int num17 = 0; num17 < num3; num17++)
                            {
                                Span<float> span5 =
                                    span3.Slice((int)(num17 * argumentValueAsArray4[2] * argumentValueAsArray4[3]));
                                Span<float> span6 = span4.Slice((int)(num17 * num2 * array4[2] * array4[3]))
                                    .Slice(num7 * array4[2] * array4[3]);
                                for (int num18 = num12; num18 < num13; num18++)
                                {
                                    for (int num19 = num14; num19 < num15; num19++)
                                    {
                                        int num20 = (int)(num10 + argumentValueAsArray3[0] * num18);
                                        int num21 = (int)(num11 + argumentValueAsArray3[1] * num19);
                                        float num22 = span6[num18 * array4[3] + num19];
                                        span5[(int)(num20 * argumentValueAsArray4[3] + num21)] += num16 * num22;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor<float> tensor = Tensor.From(array5, (from i in argumentValueAsArray4.ToArray()
            select (int)i).ToArray());
        float[] array6 = tensor.ToArray<float>();
        float[] array7 = new float[K230Kernels.ComputeSize(tensor.Shape)];
        int num23 = tensor.Dimensions[2] * tensor.Dimensions[3];
        for (int num24 = 0; num24 < array6.Length; num24++)
        {
            int channel = num24 / num23;
            array7[num24] = K230Kernels.ApplyAct0(array6[num24], argumentValueAsArray.ToArray(), channel,
                (sbyte)argumentValueAsScalar);
        }

        float[] array8 = array7.Select((float x) => (float)System.Math.Round(x)).ToArray();
        Half[] array9 = array7.Select((float x) => (Half)x).ToArray();
        Tensor<float> tensor2 = Tensor.From(array8, tensor.Shape);
        Tensor<Half> tensor3 = Tensor.From(array9, tensor.Shape);
        if (conv2DTranspose.DestType == DataTypes.UInt8)
        {
            return Value.FromTensor(tensor2.Cast<byte>(CastMode.KDefault));
        }

        if (conv2DTranspose.DestType == DataTypes.Int8)
        {
            return Value.FromTensor(tensor2.Cast<sbyte>(CastMode.KDefault));
        }

        if (conv2DTranspose.DestType == DataTypes.Int16)
        {
            return Value.FromTensor(tensor2.Cast<short>(CastMode.KDefault));
        }

        return Value.FromTensor(tensor3.Cast<Half>(CastMode.KDefault));
    }

    private IRType Visit(ITypeInferenceContext context, GNNEConv2DTranspose target, TensorType input,
        TensorType weights)
    {
        if (input.DType != DataTypes.Int8 && input.DType != DataTypes.UInt8 && input.DType != DataTypes.Int16)
        {
            new InvalidType("Unsupported input_type, should be one of [int8, uint8, int16]");
        }

        if (weights.DType != DataTypes.Int8 && weights.DType != DataTypes.UInt8 && weights.DType != DataTypes.Int16)
        {
            new InvalidType("Unsupported w_type, should be one of [int8, uint8, int16]");
        }

        if (input.DType == DataTypes.Int16 && weights.DType == DataTypes.Int16)
        {
            new InvalidType("int16 for both of input_type and w_type is not supported");
        }

        if (context.GetArgument(target, GNNEConv2DTranspose.OutputShape) is Const obj)
        {
            return new TensorType(target.DestType, new Shape(Value.FromConst(obj).AsTensor().ToArray<int>()));
        }

        return new InvalidType("Conv2dTranspose can't infer shape with dynamic outputShape");
    }

    public IRType Visit(ITypeInferenceContext context, GNNEConv2DTranspose target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, GNNEConv2DTranspose.Input);
        TensorType weights = context.CheckArgumentType<TensorType>(target, GNNEConv2DTranspose.Weights);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Input);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Weights);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.WeightsBias);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.WeightsBiasQint8);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Act);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.ActQint8);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.DeqBias);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.ShiftBits);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.ShiftBitsQint8);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Qint8Qp);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Padding);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Stride);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Dilation);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Groups);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.Is16Quant);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.PadValue);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.WeightsQInt8);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.OutputPadding);
        context.CheckArgumentType<IRType>(target, GNNEConv2DTranspose.OutputShape);
        return Visit(context, target, input, weights);
    }
}
