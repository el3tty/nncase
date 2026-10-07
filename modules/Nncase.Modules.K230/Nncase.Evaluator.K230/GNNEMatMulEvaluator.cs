using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using OrtKISharp;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public sealed class GNNEMatMulEvaluator : IEvaluator<GNNEMatMul>, IEvaluator, ITypeInferencer<GNNEMatMul>,
    ITypeInferencer, ICostEvaluator<GNNEMatMul>, ICostEvaluator
{
    public IValue Visit(IEvaluateContext context, GNNEMatMul matMul)
    {
        Tensor inputA = context.GetArgumentValueAsTensor(matMul, GNNEMatMul.InputA);
        Tensor inputB = context.GetArgumentValueAsTensor(matMul, GNNEMatMul.InputB);
        byte[] inputABias = context.GetArgumentValueAsTensor<byte>(matMul, GNNEMatMul.InputABias).ToArray();
        byte[] deqBBias = context.GetArgumentValueAsTensor<byte>(matMul, GNNEMatMul.DeqBBias).ToArray();
        Tensor<Half> act = context.GetArgumentValueAsTensor<Half>(matMul, GNNEMatMul.Act);
        Tensor shiftBits = context.GetArgumentValueAsTensor(matMul, GNNEMatMul.ShiftBits);

        // Both inputs are 4-D: [batch, channels, rows, cols]; dimension 1 is the (broadcastable) channel count.
        int aChannels = inputA.Shape[1].FixedValue;
        int bChannels = inputB.Shape[1].FixedValue;
        int aRows = inputA.Shape[2].FixedValue;
        int aCols = inputA.Shape[3].FixedValue;
        int bCols = inputB.Shape[3].FixedValue;

        // Output type: input A's shape with the last dimension replaced by input B's last dimension.
        int[] outputShape = inputA.Shape.ToValueArray();
        int lastDimIndex = outputShape.Length - 1;
        Shape inputBShape = inputB.Shape;
        outputShape[lastDimIndex] = inputBShape[inputBShape.Count - 1].FixedValue;
        TensorType outputType = new TensorType(matMul.OutputDType, outputShape);

        // Zero-point corrected (dequantized) copies of the inputs.
        float[] inputADeq = inputA.ToArray<float>();
        float[] inputBDeq = inputB.ToArray<float>();
        if (aChannels == bChannels || (aChannels < bChannels && aChannels == 1))
        {
            for (int i = 0; i < inputADeq.Length; i++)
            {
                inputADeq[i] -= (int)inputABias[i * aRows / (aRows * aCols * bCols)];
            }

            for (int i = 0; i < inputBDeq.Length; i++)
            {
                inputBDeq[i] -= (int)deqBBias[i / (aRows * aCols * bCols)];
            }
        }
        else
        {
            if (aChannels <= bChannels || bChannels != 1)
            {
                // NOTE: an unrelated exception type is thrown for incompatible channel counts (kept as is).
                throw new InvalidOleVariantTypeException("Invalid matmul");
            }

            // Input B is broadcast over channels: a single bias applies to all of it.
            for (int i = 0; i < inputADeq.Length; i++)
            {
                inputADeq[i] -= (int)inputABias[i / aRows];
            }

            for (int i = 0; i < inputBDeq.Length; i++)
            {
                inputBDeq[i] -= (int)deqBBias[0];
            }
        }

        Tensor matMulResult = OrtKI
            .MatMul(
                OrtKISharp.Tensor.MakeTensor(inputADeq,
                    ((IEnumerable<int>)inputA.Dimensions.ToArray())
                    .Select((Func<int, long>)((int i) => i)).ToArray()),
                OrtKISharp.Tensor.MakeTensor(inputBDeq,
                    ((IEnumerable<int>)inputB.Dimensions.ToArray())
                    .Select((Func<int, long>)((int i) => i)).ToArray())).ToTensor();
        float[] matMulData = matMulResult.ToArray<float>();
        float[] activated = new float[K230Kernels.ComputeSize(matMulResult.Shape)];
        if (matMulData.Length > 0)
        {
            // Hoisted out of the per-element loop; only evaluated when there is at least one element,
            // like the per-element calls they replace.
            Half[] actData = act.ToArray<Half>();
            sbyte shiftBitsValue = shiftBits.ToArray<sbyte>()[0];
            for (int k = 0; k < matMulData.Length; k++)
            {
                // The activation channel is the output row (index / cols).
                activated[k] = K230Kernels.ApplyAct0(matMulData[k], actData, k / matMulResult.Shape[3].FixedValue,
                    shiftBitsValue);
            }
        }

        float[] rounded = activated.Select((float x) => (float)System.Math.Round(x)).ToArray();
        Half[] activatedHalf = activated.Select((float x) => (Half)x).ToArray();
        Tensor<float> roundedTensor = Tensor.From(rounded, matMulResult.Shape);
        Tensor<Half> halfTensor = Tensor.From(activatedHalf, matMulResult.Shape);

        // NOTE: this compares a TensorType with a scalar DataType (via the implicit DataType -> IRType conversion),
        // so it is never true for a non-scalar output and the Half result below is always returned.
        if (outputType == DataTypes.UInt8)
        {
            return Value.FromTensor(roundedTensor.Cast<byte>(CastMode.KDefault));
        }

        if (outputType == DataTypes.Int8)
        {
            return Value.FromTensor(roundedTensor.Cast<sbyte>(CastMode.KDefault));
        }

        if (outputType == DataTypes.Int16)
        {
            return Value.FromTensor(roundedTensor.Cast<short>(CastMode.KDefault));
        }

        return Value.FromTensor(halfTensor.Cast<Half>(CastMode.KDefault));
    }

    public Cost Visit(ICostEvaluateContext context, GNNEMatMul target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    private IRType Visit(TensorType inputA, TensorType inputB, GNNEMatMul target)
    {
        DataType inputADType = inputA.DType;
        DataType inputBDType = inputB.DType;
        if (inputADType != DataTypes.Int8 && inputADType != DataTypes.UInt8 && inputADType != DataTypes.Int16)
        {
            return new InvalidType("Unsupported input_a_type, should be one of [int8, uint8, int16]");
        }

        if (inputBDType != DataTypes.Int8 && inputBDType != DataTypes.UInt8 && inputBDType != DataTypes.Int16)
        {
            return new InvalidType("Unsupported input_b_type, should be one of [int8, uint8, int16]");
        }

        if (inputADType == DataTypes.Int16 && inputBDType == DataTypes.Int16)
        {
            return new InvalidType("int16 for both of input_a_type and input_b_type is not supported");
        }

        if (target.OutputDType != DataTypes.Float16 && target.OutputDType != DataTypes.Float32)
        {
            return new InvalidType("Invalid Ouput Datatype");
        }

        // Trailing [rows of A, cols of B]; the leading dimensions are the element-wise maximum of both shapes.
        Dimension[] matrixDims = new Dimension[2];
        Shape inputAShape = inputA.Shape;
        matrixDims[0] = inputAShape[inputAShape.Count - 2];
        Shape inputBShape = inputB.Shape;
        matrixDims[1] = inputBShape[inputBShape.Count - 1];
        Shape maxShape = new Shape(from t in inputA.Shape.Zip(inputB.Shape)
            select System.Math.Max(t.First.FixedValue, t.Second.FixedValue));
        return new TensorType(target.OutputDType,
            maxShape.ToArray()[..(maxShape.Count - 2)].Concat(matrixDims).ToArray());
    }

    public IRType Visit(ITypeInferenceContext context, GNNEMatMul target)
    {
        TensorType inputA = context.CheckArgumentType<TensorType>(target, GNNEMatMul.InputA);
        TensorType inputB = context.CheckArgumentType<TensorType>(target, GNNEMatMul.InputB);
        context.CheckArgumentType<IRType>(target, GNNEMatMul.InputA);
        context.CheckArgumentType<IRType>(target, GNNEMatMul.InputB);
        context.CheckArgumentType<IRType>(target, GNNEMatMul.Act);
        context.CheckArgumentType<IRType>(target, GNNEMatMul.InputABias);
        context.CheckArgumentType<IRType>(target, GNNEMatMul.InAShiftBits);
        context.CheckArgumentType<IRType>(target, GNNEMatMul.InBShiftBits);
        context.CheckArgumentType<IRType>(target, GNNEMatMul.ShiftBits);
        context.CheckArgumentType<IRType>(target, GNNEMatMul.DeqBBias);
        return Visit(inputA, inputB, target);
    }
}
