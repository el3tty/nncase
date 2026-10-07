using System;
using System.Linq;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;

namespace Nncase.Evaluator.K230;

[EvaluatorGenerator]
[TypeInferGenerator]
public sealed class DynamicGNNEMatMulEvaluator : IEvaluator<DynamicGNNEMatMul>, IEvaluator,
    ITypeInferencer<DynamicGNNEMatMul>, ITypeInferencer, ICostEvaluator<DynamicGNNEMatMul>, ICostEvaluator
{
    /// <summary>Divisor applied to the reduction size to get the CPU cycle factor.</summary>
    private const float CpuCyclesDivisor = 768f;

    public Cost Visit(ICostEvaluateContext context, DynamicGNNEMatMul target)
    {
        TensorType inputAType = context.GetArgumentType<TensorType>(target, DynamicGNNEMatMul.InputA);
        TensorType inputBType = context.GetArgumentType<TensorType>(target, DynamicGNNEMatMul.InputB);
        TensorType actType = context.GetArgumentType<TensorType>(target, DynamicGNNEMatMul.Act);
        TensorType inputABiasType = context.GetArgumentType<TensorType>(target, DynamicGNNEMatMul.InputABias);
        TensorType inputBBiasType = context.GetArgumentType<TensorType>(target, DynamicGNNEMatMul.InputBBias);
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
                                           CostUtility.GetMemoryAccess(actType) +
                                           CostUtility.GetMemoryAccess(inputABiasType) +
                                           CostUtility.GetMemoryAccess(inputBBiasType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(returnType, (float)aCols / CpuCyclesDivisor),
        };
    }

    /// <summary>
    /// Runs the dynamic matmul kernel for the element types of the inputs; the output element type is chosen from
    /// <paramref name="outputType"/> (uint8, int8, int16, float16 or float32).
    /// </summary>
    private static Tensor GnneMatmulV2<TIA, TIB>(ReadOnlySpan<TIA> inputA, ReadOnlySpan<TIB> inputB,
        TensorType outputType, ReadOnlySpan<Half> act, ReadOnlySpan<byte> inABias, int aBatch0, int aBatch1, int aRows,
        int aCols, int bBatch0, int bBatch1, int bCols, byte deqBBias, int inAShiftBits, int inBShiftBits,
        sbyte shiftBits, bool dynamicChannel) where TIA : unmanaged where TIB : unmanaged
    {
        DataType outputDType = outputType.DType;
        if (outputDType == DataTypes.UInt8)
        {
            return RunKernel<TIA, TIB, byte>(inputA, inputB, outputType, act, inABias, aBatch0, aBatch1, aRows, aCols,
                bBatch0, bBatch1, bCols, deqBBias, inAShiftBits, inBShiftBits, shiftBits, dynamicChannel);
        }

        if (outputDType == DataTypes.Int8)
        {
            return RunKernel<TIA, TIB, sbyte>(inputA, inputB, outputType, act, inABias, aBatch0, aBatch1, aRows, aCols,
                bBatch0, bBatch1, bCols, deqBBias, inAShiftBits, inBShiftBits, shiftBits, dynamicChannel);
        }

        if (outputDType == DataTypes.Int16)
        {
            return RunKernel<TIA, TIB, short>(inputA, inputB, outputType, act, inABias, aBatch0, aBatch1, aRows, aCols,
                bBatch0, bBatch1, bCols, deqBBias, inAShiftBits, inBShiftBits, shiftBits, dynamicChannel);
        }

        if (outputDType == DataTypes.Float16)
        {
            return RunKernel<TIA, TIB, Half>(inputA, inputB, outputType, act, inABias, aBatch0, aBatch1, aRows, aCols,
                bBatch0, bBatch1, bCols, deqBBias, inAShiftBits, inBShiftBits, shiftBits, dynamicChannel);
        }

        if (outputDType == DataTypes.Float32)
        {
            return RunKernel<TIA, TIB, float>(inputA, inputB, outputType, act, inABias, aBatch0, aBatch1, aRows, aCols,
                bBatch0, bBatch1, bCols, deqBBias, inAShiftBits, inBShiftBits, shiftBits, dynamicChannel);
        }

        throw new ArgumentOutOfRangeException("inputA");
    }

    /// <summary>Allocates the output tensor (element type <typeparamref name="TO"/>) and fills it with the kernel.</summary>
    private static Tensor RunKernel<TIA, TIB, TO>(ReadOnlySpan<TIA> inputA, ReadOnlySpan<TIB> inputB,
        TensorType outputType, ReadOnlySpan<Half> act, ReadOnlySpan<byte> inABias, int aBatch0, int aBatch1, int aRows,
        int aCols, int bBatch0, int bBatch1, int bCols, byte deqBBias, int inAShiftBits, int inBShiftBits,
        sbyte shiftBits, bool dynamicChannel) where TIA : unmanaged where TIB : unmanaged where TO : unmanaged, IEquatable<TO>
    {
        Tensor<TO> output = new Tensor<TO>(outputType.Shape.ToValueArray());
        K230Kernels.DynamicGnneMatmul(inputA, inputB, output.Buffer.Span, act, inABias, aBatch0, aBatch1, aRows,
            aCols, bBatch0, bBatch1, bCols, deqBBias, inAShiftBits, inBShiftBits, shiftBits, dynamicChannel);
        return output;
    }

    /// <summary>Shape / quantization arguments shared by every input element type combination.</summary>
    private struct MatMulArgs
    {
        public TensorType OutputType;
        public Tensor<Half> Act;
        public Tensor<byte> InputABias;
        public int ABatch0;
        public int ABatch1;
        public int ARows;
        public int ACols;
        public int BBatch0;
        public int BBatch1;
        public int BCols;
        public byte InputBBias;
        public int ShiftBits;
        public bool DynamicChannel;
    }

    /// <summary>Calls GnneMatmulV2 for already typed inputs and wraps the result.</summary>
    private static IValue RunMatMul<TIA, TIB>(ReadOnlySpan<TIA> inputA, ReadOnlySpan<TIB> inputB, MatMulArgs args)
        where TIA : unmanaged where TIB : unmanaged
    {
        // The input A / input B shift bits are always 0 here; the shift bits cast is overflow-checked.
        return Value.FromTensor(GnneMatmulV2(inputA, inputB, args.OutputType, args.Act.Buffer.Span,
            args.InputABias.Buffer.Span, args.ABatch0, args.ABatch1, args.ARows, args.ACols, args.BBatch0,
            args.BBatch1, args.BCols, args.InputBBias, 0, 0, checked((sbyte)args.ShiftBits), args.DynamicChannel));
    }

    private IValue Visit(Tensor inputA, Tensor inputB, Tensor<Half> act, Tensor<byte> inputABias, byte inputBBias,
        DynamicGNNEMatMul target, int shiftBits, int dynamicChannel)
    {
        // Output shape: input A's shape with the last dimension replaced by input B's last dimension.
        int[] outputShape = inputA.Shape.ToValueArray();
        int lastDimIndex = outputShape.Length - 1;
        Shape inputBShape = inputB.Shape;
        outputShape[lastDimIndex] = inputBShape[inputBShape.Count - 1].FixedValue;
        TensorType outputType = new TensorType(target.OutputDType, outputShape);

        // Both inputs are expected to be 4-D: [batch0, batch1, rows, cols].
        MatMulArgs args = new MatMulArgs
        {
            OutputType = outputType,
            Act = act,
            InputABias = inputABias,
            ABatch0 = inputA.Shape[0].FixedValue,
            ABatch1 = inputA.Shape[1].FixedValue,
            ARows = inputA.Shape[2].FixedValue,
            ACols = inputA.Shape[3].FixedValue,
            BBatch0 = inputB.Shape[0].FixedValue,
            BBatch1 = inputB.Shape[1].FixedValue,

            // NOTE: input B's dimension 2 (its rows, equal to aCols) is not read; only dimension 3 is used.
            BCols = inputB.Shape[3].FixedValue,
            InputBBias = inputBBias,
            ShiftBits = shiftBits,
            DynamicChannel = dynamicChannel == 1,
        };

        DataType inputAType = inputA.ElementType;
        DataType inputBType = inputB.ElementType;

        // The raw bytes of each input are reinterpreted as its element type.
        if (inputAType == DataTypes.UInt8 && inputBType == DataTypes.UInt8)
        {
            return RunMatMul<byte, byte>(inputA.BytesBuffer, inputB.BytesBuffer, args);
        }

        if (inputAType == DataTypes.UInt8 && inputBType == DataTypes.Int8)
        {
            return RunMatMul<byte, sbyte>(inputA.BytesBuffer, MemoryMarshal.Cast<byte, sbyte>(inputB.BytesBuffer),
                args);
        }

        if (inputAType == DataTypes.UInt8 && inputBType == DataTypes.Int16)
        {
            return RunMatMul<byte, short>(inputA.BytesBuffer, MemoryMarshal.Cast<byte, short>(inputB.BytesBuffer),
                args);
        }

        if (inputAType == DataTypes.Int8 && inputBType == DataTypes.UInt8)
        {
            return RunMatMul<sbyte, byte>(MemoryMarshal.Cast<byte, sbyte>(inputA.BytesBuffer), inputB.BytesBuffer,
                args);
        }

        if (inputAType == DataTypes.Int8 && inputBType == DataTypes.Int8)
        {
            return RunMatMul<sbyte, sbyte>(MemoryMarshal.Cast<byte, sbyte>(inputA.BytesBuffer),
                MemoryMarshal.Cast<byte, sbyte>(inputB.BytesBuffer), args);
        }

        if (inputAType == DataTypes.Int8 && inputBType == DataTypes.Int16)
        {
            return RunMatMul<sbyte, short>(MemoryMarshal.Cast<byte, sbyte>(inputA.BytesBuffer),
                MemoryMarshal.Cast<byte, short>(inputB.BytesBuffer), args);
        }

        if (inputAType == DataTypes.Int16 && inputBType == DataTypes.UInt8)
        {
            return RunMatMul<short, byte>(MemoryMarshal.Cast<byte, short>(inputA.BytesBuffer), inputB.BytesBuffer,
                args);
        }

        if (inputAType == DataTypes.Int16 && inputBType == DataTypes.Int8)
        {
            return RunMatMul<short, sbyte>(MemoryMarshal.Cast<byte, short>(inputA.BytesBuffer),
                MemoryMarshal.Cast<byte, sbyte>(inputB.BytesBuffer), args);
        }

        throw new ArgumentOutOfRangeException(
            $"Invalid Input A {inputA.ElementType} or Input B {inputB.ElementType}");
    }

    private IRType Visit(TensorType inputA, TensorType inputB, DynamicGNNEMatMul target)
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

        // Trailing [rows of A, cols of B]; the leading (batch) dimensions come from the higher-rank input.
        Dimension[] matrixDims = new Dimension[2];
        Shape inputAShape = inputA.Shape;
        matrixDims[0] = inputAShape[inputAShape.Count - 2];
        Shape inputBShape = inputB.Shape;
        matrixDims[1] = inputBShape[inputBShape.Count - 1];
        Shape batchSource = (inputA.Shape.Rank > inputB.Shape.Rank) ? inputA.Shape : inputB.Shape;
        return new TensorType(target.OutputDType,
            batchSource.ToArray()[..(batchSource.Count - 2)].Concat(matrixDims).ToArray());
    }

    public IValue Visit(IEvaluateContext context, DynamicGNNEMatMul target)
    {
        Tensor inputA = context.GetArgumentValueAsTensor(target, DynamicGNNEMatMul.InputA);
        Tensor inputB = context.GetArgumentValueAsTensor(target, DynamicGNNEMatMul.InputB);
        Tensor<Half> act = context.GetArgumentValueAsTensor<Half>(target, DynamicGNNEMatMul.Act);
        Tensor<byte> inputABias = context.GetArgumentValueAsTensor<byte>(target, DynamicGNNEMatMul.InputABias);
        byte inputBBias = context.GetArgumentValueAsScalar<byte>(target, DynamicGNNEMatMul.InputBBias);
        int shiftBits = context.GetArgumentValueAsScalar<int>(target, DynamicGNNEMatMul.ShiftBits);
        int dynamicChannel = context.GetArgumentValueAsScalar<int>(target, DynamicGNNEMatMul.DynamicChannel);
        return Visit(inputA, inputB, act, inputABias, inputBBias, target, shiftBits, dynamicChannel);
    }

    public IRType Visit(ITypeInferenceContext context, DynamicGNNEMatMul target)
    {
        TensorType inputA = context.CheckArgumentType<TensorType>(target, DynamicGNNEMatMul.InputA);
        TensorType inputB = context.CheckArgumentType<TensorType>(target, DynamicGNNEMatMul.InputB);
        context.CheckArgumentType<IRType>(target, DynamicGNNEMatMul.Text);
        context.CheckArgumentType<IRType>(target, DynamicGNNEMatMul.InputA);
        context.CheckArgumentType<IRType>(target, DynamicGNNEMatMul.InputB);
        context.CheckArgumentType<IRType>(target, DynamicGNNEMatMul.InputABias);
        context.CheckArgumentType<IRType>(target, DynamicGNNEMatMul.InputBBias);
        context.CheckArgumentType<IRType>(target, DynamicGNNEMatMul.Act);
        context.CheckArgumentType<IRType>(target, DynamicGNNEMatMul.ShiftBits);
        context.CheckArgumentType<IRType>(target, DynamicGNNEMatMul.DynamicChannel);
        return Visit(inputA, inputB, target);
    }
}
