using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using OrtKISharp;

namespace Nncase.Evaluator.K230;

/// <summary>
/// Evaluator for <see cref="GNNEStore"/>: stores a tensor from GNNE memory, optionally widening float16 to float32.
/// </summary>
[EvaluatorGenerator]
[TypeInferGenerator]
public class GNNEStoreEvaluator : IEvaluator<GNNEStore>, IEvaluator, ITypeInferencer<GNNEStore>, ITypeInferencer,
    ICostEvaluator<GNNEStore>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, GNNEStore target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    private IValue Visit(GNNEStore target, Tensor input, Tensor strides)
    {
        DataType sourceType = input.ElementType;
        PrimType destType = target.DestType;

        // float16 -> float32 is the only supported conversion; otherwise the types must match.
        Tensor converted;
        if (sourceType == DataTypes.Float16 && destType == DataTypes.Float32)
        {
            converted = input.Cast<float>();
        }
        else
        {
            if (!(sourceType == destType))
            {
                throw new ArgumentOutOfRangeException($"{sourceType} => {destType}");
            }

            converted = input;
        }

        // Strided copy of the whole tensor: slice [0, dim) on every axis with step = strides[axis].
        // The named arguments below are evaluated in this order (starts, ends, axes, data, steps).
        return OrtKI.Slice(starts: Tensor.From(converted.Shape.Select((Dimension _) => 0L).ToArray()).ToOrtTensor(),
            ends: Tensor.From(((IEnumerable<Dimension>)converted.Shape)
                .Select((Func<Dimension, long>)((Dimension dim) => dim.FixedValue)).ToArray()).ToOrtTensor(),
            axes: Tensor.From(((IEnumerable<Dimension>)converted.Shape)
                .Select((Func<Dimension, int, long>)((Dimension _, int axis) => axis)).ToArray()).ToOrtTensor(),
            data: converted.ToOrtTensor(), steps: strides.ToOrtTensor()).ToValue();
    }

    private IRType Visit(ITypeInferenceContext context, GNNEStore target, TensorType input)
    {
        Tensor<int> strides = ((TensorConst)context.GetArgument(target, GNNEStore.Strides)).Value.Cast<int>();
        if (strides.Any((int stride) => stride != 1))
        {
            return new InvalidType("Not Support Stride != 1, Please Fix it.");
        }

        if (strides.Length != input.Shape.Rank)
        {
            return new InvalidType($"Stride Length {strides.Length} != Input Rank {input.Shape.Rank}");
        }

        DataType sourceType = input.DType;
        PrimType destType = target.DestType;
        if (sourceType != DataTypes.Float16 && destType == DataTypes.Float32)
        {
            return new InvalidType("when store output is float, input should be float16");
        }

        if (sourceType != destType && destType != DataTypes.Float32)
        {
            return new InvalidType("store input type and output type should be same");
        }

        if (sourceType != DataTypes.Int8 && sourceType != DataTypes.UInt8 && sourceType != DataTypes.Int16 &&
            sourceType != DataTypes.Float16 && sourceType != DataTypes.Float32)
        {
            return new InvalidType("store input type should be one of [int8, uint8, int16, float16]");
        }

        if (sourceType == DataTypes.Int8 || sourceType == DataTypes.UInt8 || sourceType == DataTypes.Int16 ||
            sourceType == DataTypes.Float16 || sourceType == DataTypes.Float32)
        {
            return input with { DType = destType };
        }

        // NOTE: unreachable (the checks above already reject every other source type); kept for identical structure.
        // The message below also says "Load" and has an unbalanced parenthesis; kept as-is.
        return new InvalidType("Not Support Load (Input: " + sourceType.GetDisplayName() + " or (Output: " +
                               destType.GetDisplayName());
    }

    public IValue Visit(IEvaluateContext context, GNNEStore target)
    {
        Tensor input = context.GetArgumentValueAsTensor(target, GNNEStore.Input);
        Tensor strides = context.GetArgumentValueAsTensor(target, GNNEStore.Strides);
        return Visit(target, input, strides);
    }

    public IRType Visit(ITypeInferenceContext context, GNNEStore target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, GNNEStore.Input);

        // NOTE: the Input check is redundant (result unused); kept for identical behaviour.
        context.CheckArgumentType<IRType>(target, GNNEStore.Input);
        context.CheckArgumentType<IRType>(target, GNNEStore.Strides);
        return Visit(context, target, input);
    }
}
