// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;

namespace Nncase.Evaluator.K230;

/// <summary>
/// Evaluator for <see cref="GNNELoadW"/>: loads weights into GNNE memory, optionally narrowing float32 to float16.
/// </summary>
[EvaluatorGenerator]
[TypeInferGenerator]
public class GNNELoadWEvaluator : IEvaluator<GNNELoadW>, IEvaluator, ITypeInferencer<GNNELoadW>, ITypeInferencer,
    ICostEvaluator<GNNELoadW>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, GNNELoadW target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    public IValue Visit(GNNELoadW target, Tensor input)
    {
        DataType sourceType = input.ElementType;
        PrimType destType = target.DestType;

        // float32 -> float16 is the only supported conversion; same-type loads are a plain copy.
        // NOTE: the type inferencer additionally accepts int8 <-> uint8, but this evaluator throws for it.
        if (sourceType == DataTypes.Float32 && destType == DataTypes.Float16)
        {
            return Value.FromTensor(input.Cast<Half>());
        }

        if (sourceType == destType)
        {
            return Value.FromTensor(input);
        }

        throw new NotSupportedException("GNNELoadVector Error With " + sourceType.GetDisplayName() + " => " +
                                        destType.GetDisplayName());
    }

    public IRType Visit(GNNELoadW target, TensorType input)
    {
        DataType sourceType = input.DType;
        PrimType destType = target.DestType;
        if (sourceType == DataTypes.Float32 && destType != DataTypes.Float16)
        {
            return new InvalidType("when load input type is float, output type should be float16");
        }

        // Besides identical types, only uint8 <-> int8 and float32 -> float16 conversions are allowed.
        bool isAllowedConversion =
            (sourceType == DataTypes.UInt8 && destType == DataTypes.Int8) ||
            (sourceType == DataTypes.Int8 && destType == DataTypes.UInt8) ||
            (sourceType == DataTypes.Float32 && destType == DataTypes.Float16);
        if (sourceType != destType && !isAllowedConversion)
        {
            return new InvalidType("load input type and output type should be same");
        }

        if (destType != DataTypes.Int8 && destType != DataTypes.Int16 && destType != DataTypes.Float16 &&
            destType != DataTypes.Float32 && destType != DataTypes.UInt8 &&
            destType != ExtDataTypes.DeQuantParam)
        {
            return new InvalidType("load output type should be one of [int8, int16, float16, uint8]");
        }

        if (sourceType == DataTypes.Int8 || sourceType == DataTypes.Int16 || sourceType == DataTypes.Float16 ||
            sourceType == DataTypes.Float32 || sourceType == DataTypes.UInt8 ||
            sourceType == ExtDataTypes.DeQuantParam)
        {
            // NOTE: the result keeps the *source* element type, not the destination type (unlike GNNELoad).
            return input with { DType = sourceType };
        }

        // NOTE: the message below has an unbalanced parenthesis; kept as-is.
        return new InvalidType("Not Support Load (Input: " + sourceType.GetDisplayName() + " or (Output: " +
                               destType.GetDisplayName());
    }

    public IValue Visit(IEvaluateContext context, GNNELoadW target)
    {
        Tensor input = context.GetArgumentValueAsTensor(target, GNNELoadW.Input);
        return Visit(target, input);
    }

    public IRType Visit(ITypeInferenceContext context, GNNELoadW target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, GNNELoadW.Input);

        // NOTE: redundant second check of the same argument (result unused); kept for identical behaviour.
        context.CheckArgumentType<IRType>(target, GNNELoadW.Input);
        return Visit(target, input);
    }
}
