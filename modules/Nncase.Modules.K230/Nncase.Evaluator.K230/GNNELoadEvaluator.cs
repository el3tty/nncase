// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;

namespace Nncase.Evaluator.K230;

/// <summary>
/// Evaluator for <see cref="GNNELoad"/>: loads a tensor into GNNE memory, optionally narrowing float32 to float16.
/// </summary>
[EvaluatorGenerator]
[TypeInferGenerator]
public class GNNELoadEvaluator : IEvaluator<GNNELoad>, IEvaluator, ITypeInferencer<GNNELoad>, ITypeInferencer,
    ICostEvaluator<GNNELoad>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, GNNELoad target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    public IValue Visit(GNNELoad target, Tensor input)
    {
        DataType sourceType = input.ElementType;
        PrimType destType = target.DestType;

        // float32 -> float16 is the only supported conversion; same-type loads are a plain copy.
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

    public IRType Visit(GNNELoad target, TensorType input)
    {
        DataType sourceType = input.DType;
        PrimType destType = target.DestType;
        if (sourceType == DataTypes.Float32 && destType != DataTypes.Float16)
        {
            return new InvalidType("when load input type is float, output type should be float16");
        }

        if (sourceType != destType && sourceType != DataTypes.Float32)
        {
            return new InvalidType("load input type and output type should be same");
        }

        if (destType != DataTypes.Int8 && destType != DataTypes.Int16 && destType != DataTypes.Float16 &&
            destType != DataTypes.Float32 && destType != DataTypes.UInt8)
        {
            return new InvalidType("load output type should be one of [int8, int16, float16, uint8]");
        }

        if (sourceType == DataTypes.Int8 || sourceType == DataTypes.Int16 || sourceType == DataTypes.Float16 ||
            sourceType == DataTypes.Float32 || sourceType == DataTypes.UInt8)
        {
            return input with { DType = destType };
        }

        // NOTE: the message below has an unbalanced parenthesis; kept as-is.
        return new InvalidType("Not Support Load (Input: " + sourceType.GetDisplayName() + " or (Output: " +
                               destType.GetDisplayName());
    }

    public IValue Visit(IEvaluateContext context, GNNELoad target)
    {
        Tensor input = context.GetArgumentValueAsTensor(target, GNNELoad.Input);
        return Visit(target, input);
    }

    public IRType Visit(ITypeInferenceContext context, GNNELoad target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, GNNELoad.Input);

        // NOTE: redundant second check of the same argument (result unused); kept for identical behaviour.
        context.CheckArgumentType<IRType>(target, GNNELoad.Input);
        return Visit(target, input);
    }
}
