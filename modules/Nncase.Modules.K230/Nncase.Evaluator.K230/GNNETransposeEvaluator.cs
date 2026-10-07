// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using OrtKISharp;

namespace Nncase.Evaluator.K230;

/// <summary>
/// Evaluator for <see cref="GNNETranspose"/>: transposes the input with the permutation derived from the op's Perm.
/// </summary>
[TypeInferGenerator]
public class GNNETransposeEvaluator : IEvaluator<GNNETranspose>, IEvaluator, ITypeInferencer<GNNETranspose>,
    ITypeInferencer, ICostEvaluator<GNNETranspose>, ICostEvaluator
{
    public Cost Visit(ICostEvaluateContext context, GNNETranspose target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    public IValue Visit(IEvaluateContext context, GNNETranspose tr)
    {
        OrtKISharp.Tensor input = context.GetOrtArgumentValue(tr, GNNETranspose.Input);
        long[] perm = GNNETypePatternUtility.ApplyPerm(tr.Perm);
        return OrtKI.Transpose(input, perm).ToValue();
    }

    private IRType Visit(ITypeInferenceContext context, GNNETranspose target, TensorType input)
    {
        GNNETypePatternUtility.CheckIsValidTransposeType(input.DType);
        long[] perm = GNNETypePatternUtility.ApplyPerm(target.Perm);
        return TypeInference.TransposeType(input, perm);
    }

    public IRType Visit(ITypeInferenceContext context, GNNETranspose target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, GNNETranspose.Input);

        // NOTE: redundant second check of the same argument (result unused); kept for identical behaviour.
        context.CheckArgumentType<IRType>(target, GNNETranspose.Input);
        return Visit(context, target, input);
    }
}
