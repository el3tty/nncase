// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class ReduceToGlobalReduceWindow : IRewriteRule
{
    public IPattern Pattern { get; } = Nncase.PatternMatch.F.Math.IsReduce("reduce", "call", (Reduce _) => true,
        Nncase.PatternMatch.Utility.IsWildcard("input") with { TypePattern = TypePatternUtility.HasFixedShape() },
        Nncase.PatternMatch.Utility.IsTensorConst("axis"), Nncase.PatternMatch.Utility.IsTensorConst("initValue"),
        Nncase.PatternMatch.Utility.IsTensorConst());

    private Expr? GetReplace(Reduce reduce, Call call, Expr input, TensorConst axis, TensorConst initValue)
    {
        int[] inputShape = input.CheckedShape.ToValueArray();
        int[] axisValue = axis.Value.ToArray<int>();
        if (!TryMatch(call.CheckedShape.ToValueArray(), inputShape, axisValue) || reduce.ReduceOp == ReduceOp.Prod)
        {
            return null;
        }

        int[] filter = new int[2] { inputShape[2], inputShape[3] };
        int[] stride = new int[2] { 1, 1 };
        int[] dilation = new int[2] { 1, 1 };
        int[,] padding = new int[2, 2];
        return Nncase.IR.F.NN.ReduceWindow2D(reduce.ReduceOp, input, initValue, filter, stride, padding, dilation, false,
            false);
    }

    private bool TryMatch(int[] outShape, int[] inShape, int[] axisValue)
    {
        if (outShape.Length == 4 && inShape.Length == 4 && axisValue.Length == 2 &&
            (axisValue[0] == 2 || axisValue[0] == -2) && (axisValue[1] == 3 || axisValue[1] == -1))
        {
            return true;
        }

        return false;
    }

    public Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Reduce reduce = (Reduce)result["reduce"];
        Call call = (Call)result["call"];
        Expr input = (Expr)result["input"];
        TensorConst axis = (TensorConst)result["axis"];
        TensorConst initValue = (TensorConst)result["initValue"];
        return GetReplace(reduce, call, input, axis, initValue);
    }
}
