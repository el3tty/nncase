// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class PReluToFakeActivation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = NN.IsPRelu("prelu", "preluCall", (PRelu _) => true,
        Nncase.PatternMatch.Utility.IsWildcard("input"), Nncase.PatternMatch.Utility.IsTensorConst("slope"));

    public Expr? GetReplace(PRelu prelu, Call preluCall, TensorConst slope, Expr input)
    {
        int[] inputShape = new int[4] { 1, 1, 1, 1 };
        Array.Copy(preluCall.CheckedShape.ToValueArray(), 0, inputShape, inputShape.Length - preluCall.CheckedShape.Rank,
            preluCall.CheckedShape.Rank);
        int actChannels = inputShape[1];
        ActParam2 actParam = new ActParam2(actChannels, new QuantParam(0, 1f));
        float[] slopes = slope.Value.ToArray<float>();
        for (int i = 0; i < actParam.Ks.GetLength(1); i++)
        {
            if (slopes.Length == 1)
            {
                actParam.Ks[0, i] = slopes[0];
            }
            else if (slopes.Length == actChannels)
            {
                actParam.Ks[0, i] = slopes[i];
            }
            else
            {
                actParam.Ks[0, i] = slopes[i * slopes.Length / actChannels];
            }
        }

        Tensor<float> tensor = new Tensor<float>(actParam.GetAct1Data, new int[2] { actChannels, 7 });
        return Nncase.IR.K230.F.Tensors.FakeActivation(input, None.Default, tensor, actChannels, 0, 0, 0, false,
            GnneActivationType.Uninitialized, actParam, preluCall.CheckedShape.ToValueArray());
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        PRelu prelu = (PRelu)__result["prelu"];
        Call preluCall = (Call)__result["preluCall"];
        TensorConst slope = (TensorConst)__result["slope"];
        Expr input = (Expr)__result["input"];
        return GetReplace(prelu, preluCall, slope, input);
    }
}
