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
public class LeakyReluToFakeActivation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = NN.IsLeakyRelu("leaky", "leakyCall", (LeakyRelu _) => true,
        Nncase.PatternMatch.Utility.IsWildcard("input"), Nncase.PatternMatch.Utility.IsTensorConst("alpha"));

    public Expr? GetReplace(LeakyRelu leaky, Call leakyCall, TensorConst alpha, Expr input)
    {
        int[] inputShape = new int[4] { 1, 1, 1, 1 };
        Array.Copy(leakyCall.CheckedShape.ToValueArray(), 0, inputShape, inputShape.Length - leakyCall.CheckedShape.Rank,
            leakyCall.CheckedShape.Rank);
        int actChannels = inputShape[1];
        ActParam2 actParam = new ActParam2(actChannels, new QuantParam(0, 1f));
        float alphaValue = alpha.Value.ToScalar<float>();
        for (int i = 0; i < actParam.Ks.GetLength(1); i++)
        {
            actParam.Ks[0, i] = alphaValue;
        }

        Tensor<float> tensor = new Tensor<float>(actParam.GetAct1Data, new int[2] { actChannels, 7 });
        return Nncase.IR.K230.F.Tensors.FakeActivation(input, None.Default, tensor, actChannels, 0, 0, 0, false,
            GnneActivationType.Uninitialized, actParam, leakyCall.CheckedShape.ToValueArray());
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        LeakyRelu leaky = (LeakyRelu)result["leaky"];
        Call leakyCall = (Call)result["leakyCall"];
        TensorConst alpha = (TensorConst)result["alpha"];
        Expr input = (Expr)result["input"];
        return GetReplace(leaky, leakyCall, alpha, input);
    }
}
