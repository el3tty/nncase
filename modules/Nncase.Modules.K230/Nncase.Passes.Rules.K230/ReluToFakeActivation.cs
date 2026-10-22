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
public class ReluToFakeActivation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = NN.IsRelu("relu", "reluCall", (Relu _) => true,
        Nncase.PatternMatch.Utility.IsWildcard("input") with { TypePattern = TypePatternUtility.HasFixedShape() });

    public Expr? GetReplace(Relu relu, Call reluCall, Expr input)
    {
        int[] inputShape = new int[4] { 1, 1, 1, 1 };
        Array.Copy(reluCall.CheckedShape.ToValueArray(), 0, inputShape, inputShape.Length - reluCall.CheckedShape.Rank,
            reluCall.CheckedShape.Rank);
        int actChannels = inputShape[1];
        ActParam2 actParam = new ActParam2(actChannels, new QuantParam(0, 1f));
        for (int i = 0; i < actParam.Ks.GetLength(1); i++)
        {
            actParam.Ks[0, i] = 0f;
            actParam.Bs[0, i] = 0f;
        }

        Tensor<float> tensor = new Tensor<float>(actParam.GetAct1Data, new int[2] { actChannels, 7 });
        return Nncase.IR.K230.F.Tensors.FakeActivation(input, None.Default, tensor, actChannels, 0, 0, 0, false,
            GnneActivationType.Uninitialized, actParam, reluCall.CheckedShape.ToValueArray());
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Relu relu = (Relu)result["relu"];
        Call reluCall = (Call)result["reluCall"];
        Expr input = (Expr)result["input"];
        return GetReplace(relu, reluCall, input);
    }
}
