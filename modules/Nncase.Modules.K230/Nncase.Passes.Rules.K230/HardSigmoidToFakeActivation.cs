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
public class HardSigmoidToFakeActivation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = NN.IsHardSigmoid("hardSigmoid", "call", (HardSigmoid _) => true,
        Nncase.PatternMatch.Utility.IsWildcard("input") with { TypePattern = TypePatternUtility.HasFixedShape() },
        Nncase.PatternMatch.Utility.IsWildcard("alpha"), Nncase.PatternMatch.Utility.IsWildcard("beta"));

    private Expr? GetReplace(Call call, Expr input, float alpha, float beta)
    {
        int[] inputShape = input.CheckedShape.ToValueArray();
        int[] paddedShape = new int[4] { 1, 1, 1, 1 };
        Array.Copy(inputShape, 0, paddedShape, paddedShape.Length - inputShape.Length, inputShape.Length);
        int actChannels = paddedShape[1];
        ActParam2 actParam = new ActParam2(actChannels);
        actParam.ForEachChannel(delegate(ActParam2 param, int i)
        {
            param.Ks[0, i] = alpha;
            param.Bs[0, i] = beta;
            param.Xs[0, i] = 0f;
            param.Ks[1, i] = alpha;
            param.Bs[1, i] = beta;
        });
        actParam.SetFusedClamp(new ValueRange<float>(0f, 1f));
        Tensor<float> tensor = new Tensor<float>(actParam.GetAct1Data, new int[2] { actChannels, 7 });
        return Nncase.IR.K230.F.Tensors.FakeActivation(input, None.Default, tensor, actChannels, 0, 0, 0, false,
            GnneActivationType.Uninitialized, actParam, call.CheckedShape.ToValueArray());
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Call call = (Call)result["call"];
        Expr input = (Expr)result["input"];
        float alpha = ((TensorConst)result["alpha"]).Value.ToScalar<float>();
        float beta = ((TensorConst)result["beta"]).Value.ToScalar<float>();
        return GetReplace(call, input, alpha, beta);
    }
}
