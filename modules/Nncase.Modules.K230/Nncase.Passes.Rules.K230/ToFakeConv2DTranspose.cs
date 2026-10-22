// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class ToFakeConv2DTranspose : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = NN.IsConv2DTranspose("conv2dTranspose", "call",
        (Conv2DTranspose _) => true,
        Nncase.PatternMatch.Utility.IsWildcard("input") with { TypePattern = TypePatternUtility.HasFixedShape() },
        Nncase.PatternMatch.Utility.IsWildcard("weights") with { TypePattern = TypePatternUtility.HasFixedShape() },
        Nncase.PatternMatch.Utility.IsTensorConst("bias"), Nncase.PatternMatch.Utility.IsWildcard("outShape"),
        Nncase.PatternMatch.Utility.IsWildcard("stride"), Nncase.PatternMatch.Utility.IsWildcard("padding"),
        Nncase.PatternMatch.Utility.IsWildcard("outPadding"), Nncase.PatternMatch.Utility.IsWildcard("dilation"),
        Nncase.PatternMatch.Utility.IsTensorConst("groups"), Nncase.PatternMatch.Utility.IsTensorConst("fusedClamp"));

    private Expr? GetReplace(Conv2DTranspose conv2dTranspose, Call call, Expr input, Expr weights, TensorConst bias,
        Expr outShape, Expr stride, Expr padding, Expr outPadding, Expr dilation, int groups, Tensor<float> fusedClamp)
    {
        int outChannels = weights.CheckedShape[0].FixedValue * groups;
        ActParam2 defaultConvActParam = ActParam2.GetDefaultConvActParam(outChannels, bias);
        for (int i = 0; i < outChannels; i++)
        {
            defaultConvActParam.FusedClamp[i].Min = fusedClamp[new int[1]];
            defaultConvActParam.FusedClamp[i].Max = fusedClamp[new int[1] { 1 }];
        }

        return Nncase.IR.K230.F.Tensors.FakeConv2DTranspose(input, weights, defaultConvActParam.ToFakeActData(),
            outShape, padding, outPadding, stride, dilation, groups, 0, defaultConvActParam);
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Conv2DTranspose conv2dTranspose = (Conv2DTranspose)result["conv2dTranspose"];
        Call call = (Call)result["call"];
        Expr input = (Expr)result["input"];
        Expr weights = (Expr)result["weights"];
        TensorConst bias = (TensorConst)result["bias"];
        Expr outShape = (Expr)result["outShape"];
        Expr stride = (Expr)result["stride"];
        Expr padding = (Expr)result["padding"];
        Expr outPadding = (Expr)result["outPadding"];
        Expr dilation = (Expr)result["dilation"];
        int groups = ((TensorConst)result["groups"]).Value.ToScalar<int>();
        Tensor<float> fusedClamp = ((TensorConst)result["fusedClamp"]).Value.Cast<float>();
        return GetReplace(conv2dTranspose, call, input, weights, bias, outShape, stride, padding, outPadding, dilation,
            groups, fusedClamp);
    }
}
