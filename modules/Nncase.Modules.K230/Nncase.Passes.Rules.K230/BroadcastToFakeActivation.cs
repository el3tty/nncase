// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class BroadcastToFakeActivation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = Nncase.PatternMatch.F.Tensors.IsBroadcast("broadcast", "call",
        (Broadcast _) => true, Nncase.PatternMatch.Utility.IsWildcard("input")with
        {
            TypePattern = (TypePatternUtility.HasRank((int r) => r <= 4, "GNNE not support more than 4D") &
                           TypePatternUtility.HasFixedShape())
        }, Nncase.PatternMatch.Utility.IsTensorConst("shape"));

    public Expr? GetReplace(Broadcast broadcast, Call call, Expr input, TensorConst shape)
    {
        if (!TryMatch(broadcast, input, shape))
        {
            return null;
        }

        if (call.CheckedDataType != DataTypes.Float32)
        {
            return null;
        }

        int[] broadcastShape = new int[4] { 1, 1, 1, 1 };
        Array.Copy(shape.Value.ToArray<int>(), 0, broadcastShape, broadcastShape.Length - shape.Value.Length, shape.Value.Length);
        int actChannels = broadcastShape[1];
        ActParam2 actParam = new ActParam2(actChannels);
        Tensor<float> tensor = new Tensor<float>(actParam.GetAct1Data, new int[2] { actChannels, 7 });
        return Nncase.IR.K230.F.Tensors.FakeActivation(input, None.Default, tensor, actChannels, 0, 0, 0, false,
            GnneActivationType.Add, actParam, call.CheckedShape.ToValueArray());
    }

    private bool TryMatch(Broadcast broadcast, Expr input, TensorConst shape)
    {
        if (input.CheckedShape[0] == shape.Value.ToArray<int>()[0] &&
            input.CheckedShape[1] == shape.Value.ToArray<int>()[1])
        {
            return true;
        }

        return false;
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Broadcast broadcast = (Broadcast)__result["broadcast"];
        Call call = (Call)__result["call"];
        Expr input = (Expr)__result["input"];
        TensorConst shape = (TensorConst)__result["shape"];
        return GetReplace(broadcast, call, input, shape);
    }
}
