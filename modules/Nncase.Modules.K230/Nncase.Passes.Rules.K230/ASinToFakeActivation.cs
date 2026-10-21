// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class ASinToFakeActivation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; }

    public Expr? GetReplace(Unary u, Call call, Expr input)
    {
        ActParam16 actParam = new ActParam16(1);
        ActFun actFun = new ActFun
        {
            SplitPoint0 = -1f,
            SplitPoint14 = 1f,
            SplitPointCenter = 0f,
            CenterPoint = 7,
            MinParam = new List<float> { 0f, -(float)System.Math.PI / 2f },
            MaxParam = new List<float> { 0f, (float)System.Math.PI / 2f }
        };
        actFun.Func = (float x) => MathF.Asin(x);
        Set_seg_fitting_param(actParam, actFun);
        Tensor<float> tensor = new Tensor<float>(actParam.GetAct1Data, new int[2] { 1, 49 });
        return Nncase.IR.K230.F.Tensors.FakeActivation(input, None.Default, tensor, 1, 0, 0, 0, true,
            GnneActivationType.Uninitialized, actParam, call.CheckedShape.ToValueArray());
    }

    public void Set_seg_fitting_param(ActParam16 actParam, ActFun f)
    {
        float[,] xs = actParam.Xs;
        int channel = 0;
        xs[0, channel] = f.SplitPoint0;
        xs[14, channel] = f.SplitPoint14;
        int centerPoint = f.CenterPoint;
        xs[centerPoint, channel] = f.SplitPointCenter;
        for (int i = 1; i < centerPoint; i++)
        {
            xs[i, channel] = (xs[centerPoint, channel] - xs[0, channel]) / (float)centerPoint * (float)i + xs[0, channel];
        }

        for (int j = centerPoint + 1; j < 15; j++)
        {
            xs[j, channel] = (xs[14, channel] - xs[centerPoint, channel]) / (float)(14 - centerPoint) * (float)(j - centerPoint) +
                         xs[centerPoint, channel];
        }

        actParam.Ks[0, channel] = f.MinParam[0];
        actParam.Bs[0, channel] = f.MinParam[1];
        actParam.Ks[15, channel] = f.MaxParam[0];
        actParam.Bs[15, channel] = f.MaxParam[1];
        for (int k = 1; k < 15; k++)
        {
            float slope = (f.Func(xs[k, channel]) - f.Func(xs[k - 1, channel])) / (xs[k, channel] - xs[k - 1, channel]);
            float intercept = f.Func(xs[k, channel]) - slope * xs[k, channel];
            actParam.Ks[k, channel] = slope;
            actParam.Bs[k, channel] = intercept;
        }

        actParam.SetFusedClamp(ValueRange<float>.Full);
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Unary u = (Unary)__result["u"];
        Call call = (Call)__result["call"];
        Expr input = (Expr)__result["input"];
        return GetReplace(u, call, input);
    }

    public ASinToFakeActivation()
    {
        Func<Unary, bool> condition = (Unary b) => b.UnaryOp == UnaryOp.Asin;
        Pattern = Nncase.PatternMatch.F.Math.IsUnary("u", "call", condition,
                Nncase.PatternMatch.Utility.IsWildcard("input")) with
            {
                TypePattern = TypePatternUtility.HasDataType(DataTypes.Float32)
            };
    }
}
