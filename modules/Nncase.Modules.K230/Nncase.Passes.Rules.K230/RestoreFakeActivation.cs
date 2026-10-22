// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class RestoreFakeActivation : GNNEDIFQuantRule
{
    public override Pattern Pattern { get; } = Nncase.PatternMatch.F.K230.IsFakeActivation("fakeActivation", "call",
        (FakeActivation _) => true,
        Nncase.PatternMatch.Utility.IsRangeOfMarker(
            Nncase.PatternMatch.Utility.IsWildcard("inputA"),
            Nncase.PatternMatch.Utility.IsTensorConst("inputARange")),
        Nncase.PatternMatch.Utility.IsRangeOfMarker(
            Nncase.PatternMatch.Utility.IsWildcard("inputB"),
            Nncase.PatternMatch.Utility.IsTensorConst("inputBRange")), Nncase.PatternMatch.Utility.IsWildcard("act"),
        Nncase.PatternMatch.Utility.IsWildcard("outChannels"), Nncase.PatternMatch.Utility.IsWildcard("inAShiftbits"),
        Nncase.PatternMatch.Utility.IsWildcard("inBShiftbits"), Nncase.PatternMatch.Utility.IsWildcard("outShiftbits"),
        Nncase.PatternMatch.Utility.IsWildcard("is16Segment"));

    public Expr? GetReplace(FakeActivation fakeActivation, Call call, Expr inputA, Tensor<float> inputARange,
        Expr inputB, Tensor<float> inputBRange, Expr act, Expr outChannels, Expr inAShiftbits, Expr inBShiftbits,
        Expr outShiftbits, Expr is16Segment)
    {
        return Nncase.IR.F.Math.Binary(
            (fakeActivation.Type != GnneActivationType.Add) ? BinaryOp.Mul : BinaryOp.Add,
            inputA, inputB);
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        FakeActivation fakeActivation = (FakeActivation)result["fakeActivation"];
        Call call = (Call)result["call"];
        Expr inputA = (Expr)result["inputA"];
        Tensor<float> inputARange = ((TensorConst)result["inputARange"]).Value.Cast<float>();
        Expr inputB = (Expr)result["inputB"];
        Tensor<float> inputBRange = ((TensorConst)result["inputBRange"]).Value.Cast<float>();
        Expr act = (Expr)result["act"];
        Expr outChannels = (Expr)result["outChannels"];
        Expr inAShiftbits = (Expr)result["inAShiftbits"];
        Expr inBShiftbits = (Expr)result["inBShiftbits"];
        Expr outShiftbits = (Expr)result["outShiftbits"];
        Expr is16Segment = (Expr)result["is16Segment"];
        Option = context;
        MatchResult = result;
        Init();
        return GetReplace(fakeActivation, call, inputA, inputARange, inputB, inputBRange, act, outChannels,
            inAShiftbits, inBShiftbits, outShiftbits, is16Segment);
    }
}
