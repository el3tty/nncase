// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.K230;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class GNNEMac : IRewriteRule
{
    public IPattern Pattern { get; } = Nncase.PatternMatch.F.K230.IsGNNEFusion("gnne_fusion", "call",
        (GNNEFusion _) => true, Nncase.PatternMatch.Utility.IsTensorConst("input"));

    public Expr? GetReplace(GNNEFusion gnne_fusion, Call call, Expr input)
    {
        return input;
    }

    public Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        GNNEFusion gnneFusion = (GNNEFusion)result["gnne_fusion"];
        Call call = (Call)result["call"];
        Expr input = (Expr)result["input"];
        return GetReplace(gnneFusion, call, input);
    }
}
