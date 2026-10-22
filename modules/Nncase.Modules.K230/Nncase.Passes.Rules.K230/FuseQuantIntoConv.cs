// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.IR.Math;
using Nncase.Passes.Analysis;
using Nncase.PatternMatch;
using Nncase.Utilities;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class FuseQuantIntoConv : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = FuseQuantHelper.MakePattern<GNNEConv2D>("convCall", "conv");

    private Expr? GetReplace(Quantize quant, QuantParam qp, Call st, Expr stStrides, GNNEConv2D conv,
        IReadOnlyList<Expr> convCallParams, RunPassContext context)
    {
        if (context.GetAnalysis<IExprUserAnalysisResult>()[st].Count() > 1)
        {
            return null;
        }

        ActParamBase actParamBase = FuseQuantHelper.FuseActAndQuant(quant, qp, conv.ActParam);
        return Tensors.GNNEStore(
            quant.TargetType,
            ReplaceUtility.ReplaceCallParams(
                conv.With((ActParam2)actParamBase, null, (PrimType)quant.TargetType),
                convCallParams, (GNNEConv2D.Act, GetReplaceHelper.LoadAct0((ActParam2)actParamBase))), stStrides);
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Quantize quant = (Quantize)result["quant"];
        QuantParam qp = ((TensorConst)result["qp"]).Value.ToScalar<QuantParam>();
        Call st = (Call)result["st"];
        Expr stStrides = (Expr)result["stStrides"];
        GNNEConv2D conv = (GNNEConv2D)result["conv"];
        IReadOnlyList<Expr> convCallParams = (IReadOnlyList<Expr>)result["convCallParams"];
        return GetReplace(quant, qp, st, stStrides, conv, convCallParams, context);
    }
}
