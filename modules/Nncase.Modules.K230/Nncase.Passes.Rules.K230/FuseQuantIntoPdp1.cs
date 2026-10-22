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
public sealed class FuseQuantIntoPdp1 : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = FuseQuantHelper.MakePattern<GNNEPdp1>("pdpCall", "pdp");

    private Expr? GetReplace(Quantize quant, QuantParam qp, Call st, Expr stStrides, GNNEPdp1 pdp,
        IReadOnlyList<Expr> pdpCallParams, RunPassContext context)
    {
        if (context.GetAnalysis<IExprUserAnalysisResult>()[st].Count() > 1)
        {
            return null;
        }

        QuantizeParam[] quantParams = GNNETypePatternUtility.GNNEGetQuantParams(1, qp.Scale, qp.ZeroPoint);
        DataType targetType = quant.TargetType;
        PrimType destType = (PrimType)quant.TargetType;
        return Tensors.GNNEStore(
            targetType,
            ReplaceUtility.ReplaceCallParams(pdp.With(null, destType), pdpCallParams, (GNNEPdp1.QuantParams, quantParams)),
            stStrides);
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Quantize quant = (Quantize)result["quant"];
        QuantParam qp = ((TensorConst)result["qp"]).Value.ToScalar<QuantParam>();
        Call st = (Call)result["st"];
        Expr stStrides = (Expr)result["stStrides"];
        GNNEPdp1 pdp = (GNNEPdp1)result["pdp"];
        IReadOnlyList<Expr> pdpCallParams = (IReadOnlyList<Expr>)result["pdpCallParams"];
        return GetReplace(quant, qp, st, stStrides, pdp, pdpCallParams, context);
    }
}
