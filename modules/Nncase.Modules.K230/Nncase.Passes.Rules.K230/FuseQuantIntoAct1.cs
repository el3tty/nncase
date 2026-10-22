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
public sealed class FuseQuantIntoAct1 : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = FuseQuantHelper.MakePattern<GNNEActivation>("actCall", "act");

    private Expr? GetReplace(Quantize quant, QuantParam qp, Call st, Expr stStrides, GNNEActivation act,
        IReadOnlyList<Expr> actCallParams, RunPassContext context)
    {
        if (context.GetAnalysis<IExprUserAnalysisResult>()[st].Count() > 1)
        {
            return null;
        }

        ActParamBase actParamBase = (act.ActParam.N == 2)
            ? FuseQuantHelper.FuseActAndQuant(quant, qp, (ActParam2)act.ActParam)
            : FuseQuantHelper.FuseActAndQuant(quant, qp, (ActParam16)act.ActParam);
        DataType targetType = quant.TargetType;
        ActParamBase actParam = actParamBase;
        PrimType outputDType = (PrimType)quant.TargetType;
        return Tensors.GNNEStore(
            targetType,
            ReplaceUtility.ReplaceCallParams(act.With(null, outputDType, actParam), actCallParams,
                (GNNEActivation.Act, GetReplaceHelper.LoadAct1(actParamBase))), stStrides);
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Quantize quant = (Quantize)result["quant"];
        QuantParam qp = ((TensorConst)result["qp"]).Value.ToScalar<QuantParam>();
        Call st = (Call)result["st"];
        Expr stStrides = (Expr)result["stStrides"];
        GNNEActivation act = (GNNEActivation)result["act"];
        IReadOnlyList<Expr> actCallParams = (IReadOnlyList<Expr>)result["actCallParams"];
        return GetReplace(quant, qp, st, stStrides, act, actCallParams, context);
    }
}
