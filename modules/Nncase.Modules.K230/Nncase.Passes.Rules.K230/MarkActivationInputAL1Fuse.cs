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
using Nncase.PatternMatch.F;
using Nncase.Utilities;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class MarkActivationInputAL1Fuse : RewriteRule<Pattern>, IRewriteRule
{
    public override Pattern Pattern { get; } = Nncase.PatternMatch.Utility.IsCallWildcard(
        "actCall",
        Nncase.PatternMatch.Utility.IsOp<GNNEActivation>("act"),
        Nncase.PatternMatch.F.K230.IsGNNELoad("ld", "ldCall", (GNNELoad _) => true,
            Nncase.PatternMatch.Utility.IsCallWildcard("quantCall", Nncase.PatternMatch.Utility.IsOp<Quantize>(),
                Nncase.PatternMatch.Utility.IsCallWildcard("stCall", Nncase.PatternMatch.Utility.IsOp<GNNEStore>(),
                    Nncase.PatternMatch.Utility.IsCallWildcard(
                        "convCall",
                        Nncase.PatternMatch.Utility.IsOp<GNNEConv2D>(),
                        Nncase.PatternMatch.Utility.IsWildcard("input"))))),
        Nncase.PatternMatch.Utility.IsWildcard("inputB"));

    private Expr? GetReplace(GNNEActivation act, Call stCall, Call ldCall, Call actCall, Call convCall,
        IReadOnlyList<Expr> actCallParams, RunPassContext context, Call quantCall)
    {
        int[] convInputShape = convCall[GNNEConv2D.Input].CheckedShape.ToValueArray();
        int[] convOutputShape = convCall.CheckedShape.ToValueArray();
        int[] weightsShape = convCall[GNNEConv2D.Weights].CheckedShape.ToValueArray();
        int[] source = ((TensorConst)convCall[GNNEConv2D.Padding]).Value.ToArray<int>();
        int strideH = ((TensorConst)convCall[GNNEConv2D.Stride]).Value.ToArray<int>()[0];
        int strideW = ((TensorConst)convCall[GNNEConv2D.Stride]).Value.ToArray<int>()[1];
        int dilationH = ((TensorConst)convCall[GNNEConv2D.Dilation]).Value.ToArray<int>()[0];
        int dilationW = ((TensorConst)convCall[GNNEConv2D.Dilation]).Value.ToArray<int>()[1];
        int groups = ((TensorConst)convCall[GNNEConv2D.Groups]).Value.ToScalar<int>();
        if (convInputShape[0] > 1 && convOutputShape[2] == 1 && convOutputShape[3] == 1 && convInputShape[2] == 1 && convInputShape[3] == 1 && weightsShape[2] == 1 &&
            weightsShape[3] == 1 && strideH == 1 && strideW == 1 && dilationH == 1 && dilationW == 1 && source.Sum() == 0 && groups == 1)
        {
            return null;
        }

        if (actCall[GNNEActivation.InputB] != None.Default &&
            actCall[GNNEActivation.InputA] == actCall[GNNEActivation.InputB])
        {
            return null;
        }

        IExprUserAnalysisResult analysis = context.GetAnalysis<IExprUserAnalysisResult>();
        if (analysis[stCall].Count() > 1 || analysis[quantCall].Count() > 1 || analysis[ldCall].Count() > 1 ||
            act.InputFromL1[1] || actCall.CheckedShape != convCall.CheckedShape)
        {
            return null;
        }

        Call actLoad = Nncase.IR.K230.F.Tensors.GNNELoad(DataTypes.Float16, stCall);
        IRArray<bool>? inputFromL = new bool[2] { true, false };
        return ReplaceUtility.ReplaceCallParams(act.With(null, null, null, null, inputFromL), actCallParams,
            (ldCall, actLoad));
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        GNNEActivation act = (GNNEActivation)result["act"];
        Call stCall = (Call)result["stCall"];
        Call ldCall = (Call)result["ldCall"];
        Call actCall = (Call)result["actCall"];
        Call convCall = (Call)result["convCall"];
        IReadOnlyList<Expr> actCallParams = (IReadOnlyList<Expr>)result["actCallParams"];
        Call quantCall = (Call)result["quantCall"];
        return GetReplace(act, stCall, ldCall, actCall, convCall, actCallParams, context, quantCall);
    }
}
