// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.Tile;

internal sealed class GNNEFusionConcatMergeRule : IMergeRewriteRule
{
    private Pattern? _pattern;

    public string ModuleKind => "k230";

    public IPattern Pattern => _pattern ?? (_pattern = CreatePattern(ModuleKind));

    public Pattern CreatePattern(string target_module_kind)
    {
        FusionPattern target = Nncase.PatternMatch.Utility.IsFusion("caller_fusion", target_module_kind,
            Nncase.PatternMatch.Utility.IsWildcard(),
            Nncase.PatternMatch.Utility.IsVArgs(Nncase.PatternMatch.Utility.IsWildcard()));
        Pattern[] array = new Pattern[1];
        Func<Concat, bool> condition = (Concat _) => true;
        Func<Pattern> creator = () => Nncase.PatternMatch.Utility.IsWildcard();
        array[0] = Nncase.PatternMatch.F.Tensors.IsConcat("concat", "callee", condition,
                Nncase.PatternMatch.Utility.IsTuple("tuple",
                    Nncase.PatternMatch.Utility.IsVArgsRepeat("tupleInputs", creator)))with
            {
                TypePattern = TypePatternUtility.IsFloat()
            };
        return Nncase.PatternMatch.Utility.IsCall("caller", target, array);
    }

    public Expr? GetReplace(Func<Expr, Expr> mergedFusionRewriteCallBack,
        Func<Fusion, HashSet<Fusion>, bool> mergedFusionCheckCallBack,
        Func<HashSet<Fusion>, bool> candidateFusionCheckCallBack, Action<HashSet<Fusion>> candidateFusionRecordCallBack,
        IExprUserAnalysisResult usedByReslut, IMatchResult result, RunPassContext options)
    {
        Call call = (Call)result["callee"];
        Fusion caller_fusion = (Fusion)result["caller_fusion"];
        Concat concat = (Concat)result["concat"];
        if (concat.Axis != 1 || call.CheckedShape[0] != 1)
        {
            return null;
        }

        if (usedByReslut[call].Count() > 1)
        {
            return null;
        }

        if (!ProcessFusionMerge(mergedFusionRewriteCallBack, candidateFusionCheckCallBack, concat, caller_fusion,
                result, out HashSet<Fusion> candidate_fusions, out Fusion merged_fusion))
        {
            return null;
        }

        if (mergedFusionCheckCallBack(merged_fusion, candidate_fusions))
        {
            return new Call(merged_fusion, ((IReadOnlyList<Expr>)result["tupleInputs"]).ToArray());
        }

        return null;
    }

    private bool ProcessFusionMerge(Func<Expr, Expr> mergedFusionRewriteCallBack,
        Func<HashSet<Fusion>, bool> candidate_fusion_checker, Concat concat, Fusion caller_fusion, IMatchResult result,
        out HashSet<Fusion> candidate_fusions, out Fusion merged_fusion)
    {
        //IL_00d0: Unknown result type (might be due to invalid IL or missing references)
        //IL_00d6: Unknown result type (might be due to invalid IL or missing references)
        Var[] array = (from i in ((IReadOnlyList<Expr>)result["tupleInputs"]).ToArray()
            select new Var(new TensorType(i.CheckedDataType, i.CheckedShape))).ToArray();
        merged_fusion = null;
        candidate_fusions = new HashSet<Fusion> { caller_fusion };
        string name = caller_fusion.Name + "_concat";
        Dictionary<Var, Expr> dictionary = new Dictionary<Var, Expr>();
        Var key = caller_fusion.Parameters[0];
        Expr[] fields = array;
        dictionary.Add(key, Nncase.IR.F.Tensors.Concat(new Nncase.IR.Tuple(fields), concat.Axis));
        Expr arg = new IMergeRewriteRule.FusionMerger(dictionary,
                new Dictionary<Var, Var>(array.Select((Var i) => new KeyValuePair<Var, Var>(i, i))))
            .Clone(caller_fusion.Body, default(Unit));
        arg = mergedFusionRewriteCallBack(arg);
        if (!arg.InferenceType())
        {
            throw new InvalidOperationException("Merged Fusion Type Infer Error!");
        }

        merged_fusion = new Fusion(name, ModuleKind, arg, array);
        return true;
    }
}
