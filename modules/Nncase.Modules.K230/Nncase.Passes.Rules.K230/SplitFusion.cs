// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230.F;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
internal sealed class SplitFusion : FusionMaker
{
    public override string Name { get; } = "TileSplitCase";

    public override string ModuleKind { get; } = "k230";

    public override Pattern Pattern { get; } = Nncase.PatternMatch.F.Tensors.IsSplit("split", "splitCall",
        (Split _) => true, Nncase.PatternMatch.Utility.IsWildcard("input") with
        {
            TypePattern = TypePatternUtility.IsFloat() & TypePatternUtility.HasRank(4) &
                           TypePatternUtility.HasFixedShape(),
        }, Nncase.PatternMatch.Utility.IsTensorConst("axis"), Nncase.PatternMatch.Utility.IsTensorConst("sections"));

    private Call? GetReplace(Expr input, Call splitCall, Split split, int axis, int[] sections)
    {
        DataType loadDataType = (input.CheckedDataType == DataTypes.Float32) ? DataTypes.Float16 : input.CheckedDataType;
        Var var = new Var(input.CheckedType);
        Call input2 = Nncase.IR.K230.F.Tensors.GNNELoad((PrimType)loadDataType, var);
        Call newSplitCall = Nncase.IR.F.Tensors.Split(input2, axis, sections);
        IEnumerable<Call> source = from i in Enumerable.Range(0, sections.Length)
            select Nncase.IR.K230.F.Tensors.GNNEStore(
                input.CheckedDataType,
                Nncase.IR.F.Tensors.GetItem(newSplitCall, i));
        string fullName = NextFullName();
        string moduleKind = ModuleKind;
        Expr[] fields = source.ToArray();
        Call call = new Call(new Fusion(fullName, moduleKind, new Tuple(fields), var), input);
        DumpScope.Current.DumpIR(call, "SplitFusion");
        return call;
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Expr input = (Expr)result["input"];
        Call splitCall = (Call)result["splitCall"];
        Split split = (Split)result["split"];
        int axis = ((TensorConst)result["axis"]).Value.ToScalar<int>();
        int[] sections = ((TensorConst)result["sections"]).Value.ToArray<int>();
        return GetReplace(input, splitCall, split, axis, sections);
    }
}
