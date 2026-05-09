using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230.F;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class OptCast : IRewriteRule
{
    public IPattern Pattern { get; } = Nncase.PatternMatch.F.Tensors.IsCast("cast", "call",
        (Cast cast) => cast.NewType == DataTypes.Float16 || cast.NewType == DataTypes.Float32,
        Nncase.PatternMatch.Utility.IsWildcard("input")with
        {
            TypePattern = (TypePatternUtility.HasFixedShape() &
                           TypePatternUtility.HasRank((int r) => r <= 4, "Only support rank <= 4"))
        });

    private Expr? GetReplace(Cast cast, Call call, Expr input)
    {
        if (input.CheckedDataType != DataTypes.Float16 && input.CheckedDataType != DataTypes.Float32)
        {
            return null;
        }

        if (input.CheckedShape.ToValueArray().Any((int i) => i > 65535))
        {
            return null;
        }

        int[] array = new int[4] { 1, 1, 1, 1 };
        Array.Copy(input.CheckedShape.ToValueArray(), 0, array, array.Length - input.CheckedShape.Count,
            input.CheckedShape.Count);
        PrimType @float = DataTypes.Float16;
        if (input.CheckedShape.Rank < 4)
        {
            Call input2 = Nncase.IR.F.Tensors.Reshape(input, array);
            return Nncase.IR.F.Tensors.Reshape(
                Nncase.IR.K230.F.Tensors.GNNEStore(call.CheckedDataType,
                    Nncase.IR.K230.F.Tensors.GNNELoad(@float, input2)), call.CheckedShape);
        }

        return Nncase.IR.K230.F.Tensors.GNNEStore(call.CheckedDataType,
            Nncase.IR.K230.F.Tensors.GNNELoad(@float, input));
    }

    public Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Cast cast = (Cast)__result["cast"];
        Call call = (Call)__result["call"];
        Expr input = (Expr)__result["input"];
        return GetReplace(cast, call, input);
    }
}
