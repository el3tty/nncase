using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class OptSplit : IRewriteRule
{
    public IPattern Pattern { get; } = Nncase.PatternMatch.F.Tensors.IsSplit("split", "splitCall", (Split _) => true,
        Nncase.PatternMatch.Utility.IsWildcard("input")with
        {
            TypePattern = (TypePatternUtility.IsFloat() & TypePatternUtility.HasFixedShape() &
                           TypePatternUtility.HasRank((int r) => r < 4 || r == 5, "Only support rank < 4 or rank == 5"))
        }, Nncase.PatternMatch.Utility.IsTensorConst("axis"), Nncase.PatternMatch.Utility.IsTensorConst("sections"));

    private Expr? GetReplace(Expr input, Call splitCall, int axis, int[] sections)
    {
        Expr[] fields;
        if (input.CheckedShape.Count == 5)
        {
            if (input.CheckedShape[0].FixedValue == 1)
            {
                Call input2 = Nncase.IR.F.Tensors.Squeeze(input, new int[1]);
                Call newSplitCall = Nncase.IR.F.Tensors.Split(input2, axis - 1, sections);
                fields = (from i in Enumerable.Range(0, sections.Length)
                        select Nncase.IR.F.Tensors.Unsqueeze(Nncase.IR.F.Tensors.GetItem(newSplitCall, i), new int[1]))
                    .ToArray();
                return new Tuple(fields);
            }

            return null;
        }

        int extend = 4 - input.CheckedShape.Rank;
        Call input3 = Nncase.IR.F.Tensors.Unsqueeze(input, Enumerable.Range(0, extend).ToArray());
        Call newSplitCall2 = Nncase.IR.F.Tensors.Split(input3, axis + extend, sections);
        fields = (from i in Enumerable.Range(0, sections.Length)
            select Nncase.IR.F.Tensors.Squeeze(Nncase.IR.F.Tensors.GetItem(newSplitCall2, i),
                Enumerable.Range(0, extend).ToArray())).ToArray();
        return new Tuple(fields);
    }

    public Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Expr input = (Expr)__result["input"];
        Call splitCall = (Call)__result["splitCall"];
        int axis = ((TensorConst)__result["axis"]).Value.ToScalar<int>();
        int[] sections = ((TensorConst)__result["sections"]).Value.ToArray<int>();
        return GetReplace(input, splitCall, axis, sections);
    }
}
