using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class FoldReshapeTransposeOfDepthAnything : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; }

    public Expr? GetReplace(Expr input, int[] perm1, int[] perm2, int[] shape1)
    {
        if (perm1.SequenceEqual(new int[4] { 0, 1, 3, 2 }) && perm2.SequenceEqual(new int[4] { 2, 0, 3, 1 }))
        {
            return Nncase.IR.F.Tensors.Reshape(Nncase.IR.F.Tensors.Transpose(Nncase.IR.F.Tensors.Reshape(input,
                new int[4] { shape1[3], shape1[0], shape1[2], shape1[4] }), new int[4] { 1, 2, 0, 3 }), shape1);
        }

        return null;
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Expr input = (Expr)__result["input"];
        int[] perm = ((TensorConst)__result["perm1"]).Value.ToArray<int>();
        int[] perm2 = ((TensorConst)__result["perm2"]).Value.ToArray<int>();
        int[] shape = ((TensorConst)__result["shape1"]).Value.ToArray<int>();
        return GetReplace(input, perm, perm2, shape);
    }

    public FoldReshapeTransposeOfDepthAnything()
    {
        Func<Reshape, bool> condition = (Reshape _) => true;
        Func<Transpose, bool> condition2 = (Transpose _) => true;
        Func<Reshape, bool> condition3 = (Reshape _) => true;
        Func<Transpose, bool> condition4 = (Transpose _) => true;
        Func<Reshape, bool> condition5 = (Reshape _) => true;
        Pattern = Nncase.PatternMatch.F.Tensors.IsReshape("reshape1", "reshape1Call", condition,
                Nncase.PatternMatch.F.Tensors.IsTranspose("transpose1", "transpose1Call", condition2,
                    Nncase.PatternMatch.F.Tensors.IsReshape("reshape2", "reshape2Call", condition3,
                        Nncase.PatternMatch.F.Tensors.IsTranspose("transpose2", "transpose2Call", condition4,
                            Nncase.PatternMatch.F.Tensors.IsReshape("reshape3", "reshape3Call", condition5,
                                Nncase.PatternMatch.Utility.IsWildcard("input"),
                                Nncase.PatternMatch.Utility.IsTensorConst("shape3")),
                            Nncase.PatternMatch.Utility.IsTensorConst("perm2")),
                        Nncase.PatternMatch.Utility.IsTensorConst("shape2")),
                    Nncase.PatternMatch.Utility.IsTensorConst("perm1")),
                Nncase.PatternMatch.Utility.IsTensorConst("shape1")) with
            {
                TypePattern = (TypePatternUtility.HasRank(5) & TypePatternUtility.HasFixedShape())
            };
    }
}
