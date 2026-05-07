using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.NN;
using Nncase.PatternMatch;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class SplitLargeGlobalReduceWindow2D : RewriteRule<Pattern>
{
	public override Pattern Pattern
	{
		get
		{
			OpPattern<ReduceWindow2D> targetPattern = Nncase.PatternMatch.Utility.IsOp<ReduceWindow2D>("reduceWindow2D");
			(ParameterInfo, Pattern)[] inputPatterns = new(ParameterInfo, Pattern)[8]
			{
				(ReduceWindow2D.Input, Nncase.PatternMatch.Utility.IsWildcard("input")with
				{
					TypePattern = (TypePatternUtility.HasFixedShape() & TypePatternUtility.HasRank(4))
				}),
				(ReduceWindow2D.InitValue, Nncase.PatternMatch.Utility.IsTensorConst("initValue")),
				(ReduceWindow2D.Filter, Nncase.PatternMatch.Utility.IsTensorConst("filter")),
				(ReduceWindow2D.Stride, Nncase.PatternMatch.Utility.IsTensorConst("strides")),
				(ReduceWindow2D.Padding, Nncase.PatternMatch.Utility.IsTensorConst("padding")),
				(ReduceWindow2D.Dilation, Nncase.PatternMatch.Utility.IsTensorConst("dilation")),
				(ReduceWindow2D.CeilMode, Nncase.PatternMatch.Utility.IsTensorConst("ceilMode")),
				(ReduceWindow2D.CountIncludePad, Nncase.PatternMatch.Utility.IsTensorConst("countIncludePad"))
			};
			return Nncase.PatternMatch.Utility.IsCallSpecific("call", targetPattern, inputPatterns)with
			{
				TypePattern = (TypePatternUtility.HasFixedShape() & TypePatternUtility.HasRank(4))
			};
		}
	}

	public Expr? GetReplace(Call call, ReduceWindow2D reduceWindow2D, Expr initValue, Expr input, int[] filter, int[] padding)
	{
		int[] array = input.CheckedShape.ToValueArray();
		int[] array2 = call.CheckedShape.ToValueArray();
		int num = array2[2];
		int num2 = array2[3];
		if (array[^2] == filter[0] && array[^1] == filter[1] && padding.All((int p) => p == 0) && num == 1 && num2 == 1 && filter[0] >= 320 && filter[1] >= 320)
		{
			int num3 = ((array[^2] < 640) ? 4 : 10);
			int num4 = array[^2] / num3;
			Expr[] array3 = new Expr[num3];
			Expr[] array4 = new Expr[num3];
			for (int num5 = 0; num5 < num3; num5++)
			{
				array3[num5] = Tensors.Slice(input, new int[1] { num5 * num4 }, new int[1] { num5 * num4 + num4 }, new int[1] { 2 }, new int[1] { 1 });
				array4[num5] = NN.ReduceWindow2D(reduceWindow2D.ReduceOp, array3[num5], initValue, new int[2]
				{
					num4,
					filter[1]
				}, new int[2] { 1, 1 }, new int[2, 2], new int[2] { 1, 1 }, false, false);
			}
			Call input2 = Tensors.Concat(new Tuple(array4), 2);
			return NN.ReduceWindow2D(reduceWindow2D.ReduceOp, input2, initValue, new int[2] { num3, 1 }, new int[2] { 1, 1 }, new int[2, 2], new int[2] { 1, 1 }, false, false);
		}
		return null;
	}

	public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
	{
		Call call = (Call)__result["call"];
		ReduceWindow2D reduceWindow2D = (ReduceWindow2D)__result["reduceWindow2D"];
		Expr initValue = (Expr)__result["initValue"];
		Expr input = (Expr)__result["input"];
		int[] filter = ((TensorConst)__result["filter"]).Value.ToArray<int>();
		int[] padding = ((TensorConst)__result["padding"]).Value.ToArray<int>();
		return GetReplace(call, reduceWindow2D, initValue, input, filter, padding);
	}
}
