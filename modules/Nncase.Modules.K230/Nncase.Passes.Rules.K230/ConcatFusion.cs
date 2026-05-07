using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230.F;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
internal sealed class ConcatFusion : FusionMaker
{
	public override string Name { get; } = "TileConcatCase";

	public override string ModuleKind { get; } = "k230";

	public override Pattern Pattern { get; }

	private Call? GetReplace(IReadOnlyList<Expr> tupleInputs, Call concatCall, Concat concat)
	{
		if (concat.Axis > 0)
		{
			DataType midType = ((concatCall.CheckedDataType == DataTypes.Float32) ? DataTypes.Float16 : concatCall.CheckedDataType);
			Var[] array = (from i in tupleInputs.AsEnumerable()
				select new Var(i.CheckedType)).ToArray();
			Expr[] fields = array.Select((Var i) => Nncase.IR.K230.F.Tensors.GNNELoad((PrimType)midType, i)).ToArray();
			Call input = Nncase.IR.F.Tensors.Concat(new Nncase.IR.Tuple(fields), concat.Axis);
			Call body = Nncase.IR.K230.F.Tensors.GNNEStore(concatCall.CheckedDataType, input);
			return new Call(new Fusion(base.FullName, ModuleKind, body, array), tupleInputs.ToArray());
		}
		return null;
	}

	public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
	{
		IReadOnlyList<Expr> tupleInputs = (IReadOnlyList<Expr>)__result["tupleInputs"];
		Call concatCall = (Call)__result["concatCall"];
		Concat concat = (Concat)__result["concat"];
		return GetReplace(tupleInputs, concatCall, concat);
	}

	public ConcatFusion()
	{
		Func<Concat, bool> condition = (Concat _) => true;
		Func<Pattern> creator = () => Nncase.PatternMatch.Utility.IsWildcard();
		Pattern = Nncase.PatternMatch.F.Tensors.IsConcat("concat", "concatCall", condition, Nncase.PatternMatch.Utility.IsTuple("tuple", Nncase.PatternMatch.Utility.IsVArgsRepeat("tupleInputs", creator))) with
		{
			TypePattern = (TypePatternUtility.IsFloat() & TypePatternUtility.HasRank(4) & TypePatternUtility.HasFixedShape())
		};
	}
}
