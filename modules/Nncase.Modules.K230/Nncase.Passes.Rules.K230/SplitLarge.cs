using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.TIR;

namespace Nncase.Passes.Rules.K230;

internal static class SplitLarge
{
	public static Expr? Split(Call call, Expr input, int[] splitShape, Func<(int Dim, int Axis), bool> shouldSplit, Func<int, int> getRound, Func<int, int> getConcatAxis, Func<(int Count, int ChunkSize, int CurrentSize, int Index, int Axis), (int NewBegin, int NewEnd, Padding Pads)> computeNewInputSize, Func<Expr, Padding, int, Expr> callMaker)
	{
		(int, int)[] array = (from pair in splitShape.Select((int dim, int item2) => (dim: dim, axis: item2))
			where pair.dim > 65535
			select pair).ToArray();
		if (array.Length != 1)
		{
			return null;
		}
		if (!shouldSplit(array[0]))
		{
			return null;
		}
		int axis = array[0].Item2;
		if (call.CheckedShape.Rank != 4)
		{
			return null;
		}
		int num = splitShape[axis];
		(int, int) tuple = ComputeChunkSize(num);
		int n = tuple.Item1;
		int item = tuple.Item2;
		int num2 = getRound(axis);
		int w = item / num2 * num2;
		int element = num - w * (n - 1);
		Call call2 = Tensors.Concat(new Nncase.IR.Tuple((from s in Enumerable.Range(0, n - 1)
			select w).Append(element).Select(delegate(int sliceW, int i)
		{
			int[] inShape = input.CheckedShape.ToValueArray();
			(int, int, Padding) tuple2 = computeNewInputSize((n, w, sliceW, i, axis));
			int newW = tuple2.Item1;
			int newWEnd = tuple2.Item2;
			Padding item2 = tuple2.Item3;
			int[] array2 = (from num3 in Enumerable.Range(0, 4)
				select (num3 == axis) ? newW : 0).ToArray();
			int[] array3 = (from num3 in Enumerable.Range(0, 4)
				select (num3 != axis) ? inShape[num3] : newWEnd).ToArray();
			Expr expr = Tensors.Slice(input, array2, array3, 4);
			if (input is Marker marker)
			{
				expr = marker.With(null, expr);
			}
			Expr expr2 = callMaker(expr, item2, axis);
			if (call.Users.Count != 0 && call.Users.First() is Marker marker2)
			{
				expr2 = marker2.With(null, expr2);
			}
			return expr2;
		}).ToArray()), getConcatAxis(axis));
		if (!call2.CheckedShape.ToValueArray().SequenceEqual(call.CheckedShape.ToValueArray()))
		{
			throw new InvalidOperationException("SplitLargeCall result shape is not same as origin call shape");
		}
		return call2;
	}

	private static (int N, int ChunkSize) ComputeChunkSize(int dim)
	{
		int num = 1;
		int i;
		for (i = 2; !(System.Math.Ceiling((float)dim / (float)i) < 65535.0); i++)
		{
		}
		num = (int)System.Math.Ceiling((float)dim / (float)i);
		return (N: i, ChunkSize: num);
	}
}
