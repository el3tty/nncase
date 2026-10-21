// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.TIR;

namespace Nncase.Passes.Rules.K230;

internal static class SplitLarge
{
    public static Expr? Split(Call call, Expr input, int[] splitShape, Func<(int Dim, int Axis), bool> shouldSplit,
        Func<int, int> getRound, Func<int, int> getConcatAxis,
        Func<(int Count, int ChunkSize, int CurrentSize, int Index, int Axis), (int NewBegin, int NewEnd, Padding Pads)>
            computeNewInputSize, Func<Expr, Padding, int, Expr> callMaker)
    {
        (int, int)[] splitAxes = (from pair in splitShape.Select((int dim, int item2) => (dim: dim, axis: item2))
            where pair.dim > 65535
            select pair).ToArray();
        if (splitAxes.Length != 1)
        {
            return null;
        }

        if (!shouldSplit(splitAxes[0]))
        {
            return null;
        }

        int axis = splitAxes[0].Item2;
        if (call.CheckedShape.Rank != 4)
        {
            return null;
        }

        int dimSize = splitShape[axis];
        (int, int) chunkInfo = ComputeChunkSize(dimSize);
        int n = chunkInfo.Item1;
        int chunkSize = chunkInfo.Item2;
        int roundTo = getRound(axis);
        int w = chunkSize / roundTo * roundTo;
        int element = dimSize - w * (n - 1);
        Call call2 = Tensors.Concat(new Nncase.IR.Tuple((from s in Enumerable.Range(0, n - 1)
            select w).Append(element).Select(delegate(int sliceW, int i)
        {
            int[] inShape = input.CheckedShape.ToValueArray();
            (int, int, Padding) newInputInfo = computeNewInputSize((n, w, sliceW, i, axis));
            int newW = newInputInfo.Item1;
            int newWEnd = newInputInfo.Item2;
            Padding newPadding = newInputInfo.Item3;
            int[] sliceBegin = (from num3 in Enumerable.Range(0, 4)
                select (num3 == axis) ? newW : 0).ToArray();
            int[] sliceSize = (from num3 in Enumerable.Range(0, 4)
                select (num3 != axis) ? inShape[num3] : newWEnd).ToArray();
            Expr expr = Tensors.Slice(input, sliceBegin, sliceSize, 4);
            if (input is Marker marker)
            {
                expr = marker.With(null, expr);
            }

            Expr expr2 = callMaker(expr, newPadding, axis);
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
        int chunkSize = 1;
        int i;
        for (i = 2; !(System.Math.Ceiling((float)dim / (float)i) < 65535.0); i++)
        {
        }

        chunkSize = (int)System.Math.Ceiling((float)dim / (float)i);
        return (N: i, ChunkSize: chunkSize);
    }
}
