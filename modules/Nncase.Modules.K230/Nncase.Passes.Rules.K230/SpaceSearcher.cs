// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Nncase.TIR;

namespace Nncase.Passes.Rules.K230;

internal static class SpaceSearcher
{
    public static int GetBasementSize()
    {
        return 512;
    }

    public static int GetTensorSize(int d0, int d1, int d2, int d3, int bytesPerElement)
    {
        return d0 * d1 * d2 * d3 * bytesPerElement;
    }

    public static int GetWeightSize(int r, int s, int c, int m, int bytesPerElement)
    {
        return r * s * (int)Math.Ceiling(1f * (float)c / (float)GNNEEnv.WAlignNum) * GNNEEnv.WAlignNum * m *
               bytesPerElement;
    }

    public static int GetActSize(int c, int nActParam, int bytesPerElement)
    {
        return c * nActParam * bytesPerElement;
    }

    public static int GetQargSize(int c, int bytesPerElement)
    {
        return c * bytesPerElement;
    }

    public static int GetWQargSize(int m, int bytesPerElement)
    {
        return (int)Math.Ceiling(1f * (float)m / (float)GNNEEnv.PuWidth) * GNNEEnv.PuWidth * bytesPerElement;
    }

    public static int GetInputHeight(int e, int inH, int r, int outH, int uh, int dh, in Padding p)
    {
        List<Segment1D> outputSegments = TileUtilities.SegmentBy(0, e, outH).ToList();
        int maxInputHeight = 0;
        foreach (Segment1D segment in outputSegments)
        {
            int length = TileUtilities.GetInputRowSegment(segment.Start, e, inH, r, uh, dh, in p).Length;
            if (length > maxInputHeight)
            {
                maxInputHeight = length;
            }
        }

        return maxInputHeight;
    }

    public static int GetInputWidth(int f, int inW, int s, int outW, int uw, int dw, in Padding p)
    {
        return GetInputHeight(f, inW, s, outW, uw, dw, in p);
    }

    public static int GetDeconvInputHeight(int e, int inH, int r, int outH, int uh, int dh, Padding p)
    {
        List<Segment1D> outputSegments = TileUtilities.SegmentBy(0, e, outH).ToList();
        int maxInputHeight = 0;
        foreach (Segment1D segment in outputSegments)
        {
            int length = TileUtilities.GetDeconvInputRowSegment(segment.Start, segment.Length, p.Before, inH, uh, r, dh)
                .Length;
            if (length > maxInputHeight)
            {
                maxInputHeight = length;
            }
        }

        return maxInputHeight;
    }

    public static int GetDeconvInputWidth(int f, int inW, int s, int outW, int uw, int dw, Padding p)
    {
        return GetDeconvInputHeight(f, inW, s, outW, uw, dw, p);
    }

    public static AllocateResult TryAllocate(BoxPacker bp)
    {
        ReadOnlyCollection<BoxOnGlb> boxes = bp.Boxes.AsReadOnly();
        int[] v = boxes.Select((BoxOnGlb b) => b.Box[1]).ToArray();
        int[] boxOrderByDepth = SortIndexes(v);
        _ = boxOrderByDepth.Select((int i) => boxes[i]).ToArray();
        int[] sortedIndexes = (from i in SortIndexes(boxes.Select((BoxOnGlb b) => b.Box[0]).ToArray())
            select boxOrderByDepth[i]).ToArray();
        List<int> allocOrder = sortedIndexes.ToList();
        int[] sortedIndexesCopy = sortedIndexes;
        foreach (int boxIndex in sortedIndexesCopy)
        {
            if (bp.Boxes[boxIndex].OwnerName == ItemName.Basement)
            {
                allocOrder.Remove(boxIndex);
                allocOrder.Insert(0, boxIndex);
                break;
            }
        }

        sortedIndexes = allocOrder.ToArray();
        Dictionary<ItemName, MmuItem> mmuItems = new Dictionary<ItemName, MmuItem>();
        bool isOk = true;
        for (int index = 0; index < boxes.Count; index++)
        {
            MmuItem mmuItem = bp.Allocate_item(boxes[sortedIndexes[index]].Box[0], boxes[sortedIndexes[index]].Box[1]);
            if (mmuItem != null && mmuItem.Depth == 0 && mmuItem.Width == 0)
            {
                isOk = false;
                break;
            }

            mmuItems[boxes[sortedIndexes[index]].OwnerName] = mmuItem;
        }

        return new AllocateResult { IsOk = isOk, Items = mmuItems, Boxes = bp.Boxes };
    }

    private static int[] SortIndexes<T>(T[] v)
        where T : unmanaged, IEquatable<T>
    {
        List<int> indexes = Enumerable.Range(0, v.Length).ToList();
        indexes.Sort((int i1, int i2) => Comparer<T>.Default.Compare(v[i2], v[i1]));
        return indexes.ToArray();
    }
}
