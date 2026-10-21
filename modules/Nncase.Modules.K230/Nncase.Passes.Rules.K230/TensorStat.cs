// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Rules.K230;

public class TensorStat
{
    public bool IsFirstSlice { get; set; }

    public bool IsLastSlice { get; set; }

    public int SliceIdx { get; set; }

    public TensorStat(bool isFirstSlice, bool isLastSlice, int sliceIdx = 0)
    {
        IsFirstSlice = isFirstSlice;
        IsLastSlice = isLastSlice;
        SliceIdx = sliceIdx;
    }

    public int Stat_cnt()
    {
        int count = 0;
        if (IsFirstSlice)
        {
            count++;
        }

        if (IsLastSlice)
        {
            count++;
        }

        if (IsFirstSlice && IsLastSlice)
        {
            count = 1;
        }

        return count;
    }
}
