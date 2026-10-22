// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230;
using Nncase.TIR;
using Nncase.TIR.Instructions;

namespace Nncase.Passes.Rules.K230;

public static class TileUtilities
{
    public record MfuRepeat(int LeftRepeats, int ScalarRepeat, int SliceLen, int SliceRepeat)
    {
        public int LeftRepeats
        {
            get { return _leftRepeats; }
            set { _leftRepeats = value; }
        }

        public int ScalarRepeat
        {
            get { return _scalarRepeat; }
            set { _scalarRepeat = value; }
        }

        public int SliceLen
        {
            get { return _sliceLen; }
            set { _sliceLen = value; }
        }

        public int SliceRepeat
        {
            get { return _sliceRepeat; }
            set { _sliceRepeat = value; }
        }

        private int _leftRepeats = LeftRepeats;

        private int _scalarRepeat = ScalarRepeat;

        private int _sliceLen = SliceLen;

        private int _sliceRepeat = SliceRepeat;

        public MfuRepeat()
            : this(0, 0, 0, 0)
        {
        }

        [CompilerGenerated]
        public void Deconstruct(out int leftRepeats, out int scalarRepeat, out int sliceLen, out int sliceRepeat)
        {
            leftRepeats = this.LeftRepeats;
            scalarRepeat = this.ScalarRepeat;
            sliceLen = this.SliceLen;
            sliceRepeat = this.SliceRepeat;
        }
    }

    public static void Assert(bool v, [CallerArgumentExpression("v")] string vStr = "",
        [CallerFilePath] string path = "", [CallerLineNumber] int line = 0)
    {
        if (!v)
        {
            throw new InvalidOperationException($"assert({vStr}) error!\n File \"{path}\", line {line} .");
        }
    }

    public static int GetNTcuNeeded(int nOutputRow, int totalNTcu)
    {
        int rowsPerTcu = (int)System.Math.Ceiling(1.0 * (double)nOutputRow / (double)totalNTcu);
        return (int)System.Math.Ceiling(1.0 * (double)nOutputRow / (double)rowsPerTcu);
    }

    public static List<Segment1D> GetSegmentStartEndLength(int start, int chunkSize, int upperBound)
    {
        List<Segment1D> segments = new List<Segment1D>();
        for (int i = start; i < upperBound; i += chunkSize)
        {
            Segment1D segment = new Segment1D(
                i..((i + chunkSize > upperBound) ? upperBound : (i + chunkSize)),
                new Padding(0, 0));
            segments.Add(segment);
        }

        return segments;
    }

    public static IEnumerable<int> StrideRangeBy(int chunck, int upperBound)
    {
        int acc;
        for (acc = chunck; acc < upperBound; acc = System.Math.Min(acc + chunck, upperBound))
        {
            yield return acc;
        }

        yield return acc;
    }

    public static IEnumerable<Segment1D> SegmentBy(int start, int chunkSize, int upperBound)
    {
        int count = (int)System.Math.Ceiling((float)(upperBound - start) / (float)chunkSize);
        for (int i = 0; i < count; i++)
        {
            yield return new Segment1D(
                (start + i * chunkSize)..(start + System.Math.Min((i + 1) * chunkSize, upperBound)), Padding.Zero(), i);
        }
    }

    public static IEnumerable<Segment1D> SegmentBy(Segment1D seg, int nPart)
    {
        return SegmentBy(seg.Start, (int)System.Math.Ceiling(1.0 * (double)seg.Length / (double)nPart), seg.End);
    }

    public static Segment1D GetInputRowSegment(int outputRowStart, int outputRowLength, int h, int r, int u, int d,
        in Padding p)
    {
        int dilatedKernel = (r - 1) * d + 1;
        int start = outputRowStart * u;
        int length = outputRowLength * u + (dilatedKernel - u);
        int end = start + length;
        start -= p.Before;
        end -= p.Before;
        int before = 0;
        int after = 0;
        if (start <= 0 && end <= 0)
        {
            before = end - start;
            after = 0;
            start = 0;
            end = 0;
            length = 0;
        }
        else if (start <= 0 && end >= 0 && end <= h)
        {
            before = -start;
            after = 0;
            start = 0;
            length = end - start;
        }
        else if (start <= 0 && h <= end)
        {
            before = -start;
            after = end - h;
            start = 0;
            end = h;
            length = end - start;
        }
        else if (start >= 0 && end <= h)
        {
            before = 0;
            after = 0;
            length = end - start;
        }
        else if (start >= 0 && start <= h && h <= end)
        {
            before = 0;
            after = end - h;
            end = h;
            length = end - start;
        }
        else if (h <= start)
        {
            before = 0;
            after = end - start;
            start = 0;
            end = 0;
            length = 0;
        }

        start = System.Math.Max(System.Math.Min(start, h - 1), 0);
        end = start + length;
        return new Segment1D(start..end, new Padding(before, after));
    }

    public static Segment1D GetInputColumnSegment(int outputColumnStart, int outputColumnLength, int w, int s, int u,
        int d, in Padding p)
    {
        return GetInputRowSegment(outputColumnStart, outputColumnLength, w, s, u, d, in p);
    }

    public static int GetAlignedNum(int value, int alignmentFactor)
    {
        return (int)System.Math.Ceiling(1.0 * (double)value / (double)alignmentFactor) * alignmentFactor;
    }

    public static Segment1D GetShiftStartEndLength(in Segment1D ori, int referStart)
    {
        Segment1D shifted = new Segment1D(..0, Padding.Zero());
        if (ori.Length > 0)
        {
            int start = ori.Start - referStart;
            int end = ori.End - referStart;
            shifted = shifted with { Range = start..end };
        }

        return shifted;
    }

    public static SegmentND GlbTensorIndexShift(in SegmentND ori, in SegmentND refer)
    {
        return new SegmentND(from it in ori.Zip(refer)
            select GetShiftStartEndLength(it.First, it.Second.Start));
    }

    public static int GetSliceOffsetInTensor(in SegmentND tensor, in SegmentND slice)
    {
        int length = tensor[1].Length;
        int length2 = tensor[2].Length;
        int length3 = tensor[3].Length;
        return (slice[0].Start - tensor[0].Start) * length * length2 * length3 +
               (slice[1].Start - tensor[1].Start) * length2 * length3 + (slice[2].Start - tensor[2].Start) * length3 +
               (slice[3].Start - tensor[3].Start);
    }

    public static SegmentND ShiftInputTensor(in SegmentND x, in SegmentND w, int r, int s, int dilationH, int dilationW)
    {
        List<int> widthShift = Shift1DSegment(x[3].Start, x[3].End, x.PadW.Before, x.PadW.After, w[3].Start, w[3].End, s,
            dilationW);
        List<int> heightShift = Shift1DSegment(x[2].Start, x[2].End, x.PadH.Before, x.PadH.After, w[2].Start, w[2].End, r,
            dilationH);
        Padding padding = new Padding(heightShift[3], heightShift[4]);
        Padding padding2 = new Padding(widthShift[3], widthShift[4]);
        return new SegmentND(x[0], x[1], new Segment1D(heightShift[0] ..heightShift[1], padding),
            new Segment1D(widthShift[0] ..widthShift[1], padding2));
    }

    public static SegmentND ShiftInputTensor(in SegmentND x, in SegmentND w, int r, int s, int strideH, int strideW,
        int dilationH, int dilationW)
    {
        List<int> widthShift = Shift1DSegment(x[3].Start, x[3].End, x.PadW.Before, x.PadW.After, w[3].Start, w[3].End, s,
            strideW, dilationW);
        List<int> heightShift = Shift1DSegment(x[2].Start, x[2].End, x.PadH.Before, x.PadH.After, w[2].Start, w[2].End, r,
            strideH, dilationH);
        Padding padding = new Padding(heightShift[3], heightShift[4]);
        Padding padding2 = new Padding(widthShift[3], widthShift[4]);
        return new SegmentND(x[0], x[1], new Segment1D(heightShift[0] ..heightShift[1], padding),
            new Segment1D(widthShift[0] ..widthShift[1], padding2));
    }

    public static List<SegmentND> ShiftDeconvInoutTensor(SegmentND g2LIf, SegmentND l2GOf, SegmentND l2RW, int strideH,
        int strideW, int dilationH, int dilationW)
    {
        int rowShift = dilationH * l2RW[2].Start - g2LIf.PadH.Before;
        int ofmapRowStart = l2GOf[2].Start - rowShift;
        int ifmapRowStart = g2LIf[2].Start;
        if (rowShift > 0)
        {
            ifmapRowStart = g2LIf[2].Start + (int)System.Math.Ceiling(1.0 * (double)rowShift / (double)strideH);
            ofmapRowStart = l2GOf[2].Start + (int)(System.Math.Ceiling(1.0 * (double)rowShift / (double)strideH) * (double)strideH -
                                          (double)rowShift);
        }

        int colShift = dilationW * l2RW[3].Start - g2LIf.PadW.Before;
        int ofmapColStart = l2GOf[3].Start - colShift;
        int ifmapColStart = g2LIf[3].Start;
        if (colShift > 0)
        {
            ifmapColStart = g2LIf[3].Start + (int)System.Math.Ceiling(1.0 * (double)colShift / (double)strideW);
            ofmapColStart = l2GOf[3].Start + (int)(System.Math.Ceiling(1.0 * (double)colShift / (double)strideW) * (double)strideW -
                                          (double)colShift);
        }

        int ofmapRowLen = l2GOf[2].End - ofmapRowStart;
        int l = g2LIf[2].End - ifmapRowStart;
        int ofmapRowEnd = FillZero(l, strideH) + ofmapRowStart;
        int ifmapRowEnd = g2LIf[2].End;
        if (FillZero(l, strideH) > ofmapRowLen)
        {
            ifmapRowEnd = ifmapRowStart + (int)System.Math.Ceiling(1.0 * (double)ofmapRowLen / (double)strideH);
            ofmapRowEnd = FillZero((int)System.Math.Ceiling(1.0 * (double)ofmapRowLen / (double)strideH), strideH) + ofmapRowStart;
        }

        int ofmapColLen = l2GOf[3].End - ofmapColStart;
        int l2 = g2LIf[3].End - ifmapColStart;
        int ofmapColEnd = ofmapColStart + FillZero(l2, strideW);
        int ifmapColEnd = g2LIf[3].End;
        if (FillZero(l2, strideW) > ofmapColLen)
        {
            ifmapColEnd = ifmapColStart + (int)System.Math.Ceiling(1.0 * (double)ofmapColLen / (double)strideW);
            ofmapColEnd = ofmapColStart + FillZero((int)System.Math.Ceiling(1.0 * (double)ofmapColLen / (double)strideW), strideW);
        }

        SegmentND ifmapSegment = new SegmentND(g2LIf[0], g2LIf[1], new Segment1D(ifmapRowStart..ifmapRowEnd, Padding.Zero()),
            new Segment1D(ifmapColStart..ifmapColEnd, Padding.Zero()));
        SegmentND ofmapSegment = new SegmentND(l2GOf[0], l2GOf[1], new Segment1D(ofmapRowStart..ofmapRowEnd, Padding.Zero()),
            new Segment1D(ofmapColStart..ofmapColEnd, Padding.Zero()));
        if (ifmapSegment[2].Length <= 0)
        {
            ifmapSegment[2] = new Segment1D(g2LIf[2].Start..g2LIf[2].Start, new Padding(1, 0));
            ofmapSegment[2] = new Segment1D(l2GOf[2].Start..(l2GOf[2].Start + 1), Padding.Zero());
        }

        if (ifmapSegment[3].Length <= 0)
        {
            ifmapSegment[3] = new Segment1D(g2LIf[3].Start..g2LIf[3].Start, new Padding(1, 0));
            ofmapSegment[3] = new Segment1D(l2GOf[3].Start..(l2GOf[3].Start + 1), Padding.Zero());
        }

        return new List<SegmentND> { ifmapSegment, ofmapSegment };

        static int FillZero(int num13, int interval)
        {
            return (num13 - 1) * interval + 1;
        }
    }

    public static int GetTcuIndexBits(int tcuIndex)
    {
        Assert(tcuIndex < 4, "tcuIndex < 4",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileHelper\\TileUtilities.cs",
            272);
        return 1 << tcuIndex;
    }

    public static Expr GetTcuIndexBits(Expr tcuIndex)
    {
        return Tensors.Cast(Nncase.IR.F.Math.LeftShift(1u, Tensors.Cast(tcuIndex, DataTypes.UInt32)), DataTypes.Int32);
    }

    public static int GetNTcuIndexBits(int nTcu)
    {
        Assert(nTcu >= 1 && nTcu <= 4, "nTcu >= 1 && nTcu <= 4",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileHelper\\TileUtilities.cs",
            283);
        return (1 << nTcu) - 1;
    }

    public static Expr GetNTcuIndexBits(Expr nTcu)
    {
        return Tensors.Cast(Nncase.IR.F.Math.LeftShift(1u, Tensors.Cast(nTcu, DataTypes.UInt32)), DataTypes.Int32) - 1;
    }

    public static bool IsFirstSlice(SegmentND slice, SegmentND tensor)
    {
        if (slice[0].Start == tensor[0].Start && slice[1].Start == tensor[1].Start &&
            slice[2].Start == tensor[2].Start && slice[3].Start == tensor[3].Start)
        {
            return true;
        }

        return false;
    }

    public static bool IsLastSlice(SegmentND slice, SegmentND tensor)
    {
        if (slice[0].End == tensor[0].End && slice[1].End == tensor[1].End && slice[2].End == tensor[2].End &&
            slice[3].End == tensor[3].End)
        {
            return true;
        }

        return false;
    }

    public static void CheckConvTHeight(int h, int r, int e, int strideH, Padding pH)
    {
        if (pH.Sum() > 0)
        {
            Assert(
                System.Math.Ceiling(1f * (float)e / (float)strideH).Equals(h),
                "System.Math.Ceiling(1.0f * e / strideH).Equals(h)",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileHelper\\TileUtilities.cs",
                322);
        }
        else
        {
            Assert(
                System.Math.Ceiling(1f * (float)(e - (r - 1)) / (float)strideH).Equals(h),
                "System.Math.Ceiling(1.0f * (e - (r - 1)) / strideH).Equals(h)",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileHelper\\TileUtilities.cs",
                326);
        }
    }

    public static void CheckConvTWidth(int w, int s, int f, int strideW, Padding pW)
    {
        CheckConvTHeight(w, s, f, strideW, pW);
    }

    public static void CheckConvTShape(int h, int r, int e, int strideH, Padding pH, int w, int s, int f, int strideW,
        Padding pW)
    {
        CheckConvTHeight(h, r, e, strideH, pH);
        CheckConvTWidth(w, s, f, strideW, pW);
    }

    public static int GetConvTImmediateHeight(int h, int strideH, int r)
    {
        return (h - 1) * strideH + r;
    }

    public static int GetConvTImmediateWidth(int w, int strideW, int s)
    {
        return GetConvTImmediateHeight(w, strideW, s);
    }

    public static int GetConvTPaddedH(int e, int r, int strideH)
    {
        return (int)System.Math.Ceiling(1f * (float)(e - r) / (float)strideH) + 1;
    }

    public static int GetConvTPaddedW(int f, int s, int strideW)
    {
        return GetConvTPaddedH(f, s, strideW);
    }

    public static (int I, int J) GetConvTPadHImmE(int h, int strideH, int r, int e)
    {
        int convTImmediateHeight = GetConvTImmediateHeight(h, strideH, r);
        int paddedH;
        if (convTImmediateHeight < e)
        {
            paddedH = GetConvTPaddedH(e, r, strideH);
            convTImmediateHeight = GetConvTImmediateHeight(paddedH, strideH, r);
        }
        else
        {
            paddedH = h;
        }

        return (I: paddedH, J: convTImmediateHeight);
    }

    public static (int I, int J) GetConvTPadWImmF(int w, int strideW, int s, int f)
    {
        return GetConvTPadHImmE(w, strideW, s, f);
    }

    public static Padding CalPadTopBottom(int paddedH, int h)
    {
        Assert(paddedH >= h, "paddedH >= h",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileHelper\\TileUtilities.cs",
            387);
        int before = (paddedH - h) / 2;
        int after = (int)System.Math.Ceiling((double)(paddedH - h) / 2.0);
        return new Padding(before, after);
    }

    public static Padding CalPadLeftRight(int paddedW, int w)
    {
        return CalPadTopBottom(paddedW, w);
    }

    public static (int I, int J) GetStartEndOfMidCrop(int immLen, int destLen)
    {
        Assert(immLen >= destLen, "immLen >= destLen",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileHelper\\TileUtilities.cs",
            400);
        int cropStart = (immLen - destLen) / 2;
        int cropEnd = immLen - (int)System.Math.Ceiling((float)(immLen - destLen) / 2f);
        return (I: cropStart, J: cropEnd);
    }

    public static (int I, int J) GetConvTOutputStartEnd(int oriInputLen, int paddedInputLen, int immLen, int destLen,
        Padding pad = null, int outputPadding = 0)
    {
        Assert(paddedInputLen >= oriInputLen, "paddedInputLen >= oriInputLen",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileHelper\\TileUtilities.cs",
            408);
        if (outputPadding == 0)
        {
            if (oriInputLen == paddedInputLen)
            {
                return GetStartEndOfMidCrop(immLen, destLen);
            }

            return (I: 0, J: destLen);
        }

        Assert(immLen + outputPadding == destLen + pad.Sum(), "immLen + outputPadding == destLen + pad.Sum()",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileHelper\\TileUtilities.cs",
            419);
        return (I: pad.Before, J: pad.Before + destLen);
    }

    public static Segment1D GetVnocOutputRowSegment(int inputRowStart, int inputRowLength, int r, int strideH,
        int padTop)
    {
        int paddedStart = inputRowStart + padTop;
        int outStart = paddedStart * strideH;
        int outEnd = (paddedStart + (inputRowLength - 1)) * strideH + (r - 1) + 1;
        return new Segment1D(outStart..outEnd, Padding.Zero());
    }

    public static Segment1D GetVnocOutputRowSegmentGivenRi(int inputRowStart, int inputRowLength, int ri, int strideH,
        int padTop, int padBottom = 0, int h = 0)
    {
        int paddedStart = (inputRowLength == 0) ? (h + padTop) : (inputRowStart + padTop);
        int outStart = paddedStart * strideH + ri;
        int outEnd = (paddedStart + (inputRowLength - 1) + padBottom) * strideH + ri + 1;
        System.Math.Max(outEnd - outStart, 0);
        return new Segment1D(outStart..outEnd, new Padding(0, 0));
    }

    public static Segment1D GetVnocOutputColumnSegment(int inputColumnStart, int inputColumnLength, int s, int strideW,
        int padLeft)
    {
        return GetVnocOutputRowSegment(inputColumnStart, inputColumnLength, s, strideW, padLeft);
    }

    public static Segment1D GetVnocInputRowSegment(int outputRowStart, int outputRowLength, int ddrPadTop,
        int ddrPadBot, int h, int r, int strideH)
    {
        int rowStart = h - 1;
        int rowEnd = 0;
        for (int i = 0; i < r; i++)
        {
            Segment1D vnocInputRowSegmentGivenRi = GetVnocInputRowSegmentGivenRi(outputRowStart, outputRowLength,
                ddrPadTop, ddrPadBot, h, i, strideH);
            if (vnocInputRowSegmentGivenRi.Length > 0)
            {
                rowStart = System.Math.Min(rowStart, vnocInputRowSegmentGivenRi.Start);
                rowEnd = System.Math.Max(rowEnd, vnocInputRowSegmentGivenRi.End);
            }
        }

        return new Segment1D(rowStart..rowEnd, Padding.Zero());
    }

    public static Segment1D GetVnocInputColSegment(int outputColStart, int outputColLength, int ddrPadLeft,
        int ddrPadRight, int w, int s, int strideW)
    {
        return GetVnocInputRowSegment(outputColStart, outputColLength, ddrPadLeft, ddrPadRight, w, s, strideW);
    }

    public static Segment1D GetVnocInputRowSegmentGivenRi(int outputRowStart, int outputRowLength, int ddrPadTop,
        int ddrPadBot, int h, int ri, int strideH)
    {
        int rowStart = System.Math.Max(0, (int)System.Math.Ceiling(1f * (float)(outputRowStart - ri) / (float)strideH));
        int rowEnd =
            System.Math.Min(h - 1 + ddrPadTop + ddrPadBot, (outputRowStart + outputRowLength - 1 - ri) / strideH) + 1;
        rowStart -= ddrPadTop;
        rowEnd -= ddrPadTop;
        int topPad = 0;
        int bottomPad = 0;
        if (rowStart < 0)
        {
            topPad = System.Math.Abs(rowStart);
            rowStart = 0;
        }
        else if (rowStart > h)
        {
            topPad = 0;
            rowStart = h;
        }
        else
        {
            topPad = 0;
        }

        if (rowEnd > h)
        {
            bottomPad = rowEnd - h;
            rowEnd = h;
        }
        else if (rowEnd < 0)
        {
            bottomPad = 0;
            rowEnd = 0;
        }
        else
        {
            bottomPad = 0;
        }

        Assert(rowEnd - rowStart >= 0, "inputRowLength >= 0",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileHelper\\TileUtilities.cs",
            514);
        rowStart = System.Math.Min(h - 1, rowStart);
        return new Segment1D(rowStart..rowEnd, new Padding(topPad, bottomPad));
    }

    public static Segment1D GetVnocInputColumnSegmentGivenSi(int outputColumnStart, int outputColumnLength, int padLeft,
        int padRight, int w, int si, int strideW)
    {
        return GetVnocInputRowSegmentGivenRi(outputColumnStart, outputColumnLength, padLeft, padRight, w, si, strideW);
    }

    public static Segment1D GetDeconvInputRowSegment(int es, int el, int pt, int h, int uh, int r, int dh)
    {
        int start = es - pt;
        int end = start + dh * (r - 1) + el;
        pt = 0;
        int padAfter = 0;
        if (start < 0)
        {
            pt = -start;
            start = 0;
        }
        else
        {
            pt = (int)System.Math.Ceiling(1f * (float)start / (float)uh) * uh - start;
            start = (int)System.Math.Ceiling(1f * (float)start / (float)uh);
        }

        end = (int)System.Math.Ceiling(1f * (float)end / (float)uh);
        if (end > h)
        {
            end = h;
        }

        int length = end - start;
        padAfter = el - (uh * (length - 1) + 1) + dh * (r - 1) - pt;
        return new Segment1D(start..end, new Padding(pt, padAfter));
    }

    public static Segment1D GetDeconvInputColSegment(int fs, int fl, int pl, int w, int uw, int s, int dw)
    {
        return GetDeconvInputRowSegment(fs, fl, pl, w, uw, s, dw);
    }

    public static int GetBytesPerElement(DataType type)
    {
        if (type == DataTypes.Int8 || type == DataTypes.UInt8)
        {
            return 1;
        }

        if (type == DataTypes.BFloat16 || type == DataTypes.Float16 || type == DataTypes.Int16 ||
            type == DataTypes.UInt16)
        {
            return 2;
        }

        if (type == DataTypes.Float32 || type == DataTypes.UInt32 || type == DataTypes.Int32)
        {
            return 4;
        }

        throw new ArgumentOutOfRangeException(type.GetDisplayName());
    }

    public static MfuRepeat GetMfuRepeat(int[] srcShape, int[] destShape)
    {
        if (srcShape.Sum() == 0)
        {
            return new MfuRepeat(1, 1, 1, 1);
        }

        if ((srcShape[2] != 1 && srcShape[2] != destShape[2]) || (srcShape[3] != 1 && srcShape[3] != destShape[3]))
        {
            return new MfuRepeat(1, System.Math.Max(destShape[3] / srcShape[3], 1), srcShape[3],
                System.Math.Max(destShape[2] / srcShape[2], 1));
        }

        int[] alignedShape = AlignBroadcastShape(srcShape, destShape);
        int i = 0;
        List<int> leftDims = new List<int>();
        for (; i < alignedShape.Length && (alignedShape[i] != destShape[i] || alignedShape[i] == 1); i++)
        {
            if (alignedShape[i] == 1)
            {
                leftDims.Add(destShape[i]);
                continue;
            }

            throw new NotSupportedException("Unexpected broadcast");
        }

        for (; i < alignedShape.Length && alignedShape[i] == destShape[i]; i++)
        {
        }

        List<int> sliceDims = new List<int>();
        for (; i < alignedShape.Length && alignedShape[i] != destShape[i]; i++)
        {
            if (alignedShape[i] == 1)
            {
                sliceDims.Add(destShape[i]);
                continue;
            }

            throw new NotSupportedException("Unexpected broadcast");
        }

        int sliceLen = 1;
        for (; i < alignedShape.Length; i++)
        {
            if (alignedShape[i] == destShape[i])
            {
                sliceLen *= alignedShape[i];
                continue;
            }

            if (alignedShape[i] == 1)
            {
                break;
            }

            throw new NotSupportedException("Unexpected broadcast");
        }

        List<int> scalarDims = new List<int>();
        for (; i < alignedShape.Length; i++)
        {
            if (alignedShape[i] == 1)
            {
                scalarDims.Add(destShape[i]);
                continue;
            }

            throw new NotSupportedException("Unexpected broadcast");
        }

        MfuRepeat mfuRepeat = new MfuRepeat();
        mfuRepeat.LeftRepeats = (int)((leftDims.Count == 0) ? 1 : TensorUtilities.GetProduct(leftDims.ToArray()));
        mfuRepeat.ScalarRepeat = (int)TensorUtilities.GetProduct(scalarDims.ToArray());
        mfuRepeat.SliceLen = sliceLen;
        mfuRepeat.SliceRepeat = (int)TensorUtilities.GetProduct(sliceDims.ToArray());
        if (mfuRepeat.SliceLen == 1)
        {
            if (srcShape[^1] == 1 && srcShape[^2] == 1)
            {
                if (mfuRepeat.LeftRepeats == 1 && (double)mfuRepeat.SliceRepeat >= System.Math.Pow(2.0, 16.0))
                {
                    mfuRepeat.ScalarRepeat = destShape[^1];
                    mfuRepeat.SliceRepeat /= destShape[^1];
                    return mfuRepeat;
                }

                if ((double)mfuRepeat.LeftRepeats >= System.Math.Pow(2.0, 16.0))
                {
                    mfuRepeat.ScalarRepeat = destShape[^1];
                    mfuRepeat.SliceRepeat = mfuRepeat.LeftRepeats / destShape[^1];
                    mfuRepeat.LeftRepeats = 1;
                    return mfuRepeat;
                }
            }

            if ((double)(mfuRepeat.ScalarRepeat * mfuRepeat.SliceRepeat) < System.Math.Pow(2.0, 14.0))
            {
                mfuRepeat.ScalarRepeat *= mfuRepeat.SliceRepeat;
                mfuRepeat.SliceRepeat = 1;
            }
        }

        if ((object)mfuRepeat != null && mfuRepeat.SliceRepeat == 1 && mfuRepeat.SliceLen == 1)
        {
            int dimIndex = alignedShape.Length - 1;
            while (dimIndex >= 0 && (double)(mfuRepeat.SliceLen * alignedShape[dimIndex]) < System.Math.Pow(2.0, 16.0))
            {
                mfuRepeat.SliceLen *= alignedShape[dimIndex];
                dimIndex--;
            }
        }

        return mfuRepeat;
    }

    public static Expr Split(Expr length, Expr factor)
    {
        return Tensors.Cast(
            Nncase.IR.F.Math.Ceil(Tensors.Cast(length, DataTypes.Float32) / Tensors.Cast(factor, DataTypes.Float32)),
            DataTypes.Int32);
    }

    public static Expr SplitTimes(Expr length, Expr chunck)
    {
        return Tensors.Cast(
            Nncase.IR.F.Math.Ceil(Tensors.Cast(length, DataTypes.Float32) / Tensors.Cast(chunck, DataTypes.Float32)),
            DataTypes.Int32);
    }

    public static void Ai2dUtilKpuUpdateStaticParam(Call ld, Call st, Ai2dResize resize, ref Ai2dConfig config)
    {
        int[] srcShape = ld.CheckedShape.ToValueArray();
        int[] dstShape = st.CheckedShape.ToValueArray();
        DataType dataType = ld.CheckedDataType;
        config.CscEn = 0u;
        config.CmdId = 1u;
        config.SrcFormat = (dataType == DataTypes.Int16) ? 5u : 3u;
        config.DstFormat = config.SrcFormat;
        config.CrcInd = 0u;
        config.DstInd = config.CrcInd;
        config.Shift = 0u;
        config.Sign = (!(ld.CheckedDataType == DataTypes.UInt8)) ? 1u : 0u;
        config.SrcCh0WidthLayout = (uint)((dataType == DataTypes.Int16) ? (srcShape[3] * 2) : srcShape[3]);
        config.SrcCh1WidthLayout = config.SrcCh0WidthLayout;
        config.SrcCh2WidthLayout = config.SrcCh0WidthLayout;
        config.SrcCh3WidthLayout = config.SrcCh0WidthLayout;
        config.DstCh0WidthLayout = (uint)((dataType == DataTypes.Int16) ? (dstShape[3] * 2) : dstShape[3]);
        config.DstCh1WidthLayout = config.DstCh0WidthLayout;
        config.DstCh2WidthLayout = config.DstCh0WidthLayout;
        config.DstCh3WidthLayout = config.DstCh0WidthLayout;
    }

    public static void Ai2dUtilKpuUpdateResizeParam(Call ld, Call st, Ai2dResize resize, ref Ai2dConfig config)
    {
        config.Interpolation = (resize.ResizeMethod != MFU_CROP_RESIZE.NEAREST) ? 1u : 0u;
        if (config.Interpolation == 0 && !resize.AlignCorners)
        {
            config.CordRound = 2u;
        }

        int[] srcShape = ld.CheckedShape.ToValueArray();
        int[] dstShape = st.CheckedShape.ToValueArray();
        float scaleX = 1f * (float)srcShape[3] / (float)dstShape[3];
        float scaleY = 1f * (float)srcShape[2] / (float)dstShape[2];
        float offsetX = 0f;
        float offsetY = 0f;
        if (config.Interpolation == 0)
        {
            if (resize.HalfPixelCenters)
            {
                offsetX = 0.5f * scaleX;
                offsetY = 0.5f * scaleY;
            }
            else if (resize.AlignCorners)
            {
                if (dstShape[3] > 1)
                {
                    scaleX = 1f * (float)(srcShape[3] - 1) / (float)(dstShape[3] - 1);
                }

                if (dstShape[2] > 1)
                {
                    scaleY = 1f * (float)(srcShape[2] - 1) / (float)(dstShape[2] - 1);
                }
            }
        }
        else if (resize.HalfPixelCenters)
        {
            offsetX = 0.5f * scaleX - 0.5f;
            offsetY = 0.5f * scaleY - 0.5f;
        }
        else if (resize.AlignCorners)
        {
            if (dstShape[3] > 1)
            {
                scaleX = 1f * (float)(srcShape[3] - 1) / (float)(dstShape[3] - 1);
            }

            if (dstShape[2] > 1)
            {
                scaleY = 1f * (float)(srcShape[2] - 1) / (float)(dstShape[2] - 1);
            }
        }

        scaleX *= 1024f;
        scaleY *= 1024f;
        offsetX *= 1024f;
        offsetY *= 1024f;
        FP32 fP = default(FP32);
        fP.U = 0u;
        config.M0 = new FP32 { F = scaleX }.U;
        config.M2 = new FP32 { F = offsetX }.U;
        config.M4 = new FP32 { F = scaleY }.U;
        config.M5 = new FP32 { F = offsetY }.U;
        config.BoundInd = 1u;
    }

    public static void Ai2dUtilKpuUpdateDynamicParam(TiledGlb glb, ref Ai2dConfig config, SegmentND ifmap_sram,
        SegmentND ofmap_sram, int i_pp, SegmentND ifmap, SegmentND ofmap, float offset_M2, float offset_M5,
        DataType input_type, DataType output_type, int offset_if = 0, int offset_of = 0)
    {
        config.Channel = (uint)ifmap_sram[1].Length;
        config.DstChannel = config.Channel;
        config.SrcHeightShape = (uint)ifmap_sram[2].Length;
        config.SrcWidthShape = (uint)ifmap_sram[3].Length;
        config.DstHeightShape = (uint)ofmap_sram[2].Length;
        config.DstWidthShape = (uint)ofmap_sram[3].Length;
        config.SrcX = (uint)ifmap_sram[3].Start;
        config.SrcY = (uint)ifmap_sram[2].Start;
        config.DstX = (uint)ofmap_sram[3].Start;
        config.DstY = (uint)ofmap_sram[2].Start;
        config.PadT = (uint)ifmap_sram.PadH.Before;
        config.PadB = (uint)ifmap_sram.PadH.After;
        config.PadL = (uint)ifmap_sram.PadW.Before;
        config.PadR = (uint)ifmap_sram.PadW.After;
        int ifmapPlaneBytes = ifmap[2].Length * ifmap[3].Length * GetBytesPerElement(input_type);
        config.SrcCh0Ptr = (uint)(i_pp * glb.GlbMap[ItemName.Ifmap].AllocatedBytes +
                                  (ifmap_sram[0].Start - ifmap[0].Start) * ifmapPlaneBytes * ifmap[1].Length +
                                  (ifmap_sram[1].Start - ifmap[1].Start) * ifmapPlaneBytes + offset_if);
        config.SrcCh0Ptr += (uint)(glb.GlbMap[ItemName.Ifmap].Mmu.Id << 28);
        config.SrcCh1Ptr = (uint)(config.SrcCh0Ptr + ifmapPlaneBytes);
        config.SrcCh2Ptr = (uint)(config.SrcCh1Ptr + ifmapPlaneBytes);
        config.SrcCh3Ptr = (uint)(config.SrcCh2Ptr + ifmapPlaneBytes);
        int ofmapPlaneBytes = ofmap[2].Length * ofmap[3].Length * GetBytesPerElement(output_type);
        config.DstCh0Ptr = (uint)(i_pp * glb.GlbMap[ItemName.Ofmap].AllocatedBytes +
                                  (ofmap_sram[0].Start - ofmap[0].Start) * ofmapPlaneBytes * ofmap[1].Length +
                                  (ofmap_sram[1].Start - ofmap[1].Start) * ofmapPlaneBytes + offset_of);
        config.DstCh0Ptr += (uint)(glb.GlbMap[ItemName.Ofmap].Mmu.Id << 28);
        config.DstCh1Ptr = (uint)(config.DstCh0Ptr + ofmapPlaneBytes);
        config.DstCh2Ptr = (uint)(config.DstCh1Ptr + ofmapPlaneBytes);
        config.DstCh3Ptr = (uint)(config.DstCh2Ptr + ofmapPlaneBytes);
        FP32 fP = new FP32 { U = config.M2 };
        float f = fP.F;
        f += offset_M2 * 1024f;
        fP = new FP32 { F = f };
        config.M2 = fP.U;
        fP = new FP32 { U = config.M5 };
        float f2 = fP.F;
        f2 += offset_M5 * 1024f;
        fP = new FP32 { F = f2 };
        config.M5 = fP.U;
    }

    public static List<float> Ai2dUtilMMulAdd(List<float> mScale, List<float> mBias, List<float> v_i)
    {
        List<float> result = new List<float>();
        result.Add(0f);
        result.Add(0f);
        result[0] = mScale[0] * v_i[0] + mScale[1] * v_i[1] + mBias[0];
        result[1] = mScale[2] * v_i[0] + mScale[3] * v_i[1] + mBias[1];
        return result;
    }

    public static void Ai2dUtilInvM(List<float> mOriScale, List<float> mOriBias, List<float> mInvScale,
        List<float> mInvBias)
    {
        float det = mOriScale[0] * mOriScale[3] - mOriScale[1] * mOriScale[2];
        Assert(det != 0f, "det != 0",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileHelper\\TileUtilities.cs",
            970);
        mInvScale[0] = mOriScale[3] / det;
        mInvScale[3] = mOriScale[0] / det;
        mInvScale[1] = (0f - mOriScale[1]) / det;
        mInvScale[2] = (0f - mOriScale[2]) / det;
        mInvBias[0] = (0f - mInvScale[0]) * mOriBias[0] - mInvScale[1] * mOriBias[1];
        mInvBias[1] = (0f - mInvScale[2]) * mOriBias[0] - mInvScale[3] * mOriBias[1];
    }

    public static bool Ai2dUtilsTryAllocateResizeSram(List<float> mOriScale, List<float> mOriBias, int dst_max_h,
        int dst_max_w, SegmentND ifmap, SegmentND ofmap, Ai2dFormat srcFormat)
    {
        List<Segment1D> heightSegments = GetSegmentStartEndLength(0, dst_max_h, ofmap[2].Length);
        List<Segment1D> widthSegments = GetSegmentStartEndLength(0, dst_max_w, ofmap[3].Length);
        bool fits = true;
        foreach (Segment1D heightSegment in heightSegments)
        {
            foreach (Segment1D widthSegment in widthSegments)
            {
                List<float> topLeft = new List<float> { widthSegment.Start, heightSegment.Start };
                List<float> bottomRight = new List<float> { widthSegment.End - 1, heightSegment.End - 1 };
                List<float> srcTopLeft = Ai2dUtilMMulAdd(mOriScale, mOriBias, topLeft);
                List<float> srcBottomRight = Ai2dUtilMMulAdd(mOriScale, mOriBias, bottomRight);
                int srcX0 = System.Math.Max((int)System.Math.Floor(srcTopLeft[0]), 0);
                int srcY0 = System.Math.Max((int)System.Math.Floor(srcTopLeft[1]), 0);
                if (srcFormat <= Ai2dFormat.YUV420_I420)
                {
                    if (srcX0 % 2 == 1)
                    {
                        srcX0--;
                    }

                    if (srcY0 % 2 == 1)
                    {
                        srcY0--;
                    }
                }

                int srcX1 = System.Math.Min((int)System.Math.Ceiling(srcBottomRight[0]), ifmap[3].Length - 1);
                int srcY1 = System.Math.Min((int)System.Math.Ceiling(srcBottomRight[1]), ifmap[2].Length - 1);
                int srcWidth = srcX1 - srcX0 + 1;
                int srcHeight = srcY1 - srcY0 + 1;
                if (srcFormat <= Ai2dFormat.YUV420_I420)
                {
                    if (srcWidth % 2 == 1)
                    {
                        srcX1 = System.Math.Min((int)System.Math.Ceiling(srcBottomRight[0] + 1f), ofmap[3].Length - 1);
                        srcWidth = srcX1 - srcX0 + 1;
                        if (srcWidth % 2 == 1)
                        {
                            throw new NotSupportedException("YUV420 width error.");
                        }
                    }

                    if (srcHeight % 2 == 1)
                    {
                        srcHeight = System.Math.Min((int)System.Math.Ceiling(srcBottomRight[1] + 1f), ofmap[2].Length - 1) - srcY0 + 1;
                        if (srcHeight % 2 == 1)
                        {
                            throw new NotSupportedException("YUV420 height error.");
                        }
                    }
                }

                fits = srcHeight * srcWidth <= Ai2dSram.SramSize;
                if (!fits)
                {
                    return fits;
                }
            }
        }

        return fits;
    }

    public static void Ai2dUtilResizeSramSearch(Ai2dConfig config, SegmentND ofmap, SegmentND ifmap, List<int> ret)
    {
        List<float> mOriScale = new List<float>
        {
            config.OriginM(config.M0),
            config.OriginM(config.M1),
            config.OriginM(config.M3),
            config.OriginM(config.M4),
        };
        List<float> mOriBias = new List<float> { config.OriginM(config.M2), config.OriginM(config.M5) };
        List<float> invScale = new List<float> { 0f, 0f, 0f, 0f };
        List<float> invBias = new List<float> { 0f, 0f };
        Ai2dUtilInvM(mOriScale, mOriBias, invScale, invBias);
        int maxWidth = Ai2dSram.SramLen;
        int maxHeight = Ai2dSram.SramLen;
        if (ifmap[3].Length <= Ai2dSram.SramLen)
        {
            maxWidth = ifmap[3].Length;
            maxHeight = System.Math.Min(Ai2dSram.SramSize / maxWidth, ifmap[2].Length);
        }
        else if (ifmap[2].Length <= Ai2dSram.SramLen)
        {
            maxHeight = ifmap[2].Length;
            maxWidth = System.Math.Min(Ai2dSram.SramSize / maxHeight, ifmap[3].Length);
        }

        if (maxHeight == ifmap[2].Length && maxWidth == ifmap[3].Length)
        {
            ret[0] = ofmap[2].Length;
            ret[1] = ofmap[3].Length;
        }

        List<float> topLeft = new List<float> { 0f, 0f };
        List<float> bottomRight = new List<float> { (float)maxWidth - 1f, (float)maxHeight - 1f };
        List<float> srcTopLeft = Ai2dUtilMMulAdd(invScale, invBias, topLeft);
        List<float> srcBottomRight = Ai2dUtilMMulAdd(invScale, invBias, bottomRight);
        int dstWidth = (int)System.Math.Floor(srcBottomRight[0]) - (int)System.Math.Ceiling(srcTopLeft[0]) + 1;
        int dstHeight = (int)System.Math.Floor(srcBottomRight[1]) - (int)System.Math.Ceiling(srcTopLeft[1]) + 1;
        if (config.SrcFormat <= 2)
        {
            if (dstHeight % 2 == 1)
            {
                dstHeight--;
            }

            if (dstWidth % 2 == 1)
            {
                dstWidth--;
            }
        }

        bool alternate = false;
        int step = (config.SrcFormat > 2) ? 1 : 2;
        while (dstHeight > 0 && dstWidth > 0)
        {
            Ai2dFormat srcFormat = (Ai2dFormat)config.SrcFormat;
            if (Ai2dUtilsTryAllocateResizeSram(mOriScale, mOriBias, dstHeight, dstWidth, ifmap, ofmap, srcFormat))
            {
                break;
            }

            if (ifmap[3].Length <= Ai2dSram.SramLen)
            {
                dstHeight -= step;
                continue;
            }

            if (ifmap[2].Length <= Ai2dSram.SramLen)
            {
                dstWidth -= step;
                continue;
            }

            if (alternate)
            {
                if (dstWidth - 1 <= 0)
                {
                    dstHeight -= step;
                }
                else
                {
                    dstWidth -= step;
                }
            }
            else if (dstHeight - 1 <= 0)
            {
                dstWidth -= step;
            }
            else
            {
                dstHeight -= step;
            }

            alternate = !alternate;
        }

        ret[0] = dstHeight;
        ret[1] = dstWidth;
    }

    internal static int SearchAxis(GNNEShape shape, int begin, int limit, Func<int, AllocateResult> handleAlloc)
    {
        while (begin < limit)
        {
            if (handleAlloc(begin + 1).IsOk)
            {
                begin++;
                continue;
            }

            return begin;
        }

        return begin;
    }

    internal static int SearchAxis(GNNEShape shapeA, GNNEShape shapeB, int axis, Func<int, AllocateResult> handleAlloc)
    {
        return SearchAxis(shapeA, shapeA[axis], shapeB[axis], handleAlloc);
    }

    internal static Padding[] GetPadsFromConst(Const paddings)
    {
        if (paddings.CheckedShape.Rank != 2)
        {
            throw new InvalidOperationException($"paddings rank must be 2, but get {paddings.CheckedShape.Rank}");
        }

        Tensor pads = ((TensorConst)paddings).Value;
        return (from i in Enumerable.Range(0, pads.Shape[0].FixedValue)
            select new Padding((int)pads[new int[2] { i, 0 }], (int)pads[new int[2] { i, 1 }])).ToArray();
    }

    private static List<int> Shift1DSegment(int inputStart, int inputEnd, int padBefore, int padAfter, int weightStart,
        int weightEnd, int weightLength, int dilation)
    {
        int start = inputStart - padBefore + weightStart * dilation;
        int end = inputEnd + padAfter - (weightLength - weightEnd) * dilation;
        int length = 0;
        int cropBefore = 0;
        int cropAfter = 0;
        if (end <= inputStart)
        {
            cropBefore = end - start;
            cropAfter = 0;
            start = 0;
            end = 0;
            length = 0;
        }
        else if (start <= inputStart && inputStart <= end && end <= inputEnd)
        {
            cropBefore = inputStart - start;
            cropAfter = 0;
            start = inputStart;
            length = end - start;
        }
        else if (start <= inputStart && inputEnd <= end)
        {
            cropBefore = inputStart - start;
            cropAfter = end - inputEnd;
            start = inputStart;
            end = inputEnd;
            length = end - start;
        }
        else if (inputStart <= start && end <= inputEnd)
        {
            cropBefore = 0;
            cropAfter = 0;
            length = end - start;
        }
        else if (inputStart <= start && start <= inputEnd && inputEnd <= end)
        {
            cropBefore = 0;
            cropAfter = end - inputEnd;
            end = inputEnd;
            length = end - start;
        }
        else if (inputEnd <= start)
        {
            cropBefore = 0;
            cropAfter = end - start;
            start = 0;
            end = 0;
            length = 0;
        }
        else
        {
            Assert(v: false, "false",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileHelper\\TileUtilities.cs",
                1277);
        }

        start = System.Math.Max(System.Math.Min(start, inputEnd - 1), 0);
        end = start + length;
        return new List<int>
        {
            start,
            end,
            length,
            cropBefore,
            cropAfter,
        };
    }

    private static List<int> Shift1DSegment(int inputStart, int inputEnd, int padBefore, int padAfter, int weightStart,
        int weightEnd, int kernel, int stride, int dilation)
    {
        int start = inputStart - padBefore + weightStart * dilation;
        int end = inputEnd + padAfter - (kernel - weightEnd) * dilation;
        int lastStart = end - (weightEnd - weightStart - 1) * dilation - 1;
        int strideSteps = 0;
        while (lastStart > inputEnd)
        {
            strideSteps++;
            lastStart -= stride;
        }

        int cropBefore = 0;
        int cropAfter = 0;
        if (start <= end && end <= inputStart)
        {
            cropBefore = end - start;
            cropAfter = 0;
            start = 0;
            end = 0;
        }
        else if (start <= inputStart && inputStart <= end && end <= inputEnd)
        {
            cropBefore = inputStart - start;
            cropAfter = 0;
            start = inputStart;
        }
        else if (start <= inputStart && inputEnd <= end)
        {
            int clampedEnd = System.Math.Max(System.Math.Min(end - stride * strideSteps, inputEnd), inputStart);
            cropBefore = inputStart - start;
            cropAfter = end - clampedEnd;
            start = inputStart;
            end = clampedEnd;
        }
        else if (inputStart <= start && start <= end && end <= inputEnd)
        {
            cropBefore = 0;
            cropAfter = 0;
        }
        else if (inputStart <= start && start <= inputEnd && inputEnd <= end)
        {
            int clampedEnd = System.Math.Max(System.Math.Min(end - stride * strideSteps, inputEnd), inputStart);
            cropBefore = 0;
            cropAfter = end - clampedEnd;
            end = clampedEnd;
        }
        else if (inputEnd <= start && start <= end)
        {
            cropBefore = 0;
            cropAfter = end - start;
            start = 0;
            end = 0;
        }
        else
        {
            Assert(v: false, "false",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileHelper\\TileUtilities.cs",
                1351);
        }

        return new List<int>
        {
            start,
            end,
            end - start,
            cropBefore,
            cropAfter,
        };
    }

    private static int[] AlignBroadcastShape(int[] srcShape, int[] destShape)
    {
        Assert(srcShape.Length == destShape.Length, "srcShape.Length == destShape.Length",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileHelper\\TileUtilities.cs",
            1366);
        int[] nonUnitDims = srcShape.Where((int dim) => dim != 1).ToArray();
        if (nonUnitDims.Length == 0)
        {
            return Enumerable.Repeat(1, destShape.Length).ToArray();
        }

        int firstDim = nonUnitDims[0];
        int matchIndex = 0;
        for (int srcIndex = 0; srcIndex < srcShape.Length; srcIndex++)
        {
            if (firstDim == srcShape[srcIndex])
            {
                matchIndex = srcIndex;
                break;
            }
        }

        int matchedLength = srcShape.Length - matchIndex;
        for (int destOffset = destShape.Length - matchedLength; destOffset >= 0; destOffset--)
        {
            bool isCompatible = true;
            for (int offset = 0; offset < matchedLength; offset++)
            {
                int srcDim = srcShape[matchIndex + offset];
                if (srcDim != 1 && srcDim != destShape[destOffset + offset])
                {
                    isCompatible = false;
                    break;
                }
            }

            if (isCompatible)
            {
                List<int> aligned = Enumerable.Repeat(1, destOffset).ToList();
                for (int copyIndex = 0; copyIndex < matchedLength; copyIndex++)
                {
                    aligned.Add(srcShape[matchIndex + copyIndex]);
                }

                int padCount = destShape.Length - aligned.Count;
                for (int padIndex = 0; padIndex < padCount; padIndex++)
                {
                    aligned.Add(1);
                }

                return aligned.ToArray();
            }
        }

        throw new NotSupportedException(
            $"Unexpected broadcast:  [{string.Join(",", srcShape)}] with [{string.Join(",", destShape)}]");
    }
}
