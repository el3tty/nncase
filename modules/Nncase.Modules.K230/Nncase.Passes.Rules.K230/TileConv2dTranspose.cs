// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.PatternMatch;
using Nncase.Runtime.K230;
using Nncase.TIR;
using Nncase.TIR.Instructions;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class TileConv2dTranspose : RewriteRule<Pattern>
{
    private struct ImmParam
    {
        public int PadTop;

        public int PadBottom;

        public int PadLeft;

        public int PadRight;

        public int PaddedH;

        public int ImmE;

        public int EStart;

        public int EEnd;

        public int PaddedW;

        public int ImmF;

        public int FStart;

        public int FEnd;
    }

    private static int _count = -1;

    private readonly List<Tuple<SegmentND, TensorStat>>[] _weightRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _ofmapRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _g2LIfRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _g2RWRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _l2GOfRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _g2RWSliceRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private CcrHandler _ccrHandler = new CcrHandler();

    private WeightGroupHandler? _weightGroup;

    private GprHandler _gpr = new GprHandler();

    private SsrHandler _ssr = new SsrHandler();

    private GlbSearchStrategy _strategy;

    private DataType? _inputType;

    private DataType? _outputType;

    private DataType? _weightType;

    private Call? _conv;

    private int _icPerGroup;

    private int _ocPerGroup;

    private int _groupPerPass;

    private Call? _lif;

    private Call? _lw;

    private Call? _lact;

    private Call? _lwQarg;

    private Call? _sof;

    private GNNEShape? _inputShape;

    private GNNEShape? _outputShape;

    private GNNEShape? _weightsShape;

    private Padding? _paddingH;

    private Padding? _paddingW;

    private int _outputPaddingH;

    private int _outputPaddingW;

    private int _strideH;

    private int _strideW;

    private int _dilationH;

    private int _dilationW;

    private int _groups;

    private int _kernelH;

    private int _kernelW;

    private ImmParam _immParam;

    public override Pattern Pattern { get; } = FusionPattern.IsGNNEConv2DTransposeFusion();

    private PrimFunction GetReplace(Call call, Call ld, Call st)
    {
        _count++;
        InitParameters(call, ld, st);
        TileConv2dTransposeGlb glb = SearchGlbParameters();
        int[] inputShape = ld[GNNELoad.Input].CheckedShape.ToValueArray();
        T.CreateBuffer(new TensorType(ld[GNNELoad.Input].CheckedDataType, inputShape), MemoryLocation.Input,
            out Nncase.TIR.Buffer buffer, "var ddrIf");
        T.CreateBuffer(new TensorType(st.CheckedDataType, st.CheckedShape), MemoryLocation.Output,
            out Nncase.TIR.Buffer buffer2, "var ddrOf");
        byte[] weightBytes = ((TensorConst)((Call)call[GNNEConv2DTranspose.Weights])[GNNELoadW.Input]).Value.BytesBuffer
            .ToArray();
        byte[] weightBytesCopy = new byte[weightBytes.Length];
        Array.Copy(weightBytes, weightBytesCopy, weightBytesCopy.Length);
        T.AttachBuffer(
            Const.FromTensor(Tensor.FromBytes(DataTypes.UInt8, weightBytesCopy.ToArray(), new int[1] { weightBytesCopy.Length })),
            out Nncase.TIR.Buffer buffer3, "var ddrW");
        T.AttachBuffer((TensorConst)((Call)call[GNNEConv2DTranspose.WeightsBias])[GNNELoadW.Input],
            out Nncase.TIR.Buffer buffer4, "var ddrWQarg");
        T.AttachBuffer((TensorConst)((Call)call[GNNEConv2DTranspose.Act])[GNNELoadW.Input],
            out Nncase.TIR.Buffer buffer5, "var ddrAct");
        ItemRecStatusInit();
        BuildSchedule(glb, call, ld, st, buffer, buffer2, buffer3, buffer4, buffer5, weightGroupOnly: true,
            _weightGroup);
        List<GnneAction> actions = BuildSchedule(glb, call, ld, st, buffer, buffer2, buffer3, buffer4, buffer5,
            weightGroupOnly: false, _weightGroup);
        Span<byte> bytesBuffer = buffer3.Const().Value.BytesBuffer;
        ArrangeWeights(_weightType, _weightsShape, bytesBuffer, _weightGroup);
        ActionToInstruct actionToInstruct = new ActionToInstruct();
        return T.PrimFunc($"TileConv2dTranspose_{_count}", K230RtModule.Kind, buffer, buffer2)
            .Body(actionToInstruct.Instructions(actions), I.END(GP_REGISTER.x0, 0)).Build();
    }

    private List<GnneAction> BuildSchedule(TiledGlb glb, Call conv, Call ld, Call st, Nncase.TIR.Buffer ddrIf,
        Nncase.TIR.Buffer ddrOf, Nncase.TIR.Buffer ddrW, Nncase.TIR.Buffer ddrWQarg, Nncase.TIR.Buffer ddrAct,
        bool weightGroupOnly, WeightGroupHandler weightGroup)
    {
        List<GnneAction> actions = new List<GnneAction>();
        if (!weightGroupOnly)
        {
            _gpr = new GprHandler(GNNEEnv.GprNum);
            _ssr = new SsrHandler(GNNEEnv.SsrNum);
            _ccrHandler = new CcrHandler();
        }

        GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, glb, _ccrHandler, _gpr, _ssr);
        if (!weightGroupOnly)
        {
            UpdateCcrRecStat();
        }

        if (!weightGroupOnly)
        {
            gnneActionUpdater.UpdateMmuConf();
        }

        int ofmapPp = 0;
        int ifmapPp = -1;
        int weightPp = -1;
        int nPingPongSplit = GNNEEnv.NPingPongSplit;
        List<Segment1D> segmentsN =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[0], _outputShape[0]);
        List<Segment1D> segmentsH =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[2], _immParam.ImmE);
        List<Segment1D> segmentsW =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[3], _immParam.ImmF);
        List<Segment1D> ocSegments = new List<Segment1D>();
        List<Segment1D> icSegments = new List<Segment1D>();
        if (glb.LastOutShape[1] >= _ocPerGroup)
        {
            ocSegments = TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[1], _outputShape[1]);
            icSegments = TileUtilities.GetSegmentStartEndLength(0, glb.GlbMap[ItemName.Ifmap].Dimensions[1], _inputShape[1]);
        }
        else
        {
            for (int i = 0; i < _groups; i++)
            {
                List<Segment1D> groupOcSegments =
                    TileUtilities.GetSegmentStartEndLength(i * _ocPerGroup, glb.LastOutShape[1], (i + 1) * _ocPerGroup);
                List<Segment1D> groupIcSegments = TileUtilities.GetSegmentStartEndLength(i * _icPerGroup,
                    glb.GlbMap[ItemName.Ifmap].Dimensions[1], (i + 1) * _icPerGroup);
                ocSegments.AddRange(groupOcSegments);
                icSegments.AddRange(groupIcSegments);
            }
        }

        if (!weightGroupOnly && (_weightType == DataTypes.UInt8 || _weightType == DataTypes.Int16))
        {
            gnneActionUpdater.UpdateLoadWQarg(_lwQarg, weightGroup, ddrWQarg);
        }

        if (!weightGroupOnly)
        {
            gnneActionUpdater.UpdateLoadAct(_lact, ddrAct);
        }

        Segment1D emptySegment = new Segment1D(..0, new Padding(0, 0));
        List<SegmentND> segmentHistory = new List<SegmentND>();
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        List<SegmentND> ifmapHistory = segmentHistory;
        segmentHistory = new List<SegmentND>();
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        List<SegmentND> weightHistory = segmentHistory;
        int l1Pp = 0;
        foreach (Segment1D segmentN in segmentsN)
        {
            foreach (Segment1D segmentH in segmentsH)
            {
                Segment1D vnocInputRowSegment = TileUtilities.GetVnocInputRowSegment(segmentH.Start, segmentH.Length,
                    _immParam.PadTop, _immParam.PadBottom, _inputShape[2], _weightsShape[2], _strideH);
                foreach (Segment1D segmentW in segmentsW)
                {
                    Segment1D vnocInputColSegment = TileUtilities.GetVnocInputColSegment(segmentW.Start, segmentW.Length,
                        _immParam.PadLeft, _immParam.PadRight, _inputShape[3], _weightsShape[3], _strideW);
                    foreach (Segment1D ocSegment in ocSegments)
                    {
                        SegmentND tensor = new SegmentND(segmentN, ocSegment, segmentH, segmentW);
                        int icGroupStart = ocSegment.Start / _ocPerGroup * _icPerGroup;
                        int icGroupEnd = (ocSegment.End - 1) / _ocPerGroup * _icPerGroup + _icPerGroup;
                        foreach (Segment1D icSegment in icSegments)
                        {
                            if (icSegment.Start < icGroupStart || icSegment.End > icGroupEnd)
                            {
                                continue;
                            }

                            int icStartInGroup = icSegment.Start % _icPerGroup;
                            int icEndInGroup = Math.Min(icStartInGroup + icSegment.Length, _icPerGroup);
                            SegmentND weightSegment = new SegmentND(ocSegment, new Segment1D(icStartInGroup..icEndInGroup, new Padding(0, 0)),
                                new Segment1D(.._weightsShape[2], new Padding(0, 0)),
                                new Segment1D(.._weightsShape[3], new Padding(0, 0)));
                            if (weightHistory[0] == weightSegment)
                            {
                                weightPp = 0;
                            }
                            else if (weightHistory[1] == weightSegment)
                            {
                                weightPp = 1;
                            }
                            else
                            {
                                weightPp = (weightPp + 1) % 2;
                                weightGroup.Current_aligned_offset_init();
                                if (weightGroupOnly)
                                {
                                    _weightRec[weightPp].Add(new Tuple<SegmentND, TensorStat>(weightSegment,
                                        new TensorStat(isFirstSlice: false, isLastSlice: false)));
                                }

                                weightHistory[weightPp] = weightSegment;
                            }

                            SegmentND ifmapSegment =
                                new SegmentND(segmentN, icSegment, vnocInputRowSegment, vnocInputColSegment);
                            if (ifmapHistory[0] == ifmapSegment)
                            {
                                ifmapPp = 0;
                            }
                            else if (ifmapHistory[1] == ifmapSegment)
                            {
                                ifmapPp = 1;
                            }
                            else
                            {
                                ifmapPp = (ifmapPp + 1) % 2;
                                if (!weightGroupOnly && _g2LIfRec[ifmapPp].Count > 0 &&
                                    _g2LIfRec[ifmapPp][0].Item1 == ifmapSegment)
                                {
                                    TensorStat loadStat = _g2LIfRec[ifmapPp][0].Item2;
                                    int ifmapCcrCount = ((loadStat != null && loadStat.IsFirstSlice && loadStat.IsLastSlice) ? 1 : 2);
                                    List<CcrSet> ccrsToSet = new List<CcrSet>
                                    {
                                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap, ifmapPp)),
                                            ifmapCcrCount)
                                    };
                                    List<int> stridesD = new List<int>
                                    {
                                        glb.GlbMap[ItemName.Ifmap].Dimensions[1],
                                        glb.GlbMap[ItemName.Ifmap].Dimensions[2],
                                        glb.GlbMap[ItemName.Ifmap].Dimensions[3]
                                    };
                                    gnneActionUpdater.UpdateLoadIf(ifmapSegment, _lif, ifmapPp, ddrIf, 0, stridesD,
                                        ItemName.Ifmap, ccrsToSet);
                                }

                                ifmapHistory[ifmapPp] = ifmapSegment;
                            }

                            BuildL1Schedule(gnneActionUpdater, glb, ifmapSegment, weightSegment, tensor, ifmapPp, ofmapPp, weightPp,
                                weightGroupOnly, weightGroup, l1Pp, ddrW);
                        }

                        if (!weightGroupOnly)
                        {
                            SegmentND slice = new SegmentND(tensor[0], tensor[1], tensor[2],
                                new Segment1D(
                                    Math.Max(tensor[3].Start, _immParam.FStart)..Math.Min(tensor[3].End,
                                        _immParam.FEnd), new Padding(0, 0)));
                            if (tensor[2].Start == 0)
                            {
                                int eStart = _immParam.EStart;
                                int startRowEnd = _immParam.EStart + (slice[2].End - _immParam.EStart);
                                slice[2] = new Segment1D(eStart..startRowEnd, new Padding(0, 0));
                            }

                            if (slice[2].End == _immParam.ImmE)
                            {
                                int endRowStart = _immParam.EEnd - (_immParam.EEnd - slice[2].Start);
                                int eEnd = _immParam.EEnd;
                                slice[2] = new Segment1D(endRowStart..eEnd, new Padding(0, 0));
                            }

                            SegmentND ofmapStDdr = new SegmentND(slice[0], slice[1],
                                new Segment1D((slice[2].Start - _immParam.EStart)..(slice[2].End - _immParam.EStart),
                                    new Padding(0, 0)),
                                new Segment1D((slice[3].Start - _immParam.FStart)..(slice[3].End - _immParam.FStart),
                                    new Padding(0, 0)));
                            int offset = TileUtilities.GetSliceOffsetInTensor(in tensor, in slice) *
                                         glb.GlbMap[ItemName.Ofmap].DType.SizeInBytes;
                            bool notLastSlice = !_ofmapRec[ofmapPp][0].Item2.IsLastSlice;
                            _ofmapRec[ofmapPp].RemoveAt(0);
                            List<CcrSet> ofmapFakeCcrsToSet = new List<CcrSet>();
                            if ((notLastSlice ? 1 : 0) > (false ? 1 : 0))
                            {
                                ofmapFakeCcrsToSet.Add(new CcrSet(
                                    _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.OfmapFake, ofmapPp)), 1));
                            }

                            List<CcrClr> ccrsToClr = new List<CcrClr>
                            {
                                new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, ofmapPp)))
                            };
                            gnneActionUpdater.UpdateStoreT(tensor, _sof, ofmapPp, ddrOf, offset, ofmapStDdr, ofmapFakeCcrsToSet,
                                ccrsToClr);
                        }
                        else
                        {
                            _ofmapRec[ofmapPp].Add(new Tuple<SegmentND, TensorStat>(tensor,
                                new TensorStat(isFirstSlice: false, isLastSlice: false)));
                        }

                        ofmapPp = (ofmapPp + 1) % nPingPongSplit;
                    }
                }
            }
        }

        TileUtilities.Assert(_ccrHandler.CcrSanityCheck(), "_ccrHandler.CcrSanityCheck()",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileConv2dTranspose.cs",
            291);
        return actions;
    }

    private List<List<Segment1D>> GetL1McSeg(SegmentND ifmap, SegmentND psum, int mInloop, int cInloop)
    {
        int groupCount = Math.Max(ifmap[1].Length / _icPerGroup, 1);
        List<Segment1D> ocSegments = new List<Segment1D>();
        List<Segment1D> icSegments = new List<Segment1D>();
        if (groupCount >= 2 && _groupPerPass < 2)
        {
            for (int i = 0; i < groupCount; i++)
            {
                List<Segment1D> groupOcSegments =
                    TileUtilities.GetSegmentStartEndLength(psum[1].Start + i * _ocPerGroup, mInloop,
                        psum[1].Start + (i + 1) * _ocPerGroup);
                List<Segment1D> groupIcSegments =
                    TileUtilities.GetSegmentStartEndLength(ifmap[1].Start + i * _icPerGroup, cInloop,
                        ifmap[1].Start + (i + 1) * _icPerGroup);
                ocSegments.AddRange(groupOcSegments);
                icSegments.AddRange(groupIcSegments);
            }
        }
        else
        {
            ocSegments = TileUtilities.GetSegmentStartEndLength(psum[1].Start, mInloop, psum[1].End);
            icSegments = TileUtilities.GetSegmentStartEndLength(ifmap[1].Start, cInloop, ifmap[1].End);
        }

        return new List<List<Segment1D>> { ocSegments, icSegments };
    }

    private bool HandleL1Allocate(int h, int w, int e, int f, int psumPingPangSplit, int ifBytesPerElementGlb)
    {
        if (e * f > GNNEEnv.PsumL1ElePerChan / psumPingPangSplit)
        {
            return false;
        }

        return GNNEEnv.PuHeight * h * w * ifBytesPerElementGlb <= GNNEEnv.IfL1Size;
    }

    private List<int> L1Search(TiledGlb glb, SegmentND weight, SegmentND psum)
    {
        int puHeight = Math.Min(GNNEEnv.PuHeight, _groupPerPass * _icPerGroup);
        int puWidth = Math.Min(GNNEEnv.PuWidth, _groupPerPass * _ocPerGroup);
        int ifmapRows = 1;
        int ifmapCols = 1;
        int ofmapRows = 1;
        int ofmapCols = 1;
        ifmapRows = GlbInputHeight(ofmapRows);
        ifmapCols = GlbInputWidth(ofmapCols);
        int bytesPerElement = TileUtilities.GetBytesPerElement(_inputType);
        if (!HandleL1Allocate(ifmapRows, ifmapCols, ofmapRows, ofmapCols, 1, bytesPerElement))
        {
            throw new NotSupportedException("exceeds L1 size");
        }

        while (ofmapCols < psum[3].Length && ofmapRows < psum[2].Length)
        {
            if (ofmapCols <= ofmapRows)
            {
                int nextOfmapCols = ofmapCols + 1;
                int nextIfmapCols = GlbInputWidth(nextOfmapCols);
                if (!HandleL1Allocate(ifmapRows, nextIfmapCols, ofmapRows, nextOfmapCols, 1, bytesPerElement))
                {
                    break;
                }

                ofmapCols = nextOfmapCols;
                ifmapCols = nextIfmapCols;
            }
            else
            {
                int nextOfmapRows = ofmapRows + 1;
                int nextIfmapRows = GlbInputHeight(nextOfmapRows);
                if (!HandleL1Allocate(nextIfmapRows, ifmapCols, nextOfmapRows, ofmapCols, 1, bytesPerElement))
                {
                    break;
                }

                ofmapRows = nextOfmapRows;
                ifmapRows = nextIfmapRows;
            }
        }

        while (ofmapRows < psum[2].Length)
        {
            int nextOfmapRows = ofmapRows + 1;
            int nextIfmapRows = GlbInputHeight(nextOfmapRows);
            if (!HandleL1Allocate(nextIfmapRows, ifmapCols, nextOfmapRows, ofmapCols, 1, bytesPerElement))
            {
                break;
            }

            ofmapRows = nextOfmapRows;
            ifmapRows = nextIfmapRows;
        }

        while (ofmapCols < psum[3].Length)
        {
            int nextOfmapCols = ofmapCols + 1;
            int nextIfmapCols = GlbInputWidth(nextOfmapCols);
            if (!HandleL1Allocate(ifmapRows, nextIfmapCols, ofmapRows, nextOfmapCols, 1, bytesPerElement))
            {
                break;
            }

            ofmapCols = nextOfmapCols;
            ifmapCols = nextIfmapCols;
        }

        return new List<int> { puHeight, puWidth, ofmapRows, ofmapCols };

        int GlbInputHeight(int e)
        {
            return Math.Min(
                (int)Math.Ceiling((decimal)(1f * (float)(e + _weightsShape[2] - _strideH - 1) / (float)_strideH)) + 1,
                _immParam.PaddedH - _immParam.PadTop - _immParam.PadBottom);
        }

        int GlbInputWidth(int f)
        {
            return Math.Min(
                (int)Math.Ceiling((decimal)(1f * (float)(f + _weightsShape[3] - _strideW - 1) / (float)_strideW)) + 1,
                _immParam.PaddedW - _immParam.PadLeft - _immParam.PadRight);
        }
    }

    private void BuildL1Schedule(GnneActionUpdater actionUpdater, TiledGlb glb, SegmentND ifmap, SegmentND weight,
        SegmentND psum, int iPp, int ofPp, int wPp, bool weightGroupOnly, WeightGroupHandler weightGroup, int l1Pp,
        Nncase.TIR.Buffer ddrW)
    {
        List<int> l1Params = L1Search(glb, weight, psum);
        List<Segment1D> psumRowSegments =
            TileUtilities.GetSegmentStartEndLength(psum[2].Start, l1Params[2], psum[2].End);
        List<Segment1D> psumColSegments =
            TileUtilities.GetSegmentStartEndLength(psum[3].Start, l1Params[3], psum[3].End);
        List<Segment1D> psumBatchSegments = TileUtilities.GetSegmentStartEndLength(psum[0].Start, 1, psum[0].End);
        List<List<Segment1D>> l1McSeg = GetL1McSeg(ifmap, psum, l1Params[1], l1Params[0]);
        List<Segment1D> ocSegments = l1McSeg[0];
        List<Segment1D> icSegments = l1McSeg[1];
        int chunkSize = 1;
        int chunkSize2 = 1;
        foreach (Segment1D ocSegment in ocSegments)
        {
            foreach (Segment1D batchSegment in psumBatchSegments)
            {
                foreach (Segment1D rowSegment in psumRowSegments)
                {
                    Segment1D vnocInputRowSegment = TileUtilities.GetVnocInputRowSegment(rowSegment.Start, rowSegment.Length,
                        _immParam.PadTop, _immParam.PadBottom, _inputShape[2], _weightsShape[2], _strideH);
                    foreach (Segment1D colSegment in psumColSegments)
                    {
                        Segment1D vnocInputColSegment = TileUtilities.GetVnocInputColSegment(colSegment.Start, colSegment.Length,
                            _immParam.PadLeft, _immParam.PadRight, _inputShape[3], _weightsShape[3], _strideW);
                        SegmentND convSegment = new SegmentND(batchSegment, ocSegment, rowSegment, colSegment);
                        int icGroupStart = ocSegment.Start / _ocPerGroup * _icPerGroup;
                        int icGroupEnd = (ocSegment.End - 1) / _ocPerGroup * _icPerGroup + _icPerGroup;
                        bool isFirstLoop = true;
                        foreach (Segment1D icSegment in icSegments)
                        {
                            if (icSegment.Start < icGroupStart || icSegment.End > icGroupEnd)
                            {
                                continue;
                            }

                            int icStartInGroup = icSegment.Start % _icPerGroup;
                            Segment1D icSegmentInGroup =
                                new Segment1D(
                                    new System.Range(end: Math.Min(icStartInGroup + icSegment.Length, _icPerGroup), start: icStartInGroup),
                                    new Padding(0, 0));
                            Segment1D ifmapIcSegment = icSegment;
                            List<Segment1D> kernelRowSlices =
                                TileUtilities.GetSegmentStartEndLength(0, chunkSize, _weightsShape[2]);
                            List<Segment1D> kernelColSlices =
                                TileUtilities.GetSegmentStartEndLength(0, chunkSize2, _weightsShape[3]);
                            Segment1D ifmapBatchSegment = batchSegment;
                            Segment1D weightOcSegment = ocSegment;
                            bool hasIfmapLoad = false;
                            SegmentND ifmapLoadSegment = new SegmentND(ifmapBatchSegment, ifmapIcSegment, vnocInputRowSegment,
                                vnocInputColSegment);
                            if (ifmapLoadSegment[0].Length > 0 && ifmapLoadSegment[1].Length > 0 && ifmapLoadSegment[2].Length > 0 &&
                                ifmapLoadSegment[3].Length > 0)
                            {
                                hasIfmapLoad = true;
                                if (!weightGroupOnly)
                                {
                                    int ifmapStatCount = _g2LIfRec[iPp][0].Item2.Stat_cnt();
                                    _g2LIfRec[iPp].RemoveAt(0);
                                    List<CcrClr> ifmapCcrsToClr = new List<CcrClr>();
                                    if (ifmapStatCount > 0)
                                    {
                                        ifmapCcrsToClr.Add(new CcrClr(
                                            _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap, iPp))));
                                    }

                                    actionUpdater.UpdateG2LIf(ifmapLoadSegment, ifmap, _lif, iPp, null, ifmapCcrsToClr);
                                }
                                else
                                {
                                    _g2LIfRec[iPp].Add(new Tuple<SegmentND, TensorStat>(ifmap,
                                        new TensorStat(isFirstSlice: false, isLastSlice: false)));
                                }
                            }

                            foreach (Segment1D kernelRowSlice in kernelRowSlices)
                            {
                                foreach (Segment1D kernelColSlice in kernelColSlices)
                                {
                                    SegmentND weightSegment = new SegmentND(weightOcSegment, icSegmentInGroup, kernelRowSlice, kernelColSlice);
                                    Segment1D vnocInputRowSegmentGivenRi =
                                        TileUtilities.GetVnocInputRowSegmentGivenRi(convSegment[2].Start,
                                            convSegment[2].Length, _immParam.PadTop, _immParam.PadBottom, _inputShape[2],
                                            weightSegment[2].Start, _strideH);
                                    Segment1D vnocInputColumnSegmentGivenSi =
                                        TileUtilities.GetVnocInputColumnSegmentGivenSi(convSegment[3].Start,
                                            convSegment[3].Length, _immParam.PadLeft, _immParam.PadRight, _inputShape[3],
                                            weightSegment[3].Start, _strideW);
                                    SegmentND slice = new SegmentND(ifmapBatchSegment, ifmapIcSegment, vnocInputRowSegmentGivenRi,
                                        vnocInputColumnSegmentGivenSi);
                                    Segment1D vnocOutputRowSegmentGivenRi =
                                        TileUtilities.GetVnocOutputRowSegmentGivenRi(vnocInputRowSegmentGivenRi.Start,
                                            vnocInputRowSegmentGivenRi.Length, weightSegment[2].Start, _strideH,
                                            vnocInputRowSegmentGivenRi.Padding.Before,
                                            vnocInputRowSegmentGivenRi.Padding.After, _inputShape[2]);
                                    Segment1D vnocOutputRowSegmentGivenRi2 =
                                        TileUtilities.GetVnocOutputRowSegmentGivenRi(
                                            vnocInputColumnSegmentGivenSi.Start, vnocInputColumnSegmentGivenSi.Length,
                                            weightSegment[3].Start, _strideW, vnocInputColumnSegmentGivenSi.Padding.Before,
                                            vnocInputColumnSegmentGivenSi.Padding.After, _inputShape[3]);
                                    SegmentND r2LPsum = new SegmentND(ifmapBatchSegment, weightOcSegment,
                                        vnocOutputRowSegmentGivenRi, vnocOutputRowSegmentGivenRi2);
                                    if (vnocOutputRowSegmentGivenRi.Length == 0 ||
                                        vnocOutputRowSegmentGivenRi2.Length == 0)
                                    {
                                        continue;
                                    }

                                    int weightPasses = ((!(_weightType == DataTypes.Int16)) ? 1 : 2);
                                    int inputPasses = ((!(_inputType == DataTypes.Int16)) ? 1 : 2);
                                    for (int i = 0; i < weightPasses; i++)
                                    {
                                        for (int inputPass = 0; inputPass < inputPasses; inputPass++)
                                        {
                                            if (!weightGroupOnly)
                                            {
                                                Tuple<SegmentND, TensorStat> weightRecStat = _weightRec[wPp][0];
                                                Tuple<SegmentND, TensorStat> g2RWRecStat = _g2RWRec[wPp][0];
                                                _g2RWRec[wPp].RemoveAt(0);
                                                if (g2RWRecStat.Item2.IsLastSlice)
                                                {
                                                    _weightRec[wPp].RemoveAt(0);
                                                }

                                                Tuple<SegmentND, TensorStat> g2RWSliceRecStat = _g2RWSliceRec[wPp][0];
                                                _g2RWSliceRec[wPp].RemoveAt(0);
                                                TileUtilities.Assert(
                                                    weightSegment == g2RWSliceRecStat.Item1 && g2RWRecStat.Item1 == weightRecStat.Item1,
                                                    "l2RW == g2RWSliceRecStat.Item1 && g2RWRecStat.Item1 == weightRecStat.Item1",
                                                    "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileConv2dTranspose.cs",
                                                    530);
                                                if (g2RWSliceRecStat.Item2.IsFirstSlice)
                                                {
                                                    int clearWeightFakeCcr = ((!weightRecStat.Item2.IsFirstSlice && g2RWRecStat.Item2.IsFirstSlice)
                                                        ? 1
                                                        : 0);
                                                    List<CcrClr> weightFakeCcrsToClr = new List<CcrClr>();
                                                    if (clearWeightFakeCcr > 0)
                                                    {
                                                        weightFakeCcrsToClr.Add(new CcrClr(
                                                            _ccrHandler.GetCcrItem(
                                                                _ccrHandler.GetName(ItemName.WeightFake, wPp))));
                                                    }

                                                    List<CcrSet> ccrsToSet = new List<CcrSet>
                                                    {
                                                        new CcrSet(
                                                            _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Weight,
                                                                (wPp << 1) + (g2RWSliceRecStat.Item2.SliceIdx & 1))), 1)
                                                    };
                                                    actionUpdater.UpdateLoadW(weightSegment, _lw, weightGroup, wPp, ddrW,
                                                        ccrsToSet, weightFakeCcrsToClr, 0, h2c: false, 0, ItemName.Weight, i);
                                                }

                                                int setWeightFakeCcr = 0;
                                                if (g2RWRecStat.Item2.IsLastSlice && _weightRec[wPp].Count > 0)
                                                {
                                                    setWeightFakeCcr = 1;
                                                }

                                                List<CcrSet> weightCcrsToSet = new List<CcrSet>();
                                                if (setWeightFakeCcr > 0)
                                                {
                                                    weightCcrsToSet.Add(new CcrSet(
                                                        _ccrHandler.GetCcrItem(
                                                            _ccrHandler.GetName(ItemName.WeightFake, wPp)), 1));
                                                }

                                                List<CcrClr> weightCcrsToClr = new List<CcrClr>();
                                                if (g2RWSliceRecStat.Item2.IsFirstSlice)
                                                {
                                                    weightCcrsToClr.Add(new CcrClr(_ccrHandler.GetCcrItem(
                                                        _ccrHandler.GetName(ItemName.Weight,
                                                            (wPp << 1) + (g2RWSliceRecStat.Item2.SliceIdx & 1)))));
                                                }

                                                actionUpdater.UpdateG2RW(weightSegment, weightGroup, _ocPerGroup, _lw, wPp,
                                                    i, weightCcrsToSet, weightCcrsToClr);
                                            }
                                            else
                                            {
                                                weightGroup.UpdateWeightGroup(weightSegment);
                                                _g2RWRec[wPp].Add(new Tuple<SegmentND, TensorStat>(weight,
                                                    new TensorStat(isFirstSlice: false, isLastSlice: false)));
                                                _g2RWSliceRec[wPp].Add(new Tuple<SegmentND, TensorStat>(weightSegment,
                                                    new TensorStat(isFirstSlice: false, isLastSlice: false, -1)));
                                            }

                                            if (!weightGroupOnly)
                                            {
                                                actionUpdater.UpdateL2RIf(slice, ifmapLoadSegment, _strideH, _strideW,
                                                    _icPerGroup, _lif, 0, inputPass,
                                                    ((TensorConst)_conv[GNNEConv2DTranspose.DeqBias]).Value
                                                    .ToScalar<int>());
                                            }

                                            bool releaseIf = false;
                                            int isLastPass;
                                            if (kernelColSlice == kernelColSlices[kernelColSlices.Count - 1])
                                            {
                                                if (kernelRowSlice == kernelRowSlices[kernelRowSlices.Count - 1] &&
                                                    i == weightPasses - 1)
                                                {
                                                    isLastPass = ((inputPass == inputPasses - 1) ? 1 : 0);
                                                    goto SetReleaseIf;
                                                }
                                            }

                                            isLastPass = 0;
                                            goto SetReleaseIf;
                                            SetReleaseIf:
                                            if (((uint)isLastPass & (hasIfmapLoad ? 1u : 0u)) != 0)
                                            {
                                                releaseIf = true;
                                                hasIfmapLoad = false;
                                            }

                                            bool loopStart = false;
                                            if (isFirstLoop && i == 0 && inputPass == 0)
                                            {
                                                loopStart = true;
                                                isFirstLoop = false;
                                            }

                                            bool isLastKernelSlice = icSegmentInGroup.End == _weightsShape[1] &&
                                                         kernelColSlice.End == weight[3].End && kernelRowSlice.End == weight[2].End &&
                                                         i == weightPasses - 1 && inputPass == inputPasses - 1;
                                            if (!weightGroupOnly)
                                            {
                                                int setOfmapCcr = 0;
                                                int clearOfmapCcr = 0;
                                                if (isLastKernelSlice)
                                                {
                                                    Tuple<SegmentND, TensorStat> l2GOfRecStat = _l2GOfRec[ofPp][0];
                                                    _l2GOfRec[ofPp].RemoveAt(0);
                                                    if (l2GOfRecStat.Item2.IsFirstSlice &&
                                                        !_ofmapRec[ofPp][0].Item2.IsFirstSlice)
                                                    {
                                                        clearOfmapCcr = 1;
                                                    }

                                                    if (l2GOfRecStat.Item2.IsLastSlice)
                                                    {
                                                        setOfmapCcr = 1;
                                                    }
                                                }

                                                List<CcrSet> ofmapCcrsToSet = new List<CcrSet>();
                                                if (setOfmapCcr > 0)
                                                {
                                                    ofmapCcrsToSet.Add(new CcrSet(
                                                        _ccrHandler.GetCcrItem(
                                                            _ccrHandler.GetName(ItemName.Ofmap, ofPp)), 1));
                                                }

                                                List<CcrClr> ofmapCcrsToClr = new List<CcrClr>();
                                                if (clearOfmapCcr > 0)
                                                {
                                                    ofmapCcrsToClr.Add(new CcrClr(
                                                        _ccrHandler.GetCcrItem(
                                                            _ccrHandler.GetName(ItemName.OfmapFake, ofPp))));
                                                }

                                                int shift = ((TensorConst)_conv[GNNEConv2DTranspose.ShiftBits]).Value
                                                    .ToScalar<int>();
                                                ACT0_OUTPUT_DEST destTarget = ACT0_OUTPUT_DEST.dm;
                                                actionUpdater.UpdateR2LPsum(shift, r2LPsum, convSegment, psum, ofPp, l1Pp,
                                                    destTarget, releaseIf, Math.Max(i, inputPass),
                                                    TcuComputeMode.TransposeConv2d, loopStart, isLastKernelSlice, _strideH,
                                                    _strideW, _ocPerGroup, _inputType, _weightType, _outputType,
                                                    _lact.CheckedDataType, ofmapCcrsToSet, ofmapCcrsToClr);
                                            }
                                            else if (isLastKernelSlice)
                                            {
                                                _l2GOfRec[ofPp].Add(new Tuple<SegmentND, TensorStat>(psum,
                                                    new TensorStat(isFirstSlice: false, isLastSlice: false)));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private void ArrangeWeights(DataType weightsType, GNNEShape weightsShape, Span<byte> oldWeights,
        WeightGroupHandler weightGroup)
    {
        int bytesPerElement = TileUtilities.GetBytesPerElement(weightsType);
        List<SegmentND> weightSlices = weightGroup.WeightGroupSlice();
        byte[] arranged = new byte[oldWeights.Length];
        int expectedOffset = 0;
        int writeIndex = 0;
        foreach (SegmentND weightSlice in weightSlices)
        {
            TileUtilities.Assert(expectedOffset == weightGroup.WeightGroupOffset(weightSlice) * bytesPerElement,
                "offset == weightGroup.WeightGroupOffset(slice) * bytesPerElement",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileConv2dTranspose.cs",
                652);
            for (int i = 0; i < bytesPerElement; i++)
            {
                for (int j = 0; j < weightSlice[0].Length; j++)
                {
                    for (int k = 0; k < weightSlice[2].Length; k++)
                    {
                        for (int l = 0; l < weightSlice[3].Length; l++)
                        {
                            for (int m = 0; m < weightSlice[1].Length; m++)
                            {
                                int srcIndex = ((((j + weightSlice[0].Start) * weightsShape[1] + m + weightSlice[1].Start) *
                                                weightsShape[2] + k + weightSlice[2].Start) * weightsShape[3] + l +
                                            weightSlice[3].Start) *
                                           bytesPerElement;
                                arranged[writeIndex++] = oldWeights[srcIndex + i];
                                expectedOffset++;
                            }
                        }
                    }
                }
            }
        }

        arranged.CopyTo(oldWeights);
    }

    private void UpdateCcrRecStat()
    {
        List<Tuple<SegmentND, TensorStat>>[] g2LIfRec = _g2LIfRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in g2LIfRec)
        {
            for (int j = 0; j < recList.Count; j++)
            {
                if (j == 0)
                {
                    recList[j].Item2.IsFirstSlice = true;
                }

                if (j == recList.Count - 1)
                {
                    recList[j].Item2.IsLastSlice = true;
                }

                if (j < recList.Count - 1 && recList[j].Item1 != recList[j + 1].Item1)
                {
                    recList[j].Item2.IsLastSlice = true;
                    recList[j + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        g2LIfRec = _g2RWRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in g2LIfRec)
        {
            for (int k = 0; k < recList.Count; k++)
            {
                if (k == 0)
                {
                    recList[k].Item2.IsFirstSlice = true;
                }

                if (k == recList.Count - 1)
                {
                    recList[k].Item2.IsLastSlice = true;
                }

                if (k < recList.Count - 1 && recList[k].Item1 != recList[k + 1].Item1)
                {
                    recList[k].Item2.IsLastSlice = true;
                    recList[k + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        g2LIfRec = _l2GOfRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in g2LIfRec)
        {
            for (int l = 0; l < recList.Count; l++)
            {
                if (l == 0)
                {
                    recList[l].Item2.IsFirstSlice = true;
                }

                if (l == recList.Count - 1)
                {
                    recList[l].Item2.IsLastSlice = true;
                }

                if (l < recList.Count - 1 && recList[l].Item1 != recList[l + 1].Item1)
                {
                    recList[l].Item2.IsLastSlice = true;
                    recList[l + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        g2LIfRec = _weightRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in g2LIfRec)
        {
            if (recList.Count > 0)
            {
                recList[0].Item2.IsFirstSlice = true;
                recList[recList.Count - 1].Item2.IsLastSlice = true;
            }
        }

        g2LIfRec = _ofmapRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in g2LIfRec)
        {
            if (recList.Count > 0)
            {
                recList[0].Item2.IsFirstSlice = true;
                recList[recList.Count - 1].Item2.IsLastSlice = true;
            }
        }

        for (int m = 0; m < _g2RWSliceRec.Length; m++)
        {
            List<SegmentND> uniqueSlices = new List<SegmentND>();
            List<SegmentND> wSlices = new List<SegmentND>();
            List<Tuple<SegmentND, TensorStat>> sliceRecs = new List<Tuple<SegmentND, TensorStat>>();
            for (int n = 0; n < _g2RWRec[m].Count; n++)
            {
                if (_g2RWRec[m][n].Item2.IsFirstSlice)
                {
                    uniqueSlices.Clear();
                    wSlices.Clear();
                    sliceRecs.Clear();
                }

                if (!uniqueSlices.Contains(_g2RWSliceRec[m][n].Item1))
                {
                    uniqueSlices.Add(_g2RWSliceRec[m][n].Item1);
                }

                wSlices.Add(_g2RWSliceRec[m][n].Item1);
                sliceRecs.Add(_g2RWSliceRec[m][n]);
                if (!_g2RWRec[m][n].Item2.IsLastSlice)
                {
                    continue;
                }

                foreach (SegmentND slice in uniqueSlices)
                {
                    sliceRecs[wSlices.FindIndex((SegmentND x) => x == slice)].Item2.IsFirstSlice = true;
                    sliceRecs[wSlices.FindLastIndex((SegmentND rWSliceIdx) => rWSliceIdx == slice)].Item2.IsLastSlice =
                        true;
                }

                int cnt;
                for (cnt = 0; cnt < wSlices.Count; cnt++)
                {
                    sliceRecs[cnt].Item2.SliceIdx = uniqueSlices.FindIndex((SegmentND x) => x == wSlices[cnt]);
                    _g2RWSliceRec[m][n - wSlices.Count + 1 + cnt] = sliceRecs[cnt];
                }
            }

            if (!(_weightType == DataTypes.Int16))
            {
                continue;
            }

            for (int reverseIndex = 1; reverseIndex < _g2RWSliceRec[m].Count; reverseIndex++)
            {
                if (_g2RWSliceRec[m][_g2RWSliceRec[m].Count - reverseIndex - 1].Item2.IsFirstSlice)
                {
                    _g2RWSliceRec[m][_g2RWSliceRec[m].Count - reverseIndex].Item2.IsFirstSlice = true;
                }

                if (_g2RWSliceRec[m][reverseIndex].Item2.IsLastSlice)
                {
                    _g2RWSliceRec[m][reverseIndex - 1].Item2.IsLastSlice = true;
                }
            }

            for (int sliceIndex = 0; sliceIndex < _g2RWSliceRec[m].Count; sliceIndex++)
            {
                _g2RWSliceRec[m][sliceIndex].Item2.SliceIdx = _g2RWSliceRec[m][sliceIndex].Item2.SliceIdx * 2 + (sliceIndex & 1);
            }
        }
    }

    private void ItemRecStatusInit()
    {
        for (int i = 0; i < _weightRec.Length; i++)
        {
            if (_weightRec[i] != null)
            {
                _weightRec[i].Clear();
            }
            else
            {
                _weightRec[i] = new List<Tuple<SegmentND, TensorStat>>();
            }

            if (_ofmapRec[i] != null)
            {
                _ofmapRec[i].Clear();
            }
            else
            {
                _ofmapRec[i] = new List<Tuple<SegmentND, TensorStat>>();
            }

            if (_g2LIfRec[i] != null)
            {
                _g2LIfRec[i].Clear();
            }
            else
            {
                _g2LIfRec[i] = new List<Tuple<SegmentND, TensorStat>>();
            }

            if (_g2RWRec[i] != null)
            {
                _g2RWRec[i].Clear();
            }
            else
            {
                _g2RWRec[i] = new List<Tuple<SegmentND, TensorStat>>();
            }

            if (_l2GOfRec[i] != null)
            {
                _l2GOfRec[i].Clear();
            }
            else
            {
                _l2GOfRec[i] = new List<Tuple<SegmentND, TensorStat>>();
            }

            if (_g2RWSliceRec[i] != null)
            {
                _g2RWSliceRec[i].Clear();
            }
            else
            {
                _g2RWSliceRec[i] = new List<Tuple<SegmentND, TensorStat>>();
            }
        }
    }

    private TileConv2dTransposeGlb SearchGlbParameters()
    {
        int i = 1;
        int j = 1;
        int kernelH = _weightsShape[2];
        int kernelW = _weightsShape[3];
        int ofmapRows = 1;
        int ofmapCols = Math.Min(GNNEEnv.PuWidth, _immParam.ImmF);
        if (ofmapCols < GNNEEnv.PuWidth)
        {
            ofmapRows = Math.Min((int)Math.Ceiling(1.0 * (double)GNNEEnv.PuWidth / (double)ofmapCols), _immParam.ImmE);
        }

        int ifmapRows = GetGlbInputHeight(ofmapRows);
        int ifmapCols = GlbInputWidth(ofmapCols);
        int k = 1;
        AllocateResult allocateResult;
        for (; j < _icPerGroup; j++)
        {
            allocateResult = HandleAllocate(i, j + 1, ifmapRows, ifmapCols, kernelH, kernelW, k, ofmapRows, ofmapCols);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; k < GNNEEnv.PuWidth && k < _ocPerGroup; k++)
        {
            allocateResult = HandleAllocate(i, j, ifmapRows, ifmapCols, kernelH, kernelW, k + 1, ofmapRows, ofmapCols);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        if (_strategy == GlbSearchStrategy.IfFirst)
        {
            if (_groupPerPass == 1)
            {
                while (k < _outputShape[1] && (kernelH != _weightsShape[2] || kernelW != _weightsShape[3] ||
                                               j < _weightsShape[1] || k < _outputShape[1] / 2))
                {
                    int nextOutChannels = k + 1;
                    int nextChannels = j;
                    if (k >= _ocPerGroup)
                    {
                        nextOutChannels = k + _ocPerGroup;
                        nextChannels = j + _icPerGroup;
                    }

                    allocateResult = HandleAllocate(i, nextChannels, ifmapRows, ifmapCols, kernelH, kernelW, nextOutChannels, ofmapRows, ofmapCols);
                    if (!allocateResult.IsOk)
                    {
                        break;
                    }

                    k = nextOutChannels;
                    j = nextChannels;
                }
            }
            else
            {
                while (k < _outputShape[1] && (kernelH != _weightsShape[2] || kernelW != _weightsShape[3] ||
                                               j < _weightsShape[1] || k < _weightsShape[0] / 2))
                {
                    int nextOutChannels = Math.Min(k + _groupPerPass * _ocPerGroup, _outputShape[1]);
                    int nextChannels = Math.Min(j + _groupPerPass * _icPerGroup, _inputShape[1]);
                    allocateResult = HandleAllocate(i, nextChannels, ifmapRows, ifmapCols, kernelH, kernelW, nextOutChannels, ofmapRows, ofmapCols);
                    if (!allocateResult.IsOk)
                    {
                        break;
                    }

                    k = nextOutChannels;
                    j = nextChannels;
                }
            }

            while (ofmapRows < _immParam.ImmE)
            {
                int nextOfmapRows = ofmapRows + 1;
                int nextIfmapRows = GetGlbInputHeight(nextOfmapRows);
                allocateResult = HandleAllocate(i, j, nextIfmapRows, ifmapCols, kernelH, kernelW, k, nextOfmapRows, ofmapCols);
                if (!allocateResult.IsOk)
                {
                    break;
                }

                ofmapRows = nextOfmapRows;
                ifmapRows = nextIfmapRows;
            }

            while (ofmapCols < _immParam.ImmF)
            {
                int nextOfmapCols = ofmapCols + 1;
                int nextIfmapCols = GlbInputWidth(nextOfmapCols);
                allocateResult = HandleAllocate(i, j, ifmapRows, nextIfmapCols, kernelH, kernelW, k, ofmapRows, nextOfmapCols);
                if (!allocateResult.IsOk)
                {
                    break;
                }

                ofmapCols = nextOfmapCols;
                ifmapCols = nextIfmapCols;
            }

            for (; i < _outputShape[0]; i++)
            {
                allocateResult = HandleAllocate(i + 1, j, ifmapRows, ifmapCols, kernelH, kernelW, k, ofmapRows, ofmapCols);
                if (!allocateResult.IsOk)
                {
                    break;
                }
            }
        }
        else if (_strategy == GlbSearchStrategy.WFirst)
        {
            while (ofmapRows < _immParam.ImmE)
            {
                int nextOfmapRows = ofmapRows + 1;
                int nextIfmapRows = GetGlbInputHeight(nextOfmapRows);
                allocateResult = HandleAllocate(i, j, nextIfmapRows, ifmapCols, kernelH, kernelW, k, nextOfmapRows, ofmapCols);
                if (!allocateResult.IsOk)
                {
                    break;
                }

                ofmapRows = nextOfmapRows;
                ifmapRows = nextIfmapRows;
            }

            while (ofmapCols < _immParam.ImmF)
            {
                int nextOfmapCols = ofmapCols + 1;
                int nextIfmapCols = GlbInputWidth(nextOfmapCols);
                allocateResult = HandleAllocate(i, j, ifmapRows, nextIfmapCols, kernelH, kernelW, k, ofmapRows, nextOfmapCols);
                if (!allocateResult.IsOk)
                {
                    break;
                }

                ofmapCols = nextOfmapCols;
                ifmapCols = nextIfmapCols;
            }

            for (;
                 i < _outputShape[0] && (j != _inputShape[1] || ifmapRows != _immParam.ImmE || ifmapCols != _immParam.ImmF ||
                                         i < _outputShape[0] / 2);
                 i++)
            {
                allocateResult = HandleAllocate(i + 1, j, ifmapRows, ifmapCols, kernelH, kernelW, k, ofmapRows, ofmapCols);
                if (!allocateResult.IsOk)
                {
                    break;
                }
            }

            if (_groupPerPass == 1)
            {
                while (k < _outputShape[1])
                {
                    int nextOutChannels = k + 1;
                    int nextChannels = j;
                    if (k >= _ocPerGroup)
                    {
                        nextOutChannels = k + _ocPerGroup;
                        nextChannels = j + _icPerGroup;
                    }

                    allocateResult = HandleAllocate(i, nextChannels, ifmapRows, ifmapCols, kernelH, kernelW, nextOutChannels, ofmapRows, ofmapCols);
                    if (!allocateResult.IsOk)
                    {
                        break;
                    }

                    k = nextOutChannels;
                    j = nextChannels;
                }
            }
            else
            {
                while (k < _outputShape[1])
                {
                    int nextOutChannels = Math.Min(k + _groupPerPass * _ocPerGroup, _outputShape[1]);
                    int nextChannels = Math.Min(j + _groupPerPass * _icPerGroup, _inputShape[1]);
                    allocateResult = HandleAllocate(i, nextChannels, ifmapRows, ifmapCols, kernelH, kernelW, nextOutChannels, ofmapRows, ofmapCols);
                    if (!allocateResult.IsOk)
                    {
                        break;
                    }

                    k = nextOutChannels;
                    j = nextChannels;
                }
            }
        }

        if (k % GNNEEnv.PuWidth != 0 && k > GNNEEnv.PuWidth && k <= _ocPerGroup)
        {
            k = k / GNNEEnv.PuWidth * GNNEEnv.PuWidth;
        }

        allocateResult = HandleAllocate(i, j, ifmapRows, ifmapCols, kernelH, kernelW, k, ofmapRows, ofmapCols, isFinal: true);
        TileUtilities.Assert(allocateResult.IsOk, "allocation.IsOk",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileConv2dTranspose.cs",
            1146);
        GNNEShape glbShape = new GNNEShape(i, k, ofmapRows, ofmapCols);
        return new TileConv2dTransposeGlb(allocateResult.GlbMap, allocateResult.Items, glbShape.Dims,
            GNNEEnv.NPingPongSplit, allocateResult.GlbMap[ItemName.Ifmap], allocateResult.GlbMap[ItemName.Ofmap]);

        int GetGlbInputHeight(int e)
        {
            return Math.Min(
                (int)Math.Ceiling((decimal)(1f * (float)(e + _weightsShape[2] - _strideH - 1) / (float)_strideH)) + 1,
                _immParam.PaddedH - _immParam.PadTop - _immParam.PadBottom);
        }

        int GlbInputWidth(int f)
        {
            return Math.Min(
                (int)Math.Ceiling((decimal)(1f * (float)(f + _weightsShape[3] - _strideW - 1) / (float)_strideW)) + 1,
                _immParam.PaddedW - _immParam.PadLeft - _immParam.PadRight);
        }
    }

    private AllocateResult HandleAllocate(int n, int c, int h, int w, int r, int s, int m, int e, int f,
        bool isFinal = false)
    {
        List<float> glbUsage = new List<float> { 0f, 0f };
        AllocateResult boxes = GetBoxes(n, c, h, w, r, s, m, e, f, glbUsage);
        AllocateResult allocateResult = SpaceSearcher.TryAllocate(new BoxPacker(16) { Boxes = boxes.Boxes });
        allocateResult.GlbMap = boxes.GlbMap;
        if (allocateResult.IsOk)
        {
            foreach (KeyValuePair<ItemName, TensorOnGlb> entry in boxes.GlbMap)
            {
                if (allocateResult.Items.ContainsKey(entry.Key))
                {
                    entry.Value.Mmu = allocateResult.Items[entry.Key];
                }
            }
        }

        return allocateResult;
    }

    private void InitParameters(Call convNode, Call ld, Call st)
    {
        _conv = convNode;
        _lif = ld;
        _lw = (Call)_conv[GNNEConv2DTranspose.Weights];
        _lact = (Call)_conv[GNNEConv2DTranspose.Act];
        _lwQarg = (Call)_conv[GNNEConv2DTranspose.WeightsBias];
        _sof = st;
        _inputShape = new GNNEShape(_lif.CheckedShape[0].FixedValue, _lif.CheckedShape[1].FixedValue,
            _lif.CheckedShape[2].FixedValue, _lif.CheckedShape[3].FixedValue);
        _outputShape = new GNNEShape(_conv.CheckedShape[0].FixedValue, _conv.CheckedShape[1].FixedValue,
            _conv.CheckedShape[2].FixedValue, _conv.CheckedShape[3].FixedValue);
        _groups = ((TensorConst)_conv[GNNEConv2DTranspose.Groups]).Value.ToScalar<int>();
        _weightsShape = new GNNEShape(_lw.CheckedShape[0].FixedValue * _groups,
            _lw.CheckedShape[1].FixedValue / _groups, _lw.CheckedShape[2].FixedValue, _lw.CheckedShape[3].FixedValue);
        int[] paddings = ((TensorConst)_conv[GNNEConv2DTranspose.Padding]).Value.ToArray<int>();
        _paddingH = new Padding(paddings[0], paddings[1]);
        _paddingW = new Padding(paddings[2], paddings[3]);
        _outputPaddingH = ((TensorConst)_conv[GNNEConv2DTranspose.OutputPadding]).Value.ToArray<int>()[0];
        _outputPaddingW = ((TensorConst)_conv[GNNEConv2DTranspose.OutputPadding]).Value.ToArray<int>()[1];
        _strideH = ((TensorConst)_conv[GNNEConv2DTranspose.Stride]).Value.ToArray<int>()[0];
        _strideW = ((TensorConst)_conv[GNNEConv2DTranspose.Stride]).Value.ToArray<int>()[1];
        _dilationH = ((TensorConst)_conv[GNNEConv2DTranspose.Dilation]).Value.ToArray<int>()[0];
        _dilationW = ((TensorConst)_conv[GNNEConv2DTranspose.Dilation]).Value.ToArray<int>()[1];
        _kernelH = _lw.CheckedShape[2].FixedValue;
        _kernelW = _lw.CheckedShape[3].FixedValue;
        _icPerGroup = _inputShape[1] / _groups;
        _ocPerGroup = _outputShape[1] / _groups;
        _groupPerPass = Math.Min(Math.Max(Math.Min(GNNEEnv.PuHeight / _icPerGroup, GNNEEnv.PuWidth / _ocPerGroup), 1),
            _groups);
        _strategy = DetermineSearchStrategy();
        _inputType = _lif.CheckedDataType;
        _weightType = _lw.CheckedDataType;
        _weightGroup = new WeightGroupHandler(_weightType, _weightType);
        _outputType = _conv.CheckedDataType;
        ref int paddedH = ref _immParam.PaddedH;
        ref int immE = ref _immParam.ImmE;
        (int, int) convTPadHImmE = TileUtilities.GetConvTPadHImmE(_inputShape[2], _strideH, _kernelH, _outputShape[2]);
        paddedH = convTPadHImmE.Item1;
        immE = convTPadHImmE.Item2;
        paddedH = ref _immParam.PaddedW;
        ref int immF = ref _immParam.ImmF;
        convTPadHImmE = TileUtilities.GetConvTPadWImmF(_inputShape[3], _strideW, _kernelW, _outputShape[3]);
        paddedH = convTPadHImmE.Item1;
        immF = convTPadHImmE.Item2;
        Padding padding = TileUtilities.CalPadTopBottom(_immParam.PaddedH, _inputShape[2]);
        _immParam.PadTop = padding.Before;
        _immParam.PadBottom = padding.After;
        if (_paddingH.Before > 0)
        {
            _immParam.EStart = _paddingH.Before;
            _immParam.EEnd = _immParam.EStart + _outputShape[2];
        }

        Padding padding2 = TileUtilities.CalPadLeftRight(_immParam.PaddedW, _inputShape[3]);
        _immParam.PadLeft = padding2.Before;
        _immParam.PadRight = padding2.After;
        if (_paddingW.Before > 0)
        {
            _immParam.FStart = _paddingW.Before;
            _immParam.FEnd = _immParam.FStart + _outputShape[3];
        }

        paddedH = ref _immParam.EStart;
        ref int eEnd = ref _immParam.EEnd;
        convTPadHImmE = TileUtilities.GetConvTOutputStartEnd(_inputShape[2], _immParam.PaddedH, _immParam.ImmE,
            _outputShape[2], _paddingH, _outputPaddingH);
        paddedH = convTPadHImmE.Item1;
        eEnd = convTPadHImmE.Item2;
        paddedH = ref _immParam.FStart;
        ref int fEnd = ref _immParam.FEnd;
        (paddedH, fEnd) = TileUtilities.GetConvTOutputStartEnd(_inputShape[3], _immParam.PaddedW, _immParam.ImmF,
            _outputShape[3], _paddingW, _outputPaddingW);
    }

    private GlbSearchStrategy DetermineSearchStrategy()
    {
        if (_inputShape?.Size > _weightsShape?.Size)
        {
            return GlbSearchStrategy.IfFirst;
        }

        return GlbSearchStrategy.WFirst;
    }

    private AllocateResult GetBoxes(int n, int c, int h, int w, int r, int s, int m, int e, int f, List<float> glbUsage)
    {
        List<BoxOnGlb> boxes = new List<BoxOnGlb>();
        Dictionary<ItemName, TensorOnGlb> glbMap = new Dictionary<ItemName, TensorOnGlb>();
        List<float> glbUsageBytes = new List<float> { 0f, 0f };
        int pingPongSplit = GNNEEnv.NPingPongSplit;
        if (n == _inputShape[0] && c == _inputShape[1] && h == _inputShape[2] && w == _inputShape[3])
        {
            pingPongSplit = 1;
        }

        TensorOnGlb ifmapTensor = new TensorOnGlb(new int[4] { n, c, h, w }, _inputType, 0);
        int ifmapBytes = ifmapTensor.AllocatedBytes * pingPongSplit;
        glbUsageBytes[0] += ifmapBytes;
        ifmapBytes = TileUtilities.GetAlignedNum(ifmapBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        glbUsageBytes[1] += ifmapBytes;
        int icInGroup = c;
        if (icInGroup > _icPerGroup)
        {
            icInGroup = _icPerGroup;
        }

        TensorOnGlb weightTensor = new TensorOnGlb(new int[4] { m, icInGroup, r, s }, _weightType, 0);
        int weightBytes = SpaceSearcher.GetWeightSize(r, s, icInGroup, m, TileUtilities.GetBytesPerElement(_weightType)) *
                   GNNEEnv.NPingPongSplit;
        weightTensor.AllocatedBytes = weightBytes / GNNEEnv.NPingPongSplit;
        glbUsageBytes[0] += weightBytes;
        weightBytes = TileUtilities.GetAlignedNum(weightBytes, GNNEEnv.WBankWidth * GNNEEnv.GlbBankWidth);
        glbUsageBytes[1] += weightBytes;
        TensorOnGlb ofmapTensor = new TensorOnGlb(new int[4] { n, m, e, f }, _outputType, 0);
        int ofmapBytes = ofmapTensor.AllocatedBytes * GNNEEnv.NPingPongSplit;
        glbUsageBytes[0] += ofmapBytes;
        ofmapBytes = TileUtilities.GetAlignedNum(ofmapBytes, GNNEEnv.OfmapBankWidth * GNNEEnv.GlbBankWidth);
        glbUsageBytes[1] += ofmapBytes;
        TensorOnGlb actTensor = new TensorOnGlb(new int[4] { 1, 1, _outputShape[1], GNNEEnv.ActNumPerChan },
            _lact.CheckedDataType, 0);
        int actSize = SpaceSearcher.GetActSize(_outputShape[1], GNNEEnv.ActNumPerChan,
            TileUtilities.GetBytesPerElement(_lact.CheckedDataType));
        glbUsageBytes[0] += actSize;
        actSize = TileUtilities.GetAlignedNum(actSize, GNNEEnv.ActBankWidth * GNNEEnv.GlbBankWidth);
        glbUsageBytes[1] += actSize;
        int basementSize = SpaceSearcher.GetBasementSize();
        basementSize = TileUtilities.GetAlignedNum(basementSize, GNNEEnv.GlbWidth * GNNEEnv.GlbBankWidth);
        boxes.Add(new BoxOnGlb(new int[2] { GNNEEnv.GlbWidth, basementSize / GNNEEnv.GlbWidth / GNNEEnv.GlbBankWidth },
            ItemName.Basement));
        boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.IfmapBankWidth, ifmapBytes / GNNEEnv.IfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.Ifmap));
        boxes.Add(new BoxOnGlb(new int[2] { GNNEEnv.WBankWidth, weightBytes / GNNEEnv.WBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.Weight));
        boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.OfmapBankWidth, ofmapBytes / GNNEEnv.OfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.Ofmap));
        boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.ActBankWidth, actSize / GNNEEnv.ActBankWidth / GNNEEnv.GlbBankWidth }, ItemName.Act));
        glbMap.Add(ItemName.Ifmap, ifmapTensor);
        glbMap.Add(ItemName.Weight, weightTensor);
        glbMap.Add(ItemName.Ofmap, ofmapTensor);
        glbMap.Add(ItemName.Act, actTensor);
        if (_weightType == DataTypes.UInt8 || _weightType == DataTypes.Int16)
        {
            int[] wQargDims = new int[4] { 1, 1, 1, 0 };
            wQargDims[3] = _outputShape[1];
            TensorOnGlb wQargTensor = new TensorOnGlb(wQargDims, _lwQarg.CheckedDataType, 0);
            int wQargSize = SpaceSearcher.GetWQargSize(_outputShape[1],
                TileUtilities.GetBytesPerElement(_lwQarg.CheckedDataType));
            if (_inputShape[1] == _outputShape[1] && _outputShape[1] == _groups && _groups != 1)
            {
                wQargSize = SpaceSearcher.GetWQargSize(_outputShape[1] * GNNEEnv.PuWidth,
                    TileUtilities.GetBytesPerElement(_lwQarg.CheckedDataType));
            }

            glbUsageBytes[0] += wQargSize;
            wQargSize = TileUtilities.GetAlignedNum(wQargSize, GNNEEnv.WQargBankWidth * GNNEEnv.GlbBankWidth);
            glbUsageBytes[1] += wQargSize;
            boxes.Add(new BoxOnGlb(
                new int[2] { GNNEEnv.WQargBankWidth, wQargSize / GNNEEnv.WQargBankWidth / GNNEEnv.GlbBankWidth },
                ItemName.WQarg));
            glbMap.Add(ItemName.WQarg, wQargTensor);
        }

        glbUsage[0] = glbUsageBytes[0] / (float)GNNEEnv.GlbSize;
        glbUsage[1] = glbUsageBytes[1] / (float)GNNEEnv.GlbSize;
        return new AllocateResult { Boxes = boxes, GlbMap = glbMap };
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Call call = (Call)__result["call"];
        Call ld = (Call)__result["ld"];
        Call st = (Call)__result["st"];
        return GetReplace(call, ld, st);
    }
}
