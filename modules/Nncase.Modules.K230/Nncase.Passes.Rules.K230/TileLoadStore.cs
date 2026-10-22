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
public class TileLoadStore : RewriteRule<Pattern>
{
    private static int _count = -1;

    private CcrHandler _ccrHandler = new CcrHandler();

    private GprHandler _gpr = new GprHandler();

    private SsrHandler _ssr = new SsrHandler();

    private DataType? _inputType;

    private DataType? _outputType;

    private Call? _lif;

    private Call? _sof;

    private GNNEShape? _inputShape;

    private GNNEShape? _outputShape;

    public override Pattern Pattern { get; } = FusionPattern.IsGNNELoadStoreFusion();

    private PrimFunction GetReplace(Call ldCall, Call stCall)
    {
        _count++;
        int[] inputShape = ldCall[GNNELoad.Input].CheckedShape.ToValueArray();
        T.CreateBuffer(new TensorType(ldCall[GNNELoad.Input].CheckedDataType, inputShape), MemoryLocation.Input,
            out Nncase.TIR.Buffer buffer, "var ddrIf");
        T.CreateBuffer(new TensorType(stCall.CheckedDataType, stCall.CheckedShape), MemoryLocation.Output,
            out Nncase.TIR.Buffer buffer2, "var ddrOf");
        InitParameters(ldCall, stCall);
        TiledGlb glb = SearchGlbParameters();
        List<GnneAction> actions = BuildSchedule(glb, buffer, buffer2);
        ActionToInstruct actionToInstruct = new ActionToInstruct();
        return T.PrimFunc($"TileLoadStore_{_count}", K230RtModule.Kind, buffer, buffer2)
            .Body(actionToInstruct.Instructions(actions), I.END(GP_REGISTER.x0, 0)).Build();
    }

    private void InitParameters(Call ld, Call st)
    {
        _lif = ld;
        _sof = st;
        _inputType = _lif.CheckedDataType;
        _outputType = _sof.CheckedDataType;
        _inputShape = new GNNEShape(_lif.CheckedShape[0].FixedValue, _lif.CheckedShape[1].FixedValue,
            _lif.CheckedShape[2].FixedValue, _lif.CheckedShape[3].FixedValue);
        _outputShape = new GNNEShape(_sof.CheckedShape[0].FixedValue, _sof.CheckedShape[1].FixedValue,
            _sof.CheckedShape[2].FixedValue, _sof.CheckedShape[3].FixedValue);
    }

    private TileLoadStoreGlb SearchGlbParameters()
    {
        int i = 1;
        int j = 1;
        int k = 1;
        int width = _outputShape[3];
        int w = _inputShape[3];
        int nPingPongSplit = GNNEEnv.NPingPongSplit;
        AllocateResult allocateResult = HandleAllocate(i, j, k, w, k, width, nPingPongSplit);
        if (!allocateResult.IsOk)
        {
            nPingPongSplit = 1;
        }

        for (; k < _inputShape[2] && k < 65535; k++)
        {
            int nextRows = k + 1;
            allocateResult = HandleAllocate(i, j, nextRows, w, nextRows, width, nPingPongSplit);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; j < _inputShape[1] && j < 65535; j++)
        {
            allocateResult = HandleAllocate(i, j + 1, k, w, k, width, nPingPongSplit);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; i < _inputShape[0] && i < 65535; i++)
        {
            allocateResult = HandleAllocate(i + 1, j, k, w, k, width, nPingPongSplit);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        allocateResult = HandleAllocate(i, j, k, w, k, width, nPingPongSplit, isFinal: true);
        TileUtilities.Assert(allocateResult.IsOk, "allocation.IsOk",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLoadStore.cs",
            123);
        GNNEShape glbShape = new GNNEShape(i, j, k, width);
        return new TileLoadStoreGlb(allocateResult.GlbMap, allocateResult.Items, glbShape.Dims, GNNEEnv.NPingPongSplit,
            allocateResult.GlbMap[ItemName.Ifmap], allocateResult.GlbMap[ItemName.Ofmap]);
    }

    private AllocateResult HandleAllocate(int n, int c, int h, int w, int e, int f, int nPingPongSplit,
        bool isFinal = false)
    {
        List<BoxOnGlb> boxes = new List<BoxOnGlb>();
        Dictionary<ItemName, TensorOnGlb> glbMap = new Dictionary<ItemName, TensorOnGlb>();
        int channelsPerPass = (int)Math.Ceiling(1.0 * (double)c / (double)nPingPongSplit);
        TensorOnGlb ifmapTensor = new TensorOnGlb(new int[4] { n, channelsPerPass, h, w }, _inputType, 0);
        int value = ifmapTensor.AllocatedBytes * nPingPongSplit;
        value = TileUtilities.GetAlignedNum(value, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        TensorOnGlb ofmapTensor = new TensorOnGlb(new int[4] { n, channelsPerPass, h, w }, _inputType, 0);
        int basementSize = SpaceSearcher.GetBasementSize();
        basementSize = TileUtilities.GetAlignedNum(basementSize, GNNEEnv.GlbWidth * GNNEEnv.GlbBankWidth);
        boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.GlbWidth, basementSize / GNNEEnv.GlbWidth / GNNEEnv.GlbBankWidth },
            ItemName.Basement));
        boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.IfmapBankWidth, value / GNNEEnv.IfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.Ifmap));
        AllocateResult allocateResult = SpaceSearcher.TryAllocate(new BoxPacker(16) { Boxes = boxes });
        if (allocateResult.IsOk)
        {
            ifmapTensor.Mmu = allocateResult.Items[ItemName.Ifmap];
            ofmapTensor.Mmu = allocateResult.Items[ItemName.Ifmap];
        }

        glbMap.Add(ItemName.Ifmap, ifmapTensor);
        glbMap.Add(ItemName.Ofmap, ofmapTensor);
        return new AllocateResult { IsOk = allocateResult.IsOk, Items = allocateResult.Items, GlbMap = glbMap };
    }

    private List<GnneAction> BuildSchedule(TiledGlb glb, Nncase.TIR.Buffer ddrIf, Nncase.TIR.Buffer ddrOf)
    {
        List<GnneAction> actions = new List<GnneAction>();
        GprHandler gpr = new GprHandler(GNNEEnv.GprNum);
        SsrHandler ssr = new SsrHandler(GNNEEnv.SsrNum);
        CcrHandler ccrHandler = new CcrHandler();
        GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, glb, ccrHandler, gpr, ssr);
        gnneActionUpdater.UpdateMmuConf();
        int[] strides = ((TensorConst)_sof[GNNEStore.Strides]).Value.ToArray<int>();
        Func<int, int, int, Segment1D> func = (start, end, dim) =>
        {
            int scaledStart = (int)Math.Ceiling(1.0 * (double)start / (double)strides[dim]);
            return new Segment1D(
                new System.Range(end: (int)Math.Ceiling(1.0 * (double)end / (double)strides[dim]), start: scaledStart),
                new Padding(0, 0));
        };
        foreach (Segment1D segmentN in TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[0], _inputShape[0]))
        {
            Segment1D scaledN = func(segmentN.Start, segmentN.End, 0);
            foreach (Segment1D segmentC in TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[1], _inputShape[1]))
            {
                Segment1D scaledC = func(segmentC.Start, segmentC.End, 1);
                foreach (Segment1D segmentH in TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[2],
                             _inputShape[2]))
                {
                    Segment1D scaledH = func(segmentH.Start, segmentH.End, 2);
                    foreach (Segment1D segmentW in TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[3],
                                 _inputShape[3]))
                    {
                        Segment1D scaledW = func(segmentW.Start, segmentW.End, 3);
                        SegmentND scaledSegment = new SegmentND(scaledN, scaledC, scaledH, scaledW);
                        SegmentND loadSegment = new SegmentND(scaledN, segmentC, segmentH, segmentW);
                        int chunkSize = (int)Math.Ceiling(1.0 * (double)segmentC.Length / (double)glb.NPingPongSplit);
                        List<Segment1D> channelSegments =
                            TileUtilities.GetSegmentStartEndLength(segmentC.Start, chunkSize, segmentC.End);
                        for (int channelIndex = 0; channelIndex < channelSegments.Count; channelIndex++)
                        {
                            Segment1D channelSegment = channelSegments[channelIndex];
                            Segment1D scaledChannelSegment = func(channelSegment.Start, channelSegment.End, 1);
                            List<CcrSet> ccrsToSet = new List<CcrSet>
                            {
                                new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, channelIndex)), 1),
                            };
                            SegmentND tensor = new SegmentND(loadSegment[0], channelSegment, loadSegment[2], loadSegment[3]);
                            List<int> stridesD =
                                new int[3] { tensor[1].Length, tensor[2].Length, tensor[3].Length }.ToList();
                            gnneActionUpdater.UpdateLoadIf(tensor, _lif, channelIndex, ddrIf, 0, stridesD, ItemName.Ifmap,
                                ccrsToSet);
                            List<CcrClr> ccrsToClr = new List<CcrClr>
                            {
                                new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, channelIndex))),
                            };
                            SegmentND ofmap = new SegmentND(scaledSegment[0], scaledChannelSegment, scaledSegment[2], scaledSegment[3]);
                            SegmentND slice = new SegmentND(scaledSegment[0], scaledChannelSegment, scaledSegment[2], scaledSegment[3]);
                            slice[0] = new Segment1D((slice[0].Start * strides[0])..slice[0].End, new Padding(0, 0));
                            slice[1] = new Segment1D((slice[1].Start * strides[1])..slice[1].End, new Padding(0, 0));
                            slice[2] = new Segment1D((slice[2].Start * strides[2])..slice[2].End, new Padding(0, 0));
                            slice[3] = new Segment1D((slice[3].Start * strides[3])..slice[3].End, new Padding(0, 0));
                            int sliceOffsetInTensor = TileUtilities.GetSliceOffsetInTensor(in tensor, in slice);
                            gnneActionUpdater.UpdateStoreT(ofmap, _sof, channelIndex, ddrOf, sliceOffsetInTensor, null, null,
                                ccrsToClr, ItemName.Ofmap, stridesD);
                        }
                    }
                }
            }
        }

        TileUtilities.Assert(ccrHandler.CcrSanityCheck(), "ccrHandler.CcrSanityCheck()",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLoadStore.cs",
            250);
        return actions;
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Call ldCall = (Call)result["ldCall"];
        Call stCall = (Call)result["stCall"];
        return GetReplace(ldCall, stCall);
    }
}
