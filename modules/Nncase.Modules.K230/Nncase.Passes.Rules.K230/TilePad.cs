// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.PatternMatch;
using Nncase.Runtime.K230;
using Nncase.TIR;
using Nncase.TIR.Instructions;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class TilePad : RewriteRule<Pattern>
{
    private static int _count = -1;

    private CcrHandler? _ccrHandler = new CcrHandler();

    private GprHandler? _gpr = new GprHandler();

    private SsrHandler? _ssr = new SsrHandler();

    private DataType? _inputType;

    private DataType? _outputType;

    private Call? _pad;

    private Call? _lif;

    private Call? _sof;

    private GNNEShape? _inputShape;

    private GNNEShape? _outputShape;

    public override Pattern Pattern { get; } = FusionPattern.IsGNNEPadFusion();

    private PrimFunction GetReplace(Call call, Call ld, Call st)
    {
        _count++;
        int[] inputShape = ld[GNNELoad.Input].CheckedShape.ToValueArray();
        T.CreateBuffer(
            new TensorType(ld[GNNELoad.Input].CheckedDataType, inputShape),
            MemoryLocation.Input,
            out Nncase.TIR.Buffer ddrIf,
            "var ddrIf");
        T.CreateBuffer(
            new TensorType(st.CheckedDataType, st.CheckedShape),
            MemoryLocation.Output,
            out Nncase.TIR.Buffer ddrOf,
            "var ddrOf");
        InitParameters(call, ld, st);
        TiledGlb glb = SearchGlbParameters();
        List<GnneAction> actions = BuildSchedule(call, glb, ddrIf, ddrOf);
        ActionToInstruct actionToInstruct = new ActionToInstruct();
        return T.PrimFunc($"TilePad_{_count}", K230RtModule.Kind, ddrIf, ddrOf)
            .Body(actionToInstruct.Instructions(actions), I.END(GP_REGISTER.x0, 0))
            .Build();
    }

    private void InitParameters(Call padNode, Call ld, Call st)
    {
        _pad = padNode;
        _lif = ld;
        _sof = st;
        _inputType = _lif.CheckedDataType;
        _outputType = _pad.CheckedDataType;
        _inputShape = new GNNEShape(
            _lif.CheckedShape[0].FixedValue,
            _lif.CheckedShape[1].FixedValue,
            _lif.CheckedShape[2].FixedValue,
            _lif.CheckedShape[3].FixedValue);
        _outputShape = new GNNEShape(
            _pad.CheckedShape[0].FixedValue,
            _pad.CheckedShape[1].FixedValue,
            _pad.CheckedShape[2].FixedValue,
            _pad.CheckedShape[3].FixedValue);
    }

    private TileConv2DGlb SearchGlbParameters()
    {
        int batch = 1;
        if (GNNEEnv.BatchInference)
        {
            batch = Math.Min(4, _inputShape[0]);
        }

        int channels = 1;
        int height = 1;
        int width = _outputShape[3];
        AllocateResult allocateResult;
        for (; height < _outputShape[2]; height++)
        {
            allocateResult = HandleAllocate(batch, channels, height + 1, width);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; channels < _outputShape[1]; channels++)
        {
            allocateResult = HandleAllocate(batch, channels + 1, height, width);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        allocateResult = HandleAllocate(batch, channels, height, width, isFinal: true);
        TileUtilities.Assert(
            allocateResult.IsOk,
            "allocation.IsOk",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TilePad.cs",
            103);
        GNNEShape glbShape = new GNNEShape(batch, channels, height, width);
        return new TileConv2DGlb(
            allocateResult.GlbMap,
            allocateResult.Items,
            glbShape.Dims,
            GNNEEnv.NPingPongSplit,
            allocateResult.GlbMap[ItemName.Ifmap],
            allocateResult.GlbMap[ItemName.Ofmap]);
    }

    private AllocateResult HandleAllocate(int n, int m, int e, int f, bool isFinal = false)
    {
        List<BoxOnGlb> boxes = new List<BoxOnGlb>();
        Dictionary<ItemName, TensorOnGlb> glbMap = new Dictionary<ItemName, TensorOnGlb>();
        TileUtilities.GetBytesPerElement(_inputType);
        int channelsPerSplit = (int)Math.Ceiling(1.0 * (double)m / (double)GNNEEnv.NPingPongSplit);
        TensorOnGlb ofmapTensor = new TensorOnGlb(new int[4] { n, channelsPerSplit, e, f }, _inputType, 0);
        int allocatedBytes = ofmapTensor.AllocatedBytes;
        allocatedBytes = ofmapTensor.AllocatedBytes =
            TileUtilities.GetAlignedNum(allocatedBytes, GNNEEnv.OfmapBankWidth * GNNEEnv.GlbBankWidth);
        allocatedBytes *= GNNEEnv.NPingPongSplit;
        TensorOnGlb ifmapTensor = ofmapTensor;
        int basementSize = SpaceSearcher.GetBasementSize();
        basementSize = TileUtilities.GetAlignedNum(basementSize, GNNEEnv.GlbWidth * GNNEEnv.GlbBankWidth);
        boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.GlbWidth, basementSize / GNNEEnv.GlbWidth / GNNEEnv.GlbBankWidth },
            ItemName.Basement));
        boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.OfmapBankWidth, allocatedBytes / GNNEEnv.OfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.Ofmap));
        AllocateResult allocateResult = SpaceSearcher.TryAllocate(new BoxPacker(16) { Boxes = boxes });
        if (allocateResult.IsOk)
        {
            ifmapTensor.Mmu = allocateResult.Items[ItemName.Ofmap];
            ofmapTensor.Mmu = allocateResult.Items[ItemName.Ofmap];
        }

        glbMap.Add(ItemName.Ifmap, ifmapTensor);
        glbMap.Add(ItemName.Ofmap, ofmapTensor);
        return new AllocateResult { IsOk = allocateResult.IsOk, Items = allocateResult.Items, GlbMap = glbMap };
    }

    private List<GnneAction> BuildSchedule(Call pad, TiledGlb glb, Nncase.TIR.Buffer ddrIf, Nncase.TIR.Buffer ddrOf)
    {
        int[] pads = ((TensorConst)pad[GNNEPad.Pads]).Value.ToArray<int>();
        Padding paddingN = new Padding(pads[0], pads[1]);
        Padding paddingC = new Padding(pads[2], pads[3]);
        Padding paddingH = new Padding(pads[4], pads[5]);
        Padding paddingW = new Padding(pads[6], pads[7]);
        DataType checkedDataType = pad.CheckedDataType;
        DataType dataType = checkedDataType;
        if ((object)dataType != null)
        {
            short padValueBits;
            if (dataType == DataTypes.UInt8)
            {
                padValueBits = ((TensorConst)pad[GNNEPad.Value]).Value.ToScalar<byte>();
            }
            else if (dataType == DataTypes.Int8)
            {
                padValueBits = ((TensorConst)pad[GNNEPad.Value]).Value.ToScalar<sbyte>();
            }
            else if (dataType == DataTypes.Int16)
            {
                padValueBits = ((TensorConst)pad[GNNEPad.Value]).Value.ToScalar<short>();
            }
            else
            {
                if (!(dataType == DataTypes.Float16))
                {
                    throw new ArgumentOutOfRangeException(checkedDataType.GetDisplayName());
                }

                padValueBits = BitConverter.ToInt16(
                    BitConverter.GetBytes(((TensorConst)pad[GNNEPad.Value]).Value.ToScalar<Half>()),
                    0);
            }

            int padValue = padValueBits;
            List<GnneAction> actions = new List<GnneAction>();
            GprHandler gpr = new GprHandler(GNNEEnv.GprNum);
            SsrHandler ssr = new SsrHandler(GNNEEnv.SsrNum);
            CcrHandler ccrHandler = new CcrHandler();
            GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, glb, ccrHandler, gpr, ssr);
            gnneActionUpdater.UpdateMmuConf();
            foreach (Segment1D segmentN in
                     TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[0], _outputShape[0]))
            {
                Segment1D inputSegmentN = TileUtilities.GetInputRowSegment(
                    segmentN.Start, segmentN.Length, _inputShape[0], 1, 1, 1, in paddingN);
                foreach (Segment1D segmentC in
                         TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[1], _outputShape[1]))
                {
                    Segment1D inputSegmentC = TileUtilities.GetInputRowSegment(
                        segmentC.Start, segmentC.Length, _inputShape[1], 1, 1, 1, in paddingC);
                    foreach (Segment1D segmentH in
                             TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[2], _outputShape[2]))
                    {
                        Segment1D inputSegmentH = TileUtilities.GetInputRowSegment(
                            segmentH.Start, segmentH.Length, _inputShape[2], 1, 1, 1, in paddingH);
                        foreach (Segment1D segmentW in
                                 TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[3], _outputShape[3]))
                        {
                            Segment1D inputSegmentW = TileUtilities.GetInputRowSegment(
                                segmentW.Start, segmentW.Length, _inputShape[3], 1, 1, 1, in paddingW);
                            SegmentND outputSegment = new SegmentND(segmentN, segmentC, segmentH, segmentW);
                            SegmentND inputSegment = new SegmentND(
                                inputSegmentN, inputSegmentC, inputSegmentH, inputSegmentW);
                            int chunkSize = (int)Math.Ceiling((float)segmentC.Length / (float)GNNEEnv.NPingPongSplit);
                            List<Segment1D> chunkSegments =
                                TileUtilities.GetSegmentStartEndLength(segmentC.Start, chunkSize, segmentC.End);
                            for (int chunkIndex = 0; chunkIndex < chunkSegments.Count; chunkIndex++)
                            {
                                Segment1D chunkSegment = chunkSegments[chunkIndex];
                                Segment1D inputChunkSegment = TileUtilities.GetInputRowSegment(
                                    chunkSegment.Start, chunkSegment.Length, _inputShape[1], 1, 1, 1, in paddingC);
                                SegmentND ifmap = new SegmentND(
                                    inputSegment[0], inputChunkSegment, inputSegment[2], inputSegment[3]);
                                SegmentND outputChunk = new SegmentND(
                                    outputSegment[0], chunkSegment, outputSegment[2], outputSegment[3]);
                                gnneActionUpdater.UpdateMfuMemset(
                                    outputChunk, outputChunk, padValue, _inputType,
                                    glb.GlbMap[ItemName.Ofmap].Mmu.Id, chunkIndex);
                                List<CcrSet> ccrsToSet = new List<CcrSet>
                                {
                                    new CcrSet(
                                        ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, chunkIndex)), 1),
                                };
                                int loadOffset =
                                    (inputSegmentN.Padding.Before * outputChunk[1].Length * outputChunk[2].Length *
                                     outputChunk[3].Length) +
                                    (inputChunkSegment.Padding.Before * outputChunk[2].Length *
                                     outputChunk[3].Length) +
                                    (inputSegmentH.Padding.Before * outputChunk[3].Length) +
                                    inputSegmentW.Padding.Before;
                                loadOffset *= TileUtilities.GetBytesPerElement(_inputType);
                                gnneActionUpdater.UpdateLoadIf(
                                    ifmap, _lif, chunkIndex, ddrIf, loadOffset,
                                    new List<int> { outputSegment[1].Length, outputSegment[2].Length, outputSegment[3].Length },
                                    ItemName.Ofmap, ccrsToSet);
                                List<CcrClr> ccrsToClr = new List<CcrClr>
                                {
                                    new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, chunkIndex))),
                                };
                                gnneActionUpdater.UpdateStoreT(
                                    outputChunk, _sof, chunkIndex, ddrOf, 0, null, null, ccrsToClr);
                            }
                        }
                    }
                }
            }

            TileUtilities.Assert(
                ccrHandler.CcrSanityCheck(),
                "ccrHandler.CcrSanityCheck()",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TilePad.cs",
                242);
            return actions;
        }

        throw new ArgumentOutOfRangeException(checkedDataType.GetDisplayName());
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Call call = (Call)__result["call"];
        Call ld = (Call)__result["ld"];
        Call st = (Call)__result["st"];
        return GetReplace(call, ld, st);
    }
}
