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
public class TileResize : RewriteRule<Pattern>
{
    private static int _count = -1;

    private readonly List<Tuple<SegmentND, TensorStat>>[] _r2GOfRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _ofmapRec = new List<Tuple<SegmentND, TensorStat>>[2];

    public override Pattern Pattern { get; } = FusionPattern.IsAi2DResizeFusion();

    private PrimFunction GetReplace(Call call, Call ld, Call st)
    {
        Ai2dResize resize = (Ai2dResize)call.Target;
        _count++;
        ItemRecStatusInit();
        TileAi2DResizeGlb glb = SearchGlbParameters(ld, st);
        int[] inputShape = ld[GNNELoad.Input].CheckedShape.ToValueArray();
        T.CreateBuffer(new TensorType(ld[GNNELoad.Input].CheckedDataType, inputShape), MemoryLocation.Input,
            out Nncase.TIR.Buffer ddrIf, "var ddrIf");
        T.CreateBuffer(new TensorType(st.CheckedDataType, st.CheckedShape), MemoryLocation.Output,
            out Nncase.TIR.Buffer ddrOf, "var ddrOf");
        BuildSchedule(glb, resize, ld, st, ddrIf, ddrOf, recStat: true);
        List<GnneAction> actions = BuildSchedule(glb, resize, ld, st, ddrIf, ddrOf, recStat: false);
        ActionToInstruct actionToInstruct = new ActionToInstruct();
        return T.PrimFunc($"TileResize_{_count}", K230RtModule.Kind, ddrIf, ddrOf)
            .Body(actionToInstruct.Instructions(actions), I.END(GP_REGISTER.x0, 0)).Build();
    }

    private List<GnneAction> BuildSchedule(TiledGlb glb, Ai2dResize resize, Call ld, Call st, Nncase.TIR.Buffer ddrIf,
        Nncase.TIR.Buffer ddrOf, bool recStat)
    {
        List<GnneAction> actions = new List<GnneAction>();
        GprHandler gpr = new GprHandler(GNNEEnv.GprNum);
        SsrHandler ssr = new SsrHandler(GNNEEnv.SsrNum);
        CcrHandler ccrHandler = new CcrHandler();
        GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, glb, ccrHandler, gpr, ssr);
        TileUtilities.Assert(!resize.AlignCorners || !resize.HalfPixelCenters,
            "!(resize.AlignCorners && resize.HalfPixelCenters)",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileAi2DResize.cs",
            51);
        if (!recStat)
        {
            for (int recIndex = 0; recIndex < _r2GOfRec.Length; recIndex++)
            {
                for (int sliceIndex = 0; sliceIndex < _r2GOfRec[recIndex].Count; sliceIndex++)
                {
                    if (sliceIndex == 0)
                    {
                        _r2GOfRec[recIndex][sliceIndex].Item2.IsFirstSlice = true;
                    }

                    if (sliceIndex == _r2GOfRec[recIndex].Count - 1)
                    {
                        _r2GOfRec[recIndex][sliceIndex].Item2.IsLastSlice = true;
                    }

                    if (sliceIndex < _r2GOfRec[recIndex].Count - 1 && _r2GOfRec[recIndex][sliceIndex].Item1 != _r2GOfRec[recIndex][sliceIndex + 1].Item1)
                    {
                        _r2GOfRec[recIndex][sliceIndex].Item2.IsLastSlice = true;
                        _r2GOfRec[recIndex][sliceIndex + 1].Item2.IsFirstSlice = true;
                    }
                }
            }

            for (int ofmapIndex = 0; ofmapIndex < _ofmapRec.Length; ofmapIndex++)
            {
                if (_ofmapRec[ofmapIndex].Count > 0)
                {
                    _ofmapRec[ofmapIndex][0].Item2.IsFirstSlice = true;
                    _ofmapRec[ofmapIndex][_ofmapRec[ofmapIndex].Count - 1].Item2.IsLastSlice = true;
                }
            }

            if (!recStat)
            {
                gnneActionUpdater.UpdateMmuConf();
            }
        }

        GNNEShape inputShape = new GNNEShape(ld.CheckedShape.ToValueArray());
        GNNEShape outputShape = new GNNEShape(st.CheckedShape.ToValueArray());
        DataType inputDataType = ld.CheckedDataType;
        DataType outputDataType = st[GNNEStore.Input].CheckedDataType;
        List<Segment1D> segmentsN =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[0], outputShape[0]);
        List<Segment1D> segmentsC =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[1], outputShape[1]);
        List<Segment1D> segmentsH =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[2], outputShape[2]);
        List<Segment1D> segmentsW =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[3], outputShape[3]);
        int pingPongIndex = 0;
        foreach (Segment1D segmentN in segmentsN)
        {
            Segment1D inputSegmentN = segmentN;
            foreach (Segment1D segmentC in segmentsC)
            {
                Segment1D inputSegmentC = segmentC;
                foreach (Segment1D segmentH in segmentsH)
                {
                    Segment1D inputSegmentH = new Segment1D(0..inputShape[2], Padding.Zero());
                    foreach (Segment1D segmentW in segmentsW)
                    {
                        Segment1D inputSegmentW = new Segment1D(0..inputShape[3], Padding.Zero());
                        SegmentND outputSegment = new SegmentND(segmentN, segmentC, segmentH, segmentW);
                        SegmentND inputSegment = new SegmentND(inputSegmentN, inputSegmentC, inputSegmentH, inputSegmentW);
                        Ai2dConfig config = new Ai2dConfig();
                        TileUtilities.Ai2dUtilKpuUpdateStaticParam(ld, st, resize, ref config);
                        TileUtilities.Ai2dUtilKpuUpdateResizeParam(ld, st, resize, ref config);
                        List<int> sramChunkSizes = new List<int> { 0, 0 };
                        TileUtilities.Ai2dUtilResizeSramSearch(config, outputSegment, inputSegment, sramChunkSizes);
                        int chunkSize = ((inputDataType == DataTypes.Int16) ? 2 : 4);
                        List<Segment1D> chunksC =
                            TileUtilities.GetSegmentStartEndLength(segmentC.Start, chunkSize, segmentC.End);
                        List<Segment1D> chunksH =
                            TileUtilities.GetSegmentStartEndLength(segmentH.Start, sramChunkSizes[0], segmentH.End);
                        List<Segment1D> chunksW =
                            TileUtilities.GetSegmentStartEndLength(segmentW.Start, sramChunkSizes[1], segmentW.End);
                        if (!recStat)
                        {
                            List<CcrSet> ccrsToSet = new List<CcrSet>
                            {
                                new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, pingPongIndex)),
                                    (chunksC.Count * chunksH.Count *
                                        chunksW.Count <= 1)
                                        ? 1
                                        : 2)
                            };
                            int offsetD = 0;
                            List<int> stridesD = new List<int>
                            {
                                inputSegment[1].Length, inputSegment[2].Length, inputSegment[3].Length
                            };
                            gnneActionUpdater.UpdateLoadIf(inputSegment, ld, pingPongIndex, ddrIf, offsetD, stridesD,
                                ItemName.Ifmap, ccrsToSet);
                        }

                        List<CcrClr> pendingCcrClr = new List<CcrClr>();
                        int pendingInstrBytes = 0;
                        foreach (Segment1D chunkC in chunksC)
                        {
                            bool isFirstWrite = true;
                            foreach (Segment1D chunkH in chunksH)
                            {
                                foreach (Segment1D chunkW in chunksW)
                                {
                                    List<CcrSet> ofmapCcrsToSet;
                                    bool writeAll;
                                    if (!recStat)
                                    {
                                        TileUtilities.Ai2dUtilKpuUpdateResizeParam(ld, st, resize, ref config);
                                        List<float> startPoint = new List<float> { chunkW.Start, chunkH.Start };
                                        List<float> endPoint = new List<float> { chunkW.End - 1, chunkH.End - 1 };
                                        List<float> mScale = new List<float>
                                        {
                                            config.OriginM(config.M0),
                                            config.OriginM(config.M1),
                                            config.OriginM(config.M3),
                                            config.OriginM(config.M4)
                                        };
                                        List<float> mBias = new List<float>
                                        {
                                            config.OriginM(config.M2), config.OriginM(config.M5)
                                        };
                                        List<float> startMapped = TileUtilities.Ai2dUtilMMulAdd(mScale, mBias, startPoint);
                                        List<float> endMapped = TileUtilities.Ai2dUtilMMulAdd(mScale, mBias, endPoint);
                                        int srcStartW = Math.Max((int)Math.Floor(startMapped[0]), 0);
                                        int srcStartH = Math.Max((int)Math.Floor(startMapped[1]), 0);
                                        int srcEndW = Math.Min((int)Math.Ceiling(endMapped[0]), inputShape[3] - 1);
                                        int srcEndH = Math.Min((int)Math.Ceiling(endMapped[1]), inputShape[2] - 1);
                                        List<float> originPoint = new List<float> { 0f, 0f };
                                        List<float> originMapped = TileUtilities.Ai2dUtilMMulAdd(mScale, mBias, originPoint);
                                        float offsetMW = 0f;
                                        if (srcStartW != 0)
                                        {
                                            offsetMW = startMapped[0] - (float)srcStartW - originMapped[0];
                                        }

                                        float offsetMH = 0f;
                                        if (srcStartH != 0)
                                        {
                                            offsetMH = startMapped[1] - (float)srcStartH - originMapped[1];
                                        }

                                        int srcWidth = srcEndW - srcStartW + 1;
                                        int srcHeight = srcEndH - srcStartH + 1;
                                        Segment1D srcSegmentH = new Segment1D(srcStartH..(srcStartH + srcHeight), Padding.Zero());
                                        Segment1D srcSegmentW = new Segment1D(srcStartW..(srcStartW + srcWidth), Padding.Zero());
                                        SegmentND ifmapSram = new SegmentND(inputSegment[0], chunkC, srcSegmentH,
                                            srcSegmentW);
                                        SegmentND outputChunk = new SegmentND(outputSegment[0], chunkC, chunkH, chunkW);
                                        TileUtilities.Ai2dUtilKpuUpdateDynamicParam(glb, ref config, ifmapSram,
                                            outputChunk, pingPongIndex, inputSegment, outputSegment, offsetMW, offsetMH,
                                            inputDataType, outputDataType);
                                        int releaseIfmap = 0;
                                        if (TileUtilities.IsFirstSlice(outputChunk, outputSegment) ||
                                            TileUtilities.IsLastSlice(outputChunk, outputSegment))
                                        {
                                            releaseIfmap = 1;
                                        }

                                        if (releaseIfmap == 1)
                                        {
                                            pendingCcrClr.Add(new CcrClr(
                                                ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, pingPongIndex))));
                                        }

                                        int setOfmapCcr = 0;
                                        int clearOfmapFake = 0;
                                        Tuple<SegmentND, TensorStat> sliceRecord = _r2GOfRec[pingPongIndex][0];
                                        _r2GOfRec[pingPongIndex].RemoveAt(0);
                                        if (sliceRecord.Item2.IsFirstSlice && !_ofmapRec[pingPongIndex][0].Item2.IsFirstSlice)
                                        {
                                            clearOfmapFake = 1;
                                        }

                                        if (sliceRecord.Item2.IsLastSlice)
                                        {
                                            setOfmapCcr = 1;
                                        }

                                        if (clearOfmapFake > 0)
                                        {
                                            pendingCcrClr.Add(new CcrClr(
                                                ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.OfmapFake, pingPongIndex))));
                                        }

                                        ofmapCcrsToSet = new List<CcrSet>();
                                        if (setOfmapCcr > 0)
                                        {
                                            ofmapCcrsToSet.Add(new CcrSet(
                                                ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, pingPongIndex)), 1));
                                        }

                                        writeAll = false;
                                        if (isFirstWrite)
                                        {
                                            pendingInstrBytes += 144;
                                            isFirstWrite = false;
                                            writeAll = true;
                                        }
                                        else
                                        {
                                            pendingInstrBytes += 32;
                                        }

                                        int isLastChunkHW;
                                        if (chunkH == chunksH[chunksH.Count - 1])
                                        {
                                            isLastChunkHW = ((chunkW == chunksW[chunksW.Count - 1])
                                                ? 1
                                                : 0);
                                        }
                                        else
                                        {
                                            isLastChunkHW = 0;
                                        }

                                        int projectedInstrBytes = pendingInstrBytes;
                                        if (isLastChunkHW == 0)
                                        {
                                            projectedInstrBytes += 32;
                                        }
                                        else if (chunkC != chunksC[chunksC.Count - 1])
                                        {
                                            projectedInstrBytes += 144;
                                        }

                                        if (projectedInstrBytes > 1024)
                                        {
                                            pendingInstrBytes = 0;
                                            config.IntrMask = 0u;
                                        }
                                        else
                                        {
                                            if (chunkC == chunksC[chunksC.Count - 1])
                                            {
                                                if (chunkH == chunksH[chunksH.Count - 1])
                                                {
                                                    if (chunkW == chunksW[
                                                            chunksW.Count - 1])
                                                    {
                                                        config.IntrMask = 0u;
                                                        goto EmitAi2dResize;
                                                    }
                                                }
                                            }

                                            config.IntrMask = 1u;
                                        }

                                        goto EmitAi2dResize;
                                    }

                                    _r2GOfRec[pingPongIndex].Add(new Tuple<SegmentND, TensorStat>(outputSegment,
                                        new TensorStat(isFirstSlice: false, isLastSlice: false)));
                                    continue;
                                    EmitAi2dResize:
                                    gnneActionUpdater.UpdateAi2dResize(config, ofmapCcrsToSet, pendingCcrClr, writeAll);
                                    if (config.IntrMask == 0)
                                    {
                                        pendingCcrClr.Clear();
                                    }
                                }
                            }
                        }

                        if (!recStat)
                        {
                            bool needsOfmapFake = !_ofmapRec[pingPongIndex][0].Item2.IsLastSlice;
                            _ofmapRec[pingPongIndex].RemoveAt(0);
                            List<CcrSet> ofmapFakeCcrsToSet = new List<CcrSet>();
                            if ((needsOfmapFake ? 1 : 0) > (false ? 1 : 0))
                            {
                                ofmapFakeCcrsToSet.Add(new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.OfmapFake, pingPongIndex)),
                                    1));
                            }

                            List<CcrClr> ccrsToClr = new List<CcrClr>
                            {
                                new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, pingPongIndex)))
                            };
                            gnneActionUpdater.UpdateStoreT(outputSegment, st, pingPongIndex, ddrOf, 0, null, ofmapFakeCcrsToSet, ccrsToClr);
                        }
                        else
                        {
                            _ofmapRec[pingPongIndex].Add(new Tuple<SegmentND, TensorStat>(outputSegment,
                                new TensorStat(isFirstSlice: false, isLastSlice: false)));
                        }

                        pingPongIndex = (pingPongIndex + 1) % 2;
                    }
                }
            }
        }

        TileUtilities.Assert(ccrHandler.CcrSanityCheck(), "ccrHandler.CcrSanityCheck()",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileAi2DResize.cs",
            303);
        return actions;
    }

    private TileAi2DResizeGlb SearchGlbParameters(Call ld, Call st)
    {
        GNNEShape inputShape = new GNNEShape(ld[GNNELoad.Input].CheckedShape.ToValueArray());
        GNNEShape outputShape = new GNNEShape(st.CheckedShape.ToValueArray());
        int batch = 1;
        int channels = Math.Min(4, outputShape[1]);
        int outHeight = outputShape[2];
        int outWidth = outputShape[3];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];
        AllocateResult allocateResult = HandleAllocate(ld, st, batch, channels, inputHeight, inputWidth, outHeight, outWidth);
        if (!allocateResult.IsOk)
        {
            channels = 1;
        }

        while (channels < outputShape[1])
        {
            int nextChannels = Math.Min(channels + 4, outputShape[1]);
            allocateResult = HandleAllocate(ld, st, batch, nextChannels, inputHeight, inputWidth, outHeight, outWidth);
            if (!allocateResult.IsOk)
            {
                break;
            }

            channels = nextChannels;
        }

        while (channels < outputShape[1])
        {
            int nextChannels = Math.Min(channels + 1, outputShape[1]);
            allocateResult = HandleAllocate(ld, st, batch, nextChannels, inputHeight, inputWidth, outHeight, outWidth);
            if (!allocateResult.IsOk)
            {
                break;
            }

            channels = nextChannels;
        }

        allocateResult = HandleAllocate(ld, st, batch, channels, inputHeight, inputWidth, outHeight, outWidth, isFinal: true);
        TileUtilities.Assert(allocateResult.IsOk, "allocation.IsOk",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileAi2DResize.cs",
            359);
        GNNEShape glbShape = new GNNEShape(batch, channels, outHeight, outWidth);
        return new TileAi2DResizeGlb(allocateResult.GlbMap, allocateResult.Items, glbShape.Dims,
            GNNEEnv.NPingPongSplit, allocateResult.GlbMap[ItemName.Ifmap], allocateResult.GlbMap[ItemName.Ofmap]);
    }

    private AllocateResult HandleAllocate(Call ld, Call st, int n, int c, int h, int w, int e, int f,
        bool isFinal = false)
    {
        BoxPacker boxPacker = new BoxPacker(16);
        Dictionary<ItemName, TensorOnGlb> glbMap = new Dictionary<ItemName, TensorOnGlb>();
        TileUtilities.GetBytesPerElement(ld.CheckedDataType);
        TensorOnGlb ifmapTensor = new TensorOnGlb(new int[4] { n, c, h, w }, ld.CheckedDataType, 0);
        int ifmapBytes = ifmapTensor.GlbNByte;
        ifmapBytes = (ifmapTensor.AllocatedBytes =
            TileUtilities.GetAlignedNum(ifmapBytes, GNNEEnv.OfmapBankWidth * GNNEEnv.GlbBankWidth));
        ifmapBytes *= GNNEEnv.NPingPongSplit;
        TensorOnGlb ofmapTensor = new TensorOnGlb(new int[4] { n, c, e, f }, st.CheckedDataType, 0);
        int ofmapBytes = ofmapTensor.GlbNByte;
        ofmapBytes = (ofmapTensor.AllocatedBytes =
            TileUtilities.GetAlignedNum(ofmapBytes, GNNEEnv.OfmapBankWidth * GNNEEnv.GlbBankWidth));
        ofmapBytes *= GNNEEnv.NPingPongSplit;
        int basementSize = SpaceSearcher.GetBasementSize();
        basementSize = TileUtilities.GetAlignedNum(basementSize, GNNEEnv.GlbWidth * GNNEEnv.GlbBankWidth);
        boxPacker.Boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.GlbWidth, basementSize / GNNEEnv.GlbWidth / GNNEEnv.GlbBankWidth },
            ItemName.Basement));
        boxPacker.Boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.IfmapBankWidth, ifmapBytes / GNNEEnv.IfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.Ifmap));
        boxPacker.Boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.OfmapBankWidth, ofmapBytes / GNNEEnv.OfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.Ofmap));
        AllocateResult allocateResult = SpaceSearcher.TryAllocate(boxPacker);
        if (allocateResult.IsOk)
        {
            ifmapTensor.Mmu = allocateResult.Items[ItemName.Ifmap];
            ofmapTensor.Mmu = allocateResult.Items[ItemName.Ofmap];
        }

        glbMap.Add(ItemName.Ifmap, ifmapTensor);
        glbMap.Add(ItemName.Ofmap, ofmapTensor);
        allocateResult.GlbMap = glbMap;
        return allocateResult;
    }

    private void ItemRecStatusInit()
    {
        for (int recIndex = 0; recIndex < _r2GOfRec.Length; recIndex++)
        {
            if (_r2GOfRec[recIndex] != null)
            {
                _r2GOfRec[recIndex].Clear();
            }
            else
            {
                _r2GOfRec[recIndex] = new List<Tuple<SegmentND, TensorStat>>();
            }
        }

        for (int ofmapIndex = 0; ofmapIndex < _ofmapRec.Length; ofmapIndex++)
        {
            if (_ofmapRec[ofmapIndex] != null)
            {
                _ofmapRec[ofmapIndex].Clear();
            }
            else
            {
                _ofmapRec[ofmapIndex] = new List<Tuple<SegmentND, TensorStat>>();
            }
        }
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Call call = (Call)__result["call"];
        Call ld = (Call)__result["ld"];
        Call st = (Call)__result["st"];
        return GetReplace(call, ld, st);
    }
}
