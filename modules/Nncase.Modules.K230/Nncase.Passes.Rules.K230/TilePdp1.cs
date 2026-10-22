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
public class TilePdp1 : RewriteRule<Pattern>
{
    private static int _count = -1;

    public override Pattern Pattern { get; } = FusionPattern.IsGNNEPDP1Fusion();

    private PrimFunction GetReplace(Call call, GNNEPdp1 callOp, Call ld, Call st)
    {
        _count++;
        TilePDP1Glb glb = SearchGlbParameters(ld, st, call);
        int[] inputShape = ld.Arguments[0].CheckedShape.ToValueArray();
        T.CreateBuffer(new TensorType(ld[GNNELoad.Input].CheckedDataType, inputShape), MemoryLocation.Input,
            out Nncase.TIR.Buffer buffer, "var ddrIf");
        T.CreateBuffer(new TensorType(st.CheckedDataType, st.CheckedShape), MemoryLocation.Output,
            out Nncase.TIR.Buffer buffer2, "var ddrOf");
        List<GnneAction> actions = (!IsGlobalPdp(ld, call))
            ? BuildSchedule(glb, call, ld, st, buffer, buffer2)
            : BuildScheduleGlobalPdp(glb, callOp, call, ld, st, buffer, buffer2);
        ActionToInstruct actionToInstruct = new ActionToInstruct();
        return T.PrimFunc($"TilePDP1_{_count}", K230RtModule.Kind, buffer, buffer2)
            .Body(actionToInstruct.Instructions(actions), I.END(GP_REGISTER.x0, 0)).Build();
    }

    private List<GnneAction> BuildSchedule(TiledGlb glb, Call pdp, Call ld, Call st, Nncase.TIR.Buffer ddrIf,
        Nncase.TIR.Buffer ddrOf)
    {
        int[] subArray = ((TensorConst)pdp[GNNEPdp1.Padding]).Value.ToArray<int>()[..2];
        int[] subArray2 = ((TensorConst)pdp[GNNEPdp1.Padding]).Value.ToArray<int>()[2..4];
        int u = ((TensorConst)pdp[GNNEPdp1.Stride]).Value.ToArray<int>()[0];
        int u2 = ((TensorConst)pdp[GNNEPdp1.Stride]).Value.ToArray<int>()[1];
        int r = ((TensorConst)pdp[GNNEPdp1.Filter]).Value.ToArray<int>()[0];
        int s = ((TensorConst)pdp[GNNEPdp1.Filter]).Value.ToArray<int>()[1];
        List<GnneAction> actions = new List<GnneAction>();
        GprHandler gpr = new GprHandler(GNNEEnv.GprNum);
        SsrHandler ssr = new SsrHandler(GNNEEnv.SsrNum);
        CcrHandler ccrHandler = new CcrHandler();
        GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, glb, ccrHandler, gpr, ssr);
        gnneActionUpdater.UpdateMmuConf();
        GNNEShape inputShape = new GNNEShape(ld.Arguments[0].CheckedShape.ToValueArray());
        int[] lastOutShape = glb.LastOutShape;
        Shape outShape = pdp.CheckedShape;
        IEnumerable<SegmentND> ofmapSegments =
            from glbOutputBatch in TileUtilities.SegmentBy(0, lastOutShape[0], outShape[0].FixedValue)
            from glbOutputChannel in TileUtilities.SegmentBy(0, lastOutShape[1], outShape[1].FixedValue)
            from glbOutputRow in TileUtilities.SegmentBy(0, lastOutShape[2], outShape[2].FixedValue)
            from glbOutputColumn in TileUtilities.SegmentBy(0, lastOutShape[3], outShape[3].FixedValue)
            select new SegmentND(glbOutputBatch, glbOutputChannel, glbOutputRow, glbOutputColumn);
        int ifmapIndex = 0;
        foreach (SegmentND ofmapSegment in ofmapSegments)
        {
            Segment1D inputRowSegment = TileUtilities.GetInputRowSegment(ofmapSegment[2].Start, ofmapSegment[2].Length, inputShape[2], r,
                u, 1, new Padding(subArray[0], subArray[1]));
            Segment1D inputColumnSegment = TileUtilities.GetInputColumnSegment(ofmapSegment[3].Start, ofmapSegment[3].Length,
                inputShape[3], s, u2, 1, new Padding(subArray2[0], subArray2[1]));
            SegmentND ifmapSegment = new SegmentND(ofmapSegment[0], ofmapSegment[1], inputRowSegment, inputColumnSegment);
            List<CcrSet> ccrsToSet = new List<CcrSet>
            {
                new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, ifmapIndex)), 1),
            };
            SegmentND ifmap = ifmapSegment;
            List<int> stridesD = new int[3]
            {
                glb.GlbMap[ItemName.Ifmap].Dimensions[1], glb.GlbMap[ItemName.Ifmap].Dimensions[2],
                glb.GlbMap[ItemName.Ifmap].Dimensions[3],
            }.ToList();
            gnneActionUpdater.UpdateLoadIf(ifmap, ld, ifmapIndex, ddrIf, 0, stridesD, ItemName.Ifmap, ccrsToSet);
            List<CcrSet> ccrsToSet2 = new List<CcrSet>
            {
                new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, ifmapIndex)), 1),
            };
            List<CcrClr> ccrsToClr = new List<CcrClr>
            {
                new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, ifmapIndex))),
            };
            SegmentND ofmap = ofmapSegment;
            gnneActionUpdater.UpdateMfuPdp1(pdp, ifmap, ofmap, ifmapIndex, ccrsToSet2, ccrsToClr);
            List<CcrClr> ccrsToClr2 = new List<CcrClr>
            {
                new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, ifmapIndex))),
            };
            gnneActionUpdater.UpdateStoreT(ofmap, st, ifmapIndex, ddrOf, 0, null, null, ccrsToClr2);
            ifmapIndex = (ifmapIndex + 1) % 2;
        }

        TileUtilities.Assert(ccrHandler.CcrSanityCheck(), "ccrHandler.CcrSanityCheck()",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TilePdp1.cs",
            104);
        return actions;
    }

    private List<GnneAction> BuildScheduleGlobalPdp(TiledGlb glb, GNNEPdp1 globalPdp, Call pdp, Call ld, Call st,
        Nncase.TIR.Buffer ddrIf, Nncase.TIR.Buffer ddrOf)
    {
        int[] subArray = ((TensorConst)pdp[GNNEPdp1.Padding]).Value.ToArray<int>()[..2];
        int[] subArray2 = ((TensorConst)pdp[GNNEPdp1.Padding]).Value.ToArray<int>()[2..4];
        int u = ((TensorConst)pdp[GNNEPdp1.Stride]).Value.ToArray<int>()[0];
        int u2 = ((TensorConst)pdp[GNNEPdp1.Stride]).Value.ToArray<int>()[1];
        int filterH = ((TensorConst)pdp[GNNEPdp1.Filter]).Value.ToArray<int>()[0];
        int filterW = ((TensorConst)pdp[GNNEPdp1.Filter]).Value.ToArray<int>()[1];
        List<GnneAction> actions = new List<GnneAction>();
        GprHandler gpr = new GprHandler(GNNEEnv.GprNum);
        SsrHandler ssr = new SsrHandler(GNNEEnv.SsrNum);
        CcrHandler ccrHandler = new CcrHandler();
        GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, glb, ccrHandler, gpr, ssr);
        gnneActionUpdater.UpdateMmuConf();
        int s = 0;
        int r = 0;
        SplitGlobalPdp1(filterW, filterH, ref r, ref s);
        GNNEShape inputShape = new GNNEShape(ld.Arguments[0].CheckedShape.ToValueArray());
        int[] lastOutShape = glb.LastOutShape;
        Shape outShape = pdp.CheckedShape;
        SegmentND[] ofmapSegments = (from glbOutputBatch in TileUtilities.SegmentBy(0, lastOutShape[0], outShape[0].FixedValue)
            from glbOutputChannel in TileUtilities.SegmentBy(0, lastOutShape[1], outShape[1].FixedValue)
            from glbOutputRow in TileUtilities.SegmentBy(0, lastOutShape[2], outShape[2].FixedValue)
            from glbOutputColumn in TileUtilities.SegmentBy(0, lastOutShape[3], outShape[3].FixedValue)
            select new SegmentND(glbOutputBatch, glbOutputChannel, glbOutputRow, glbOutputColumn)).ToArray();
        List<SegmentND> ifmapSegments = new List<SegmentND>();
        SegmentND[] ofmapSegmentsCopy = ofmapSegments;
        foreach (SegmentND ofmapSegment in ofmapSegmentsCopy)
        {
            Segment1D inputRowSegment = TileUtilities.GetInputRowSegment(ofmapSegment[2].Start, ofmapSegment[2].Length,
                inputShape[2], filterH, u, 1, new Padding(subArray[0], subArray[1]));
            Segment1D inputColumnSegment = TileUtilities.GetInputColumnSegment(ofmapSegment[3].Start, ofmapSegment[3].Length,
                inputShape[3], filterW, u2, 1, new Padding(subArray2[0], subArray2[1]));
            SegmentND ifmapSegment = new SegmentND(ofmapSegment[0], ofmapSegment[1], inputRowSegment, inputColumnSegment);
            ifmapSegments.Add(ifmapSegment);
        }

        int ifmapIndex = 0;
        bool[] isFirstLoad = new bool[2] { true, true };
        bool storedPp0 = false;
        bool storedPp1 = false;
        for (int segmentIndex = 0; segmentIndex < ofmapSegments.Length; segmentIndex++)
        {
            SegmentND ofmapSlice = ofmapSegments.ToList()[segmentIndex];
            SegmentND ifmapSlice = ifmapSegments[segmentIndex];
            if (ofmapSegments.Length == 1)
            {
                storedPp0 = true;
            }
            else
            {
                if ((segmentIndex == ofmapSegments.Length - 1 || segmentIndex == ofmapSegments.Length - 2) && (segmentIndex & 1) == 0)
                {
                    storedPp0 = true;
                }

                if ((segmentIndex == ofmapSegments.Length - 1 || segmentIndex == ofmapSegments.Length - 2) && (segmentIndex & 1) == 1)
                {
                    storedPp1 = true;
                }
            }

            if (segmentIndex == ofmapSegments.Length - 2)
            {
                if (ifmapSegments[ifmapSegments.Count - 1][1].Length == 1)
                {
                    storedPp1 = true;
                }
            }

            List<Segment1D> rowSegments = TileUtilities.GetSegmentStartEndLength(0, r, inputShape[2]);
            List<Segment1D> colSegments = TileUtilities.GetSegmentStartEndLength(0, s, inputShape[3]);
            List<CcrSet> ifmapCcrsToSet = new List<CcrSet>();
            ifmapCcrsToSet.Add(new CcrSet(
                ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, ifmapIndex)),
                (rowSegments.Count * colSegments.Count == 1) ? 1 : 2));
            SegmentND ifmapLoadSegment = new SegmentND(ifmapSlice);
            int alignedNum = TileUtilities.GetAlignedNum(
                ifmapSlice[3].Length,
                (TileUtilities.GetBytesPerElement(ld.CheckedDataType) == 1) ? 32 : 16);
            SegmentND tensor = new SegmentND(ifmapSlice[0], ifmapSlice[1], ifmapSlice[2],
                new Segment1D(..alignedNum, new Padding(0, 0)));
            List<int> stridesD = new int[3]
            {
                glb.GlbMap[ItemName.Ifmap].Dimensions[1], glb.GlbMap[ItemName.Ifmap].Dimensions[2],
                glb.GlbMap[ItemName.Ifmap].Dimensions[3],
            }.ToList();
            gnneActionUpdater.UpdateLoadIf(ifmapLoadSegment, ld, ifmapIndex, ddrIf, 0, stridesD, ItemName.Ifmap, ifmapCcrsToSet);
            alignedNum = TileUtilities.GetAlignedNum(colSegments.Count, 16);
            SegmentND ofmapTensorSegment = new SegmentND(ofmapSlice[0], ofmapSlice[1],
                new Segment1D(..rowSegments.Count, new Padding(0, 0)),
                new Segment1D(..colSegments.Count, new Padding(0, 0)));
            SegmentND tensor2 = new SegmentND(ofmapSlice[0], ofmapSlice[1], ofmapTensorSegment[2],
                new Segment1D(..alignedNum, new Padding(0, 0)));
            PDP_FUNCTION pdpOp =
                PdpFunc((globalPdp.ReduceOp == MFU_PDP_OP.AVERAGE) ? MFU_PDP_OP.SUM : globalPdp.ReduceOp);
            Half sumScale = (Half)(1.0 / (double)inputShape[2]);
            Half sumScale2 = (Half)(1.0 / (double)inputShape[3]);
            for (int rowIndex = 0; rowIndex < rowSegments.Count; rowIndex++)
            {
                for (int colIndex = 0; colIndex < colSegments.Count; colIndex++)
                {
                    int isFirstWindow = (rowIndex == 0 && colIndex == 0) ? 1 : 0;
                    List<CcrClr> ifmapCcrsToClr = new List<CcrClr>();
                    if (isFirstWindow > 0)
                    {
                        ifmapCcrsToClr.Add(new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, ifmapIndex))));
                    }

                    if (rowIndex == 0 && colIndex == 0 && !isFirstLoad[ifmapIndex])
                    {
                        ifmapCcrsToClr.Add(new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.OfmapFake, ifmapIndex))));
                    }

                    SegmentND slice = new SegmentND(ifmapLoadSegment[0], ifmapLoadSegment[1], rowSegments[rowIndex],
                        colSegments[colIndex]);
                    SegmentND slice2 = new SegmentND(ofmapTensorSegment[0], ofmapTensorSegment[1],
                        new Segment1D(rowIndex..(rowIndex + 1), new Padding(0, 0)),
                        new Segment1D(colIndex..(colIndex + 1), new Padding(0, 0)));
                    int offsetS = TileUtilities.GetSliceOffsetInTensor(in tensor, in slice) *
                                  TileUtilities.GetBytesPerElement(ld.CheckedDataType);
                    int offsetD = TileUtilities.GetSliceOffsetInTensor(in tensor2, in slice2) *
                                  TileUtilities.GetBytesPerElement(DataTypes.Float16);
                    gnneActionUpdater.UpdateMfuGlobalPdp1(pdp, ld.CheckedDataType, DataTypes.Float16, pdpOp, slice,
                        slice2, ifmapIndex, sumScale, null, ifmapCcrsToClr, offsetS, offsetD);
                }
            }

            SegmentND ifmap = ofmapTensorSegment;
            SegmentND ofmap = new SegmentND(ofmapTensorSegment[0], ofmapTensorSegment[1], new Segment1D(..1, new Padding(0, 0)),
                new Segment1D(..1, new Padding(0, 0)));
            List<CcrSet> pdpCcrsToSet = new List<CcrSet>();
            pdpCcrsToSet.Add(new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, ifmapIndex)), 1));
            List<CcrClr> pdpCcrsToClr = new List<CcrClr>();
            pdpCcrsToClr.Add(new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, ifmapIndex))));
            int offsetS2 = 0;
            int offsetD2 = 0;
            gnneActionUpdater.UpdateMfuGlobalPdp1(pdp, DataTypes.Float16, pdp.CheckedDataType, pdpOp, ifmap, ofmap,
                ifmapIndex, sumScale2, pdpCcrsToSet, pdpCcrsToClr, offsetS2, offsetD2, ItemName.Ofmap);
            List<CcrClr> ccrsToClr = new List<CcrClr>
            {
                new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, ifmapIndex))),
            };
            bool notStored = (ifmapIndex == 0) ? (!storedPp0) : (!storedPp1);
            List<CcrSet> storeCcrsToSet = new List<CcrSet>();
            if ((notStored ? 1 : 0) > (false ? 1 : 0))
            {
                storeCcrsToSet.Add(new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.OfmapFake, ifmapIndex)), 1));
            }

            List<int> stridesS = new int[3]
            {
                glb.GlbMap[ItemName.Ofmap].Dimensions[1], glb.GlbMap[ItemName.Ofmap].Dimensions[2],
                glb.GlbMap[ItemName.Ofmap].Dimensions[3] *
                ((pdp.CheckedDataType == DataTypes.Float16 || pdp.CheckedDataType == DataTypes.Int16) ? 1 : 2),
            }.ToList();
            gnneActionUpdater.UpdateStoreT(ofmap, st, ifmapIndex, ddrOf, 0, null, storeCcrsToSet, ccrsToClr, ItemName.Ofmap, stridesS);
            isFirstLoad[ifmapIndex] = false;
            ifmapIndex = (ifmapIndex + 1) % 2;
        }

        TileUtilities.Assert(ccrHandler.CcrSanityCheck(), "ccrHandler.CcrSanityCheck()",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TilePdp1.cs",
            270);
        return actions;

        static PDP_FUNCTION PdpFunc(MFU_PDP_OP op)
        {
            if (op <= MFU_PDP_OP.SUM)
            {
                switch ((uint)op)
                {
                    case 1u:
                        return PDP_FUNCTION.min;
                    case 0u:
                        return PDP_FUNCTION.max;
                    case 2u:
                        return PDP_FUNCTION.average;
                    case 3u:
                        return PDP_FUNCTION.sum;
                }
            }

            return PDP_FUNCTION.min;
        }
    }

    private TilePDP1Glb SearchGlbParameters(Call ld, Call st, Call pdp)
    {
        int[] subArray = ((TensorConst)pdp[GNNEPdp1.Padding]).Value.ToArray<int>()[..2];
        int[] subArray2 = ((TensorConst)pdp[GNNEPdp1.Padding]).Value.ToArray<int>()[2..4];
        int uh = ((TensorConst)pdp[GNNEPdp1.Stride]).Value.ToArray<int>()[0];
        int uh2 = ((TensorConst)pdp[GNNEPdp1.Stride]).Value.ToArray<int>()[1];
        int filterH = ((TensorConst)pdp[GNNEPdp1.Filter]).Value.ToArray<int>()[0];
        int filterW = ((TensorConst)pdp[GNNEPdp1.Filter]).Value.ToArray<int>()[1];
        GNNEShape inputShape = new GNNEShape(ld.Arguments[0].CheckedShape.ToValueArray());
        GNNEShape outputShape = new GNNEShape(pdp.CheckedShape.ToValueArray());
        bool isGlobalPdp = IsGlobalPdp(ld, pdp);
        int i = 1;
        int j = 1;
        int ofmapRows = 1;
        int ofmapCols = outputShape[3];
        int h = SpaceSearcher.GetInputHeight(ofmapRows, inputShape[2], filterH, outputShape[2], uh, 1,
            new Padding(subArray[0], subArray[1]));
        int inputHeight = SpaceSearcher.GetInputHeight(ofmapCols, inputShape[3], filterW, outputShape[3], uh2, 1,
            new Padding(subArray2[0], subArray2[1]));
        DataType inType = ld.CheckedDataType;
        DataType outType = isGlobalPdp ? DataTypes.Float16 : pdp.CheckedDataType;
        if (isGlobalPdp)
        {
            int s = 0;
            int r = 0;
            SplitGlobalPdp1(filterW, filterH, ref r, ref s);
            ofmapRows = (int)Math.Ceiling(1.0 * (double)filterH / (double)r);
            ofmapCols = (int)Math.Ceiling(1.0 * (double)filterW / (double)s);
            TileUtilities.Assert(
                ofmapCols <= 64 && ofmapRows <= 16 && ofmapRows * ofmapCols <= 256,
                "ow <= 64 && oh <= 16 && oh * ow <= 256",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TilePdp1.cs",
                308);
        }

        AllocateResult allocateResult = new AllocateResult();
        while (ofmapRows < outputShape[2])
        {
            int nextOfmapRows = ofmapRows + 1;
            int inputHeight2 = SpaceSearcher.GetInputHeight(nextOfmapRows, inputShape[2], filterH, outputShape[2], uh, 1,
                new Padding(subArray[0], subArray[1]));
            allocateResult = HandleAllocate(i, j, inputHeight2, inputHeight, ofmapRows, ofmapCols, inType, outType);
            if (!allocateResult.IsOk)
            {
                break;
            }

            ofmapRows = nextOfmapRows;
            h = inputHeight2;
        }

        for (; j < inputShape[1]; j++)
        {
            allocateResult = HandleAllocate(i, j + 1, h, inputHeight, ofmapRows, ofmapCols, inType, outType);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; i < inputShape[0]; i++)
        {
            allocateResult = HandleAllocate(i + 1, j, h, inputHeight, ofmapRows, ofmapCols, inType, outType);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        allocateResult = HandleAllocate(i, j, h, inputHeight, ofmapRows, ofmapCols, inType, outType, isFinal: true);
        TileUtilities.Assert(allocateResult.IsOk, "allocation.IsOk",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TilePdp1.cs",
            358);
        if (isGlobalPdp)
        {
            ofmapRows = 1;
            ofmapCols = 1;
        }

        GNNEShape glbShape = new GNNEShape(i, j, ofmapRows, ofmapCols);
        return new TilePDP1Glb(allocateResult.GlbMap, allocateResult.Items, glbShape.Dims, GNNEEnv.NPingPongSplit,
            allocateResult.GlbMap[ItemName.Ifmap], allocateResult.GlbMap[ItemName.Ofmap]);
    }

    private AllocateResult HandleAllocate(int n, int c, int h, int w, int e, int f, DataType inType, DataType outType,
        bool isFinal = false)
    {
        BoxPacker boxPacker = new BoxPacker(16);
        Dictionary<ItemName, TensorOnGlb> glbMap = new Dictionary<ItemName, TensorOnGlb>();
        int alignmentFactor = (TileUtilities.GetBytesPerElement(inType) == 1) ? 32 : 16;
        int alignedNum = TileUtilities.GetAlignedNum(w, alignmentFactor);
        TensorOnGlb ifmapTensor = new TensorOnGlb(new int[4] { n, c, h, alignedNum }, inType, 0);
        int ifmapBytes = ifmapTensor.GlbNByte * GNNEEnv.NPingPongSplit;
        ifmapBytes = TileUtilities.GetAlignedNum(ifmapBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        alignmentFactor = (TileUtilities.GetBytesPerElement(outType) == 1) ? 32 : 16;
        int alignedNum2 = TileUtilities.GetAlignedNum(f, alignmentFactor);
        TensorOnGlb ofmapTensor = new TensorOnGlb(new int[4] { n, c, e, alignedNum2 }, outType, 0);
        int ofmapBytes = ofmapTensor.GlbNByte * GNNEEnv.NPingPongSplit;
        ofmapBytes = TileUtilities.GetAlignedNum(ofmapBytes, GNNEEnv.OfmapBankWidth * GNNEEnv.GlbBankWidth);
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

    private void SplitGlobalPdp1(int kernelW, int kernelH, ref int r, ref int s)
    {
        r = (kernelH > 16) ? 16 : kernelH;
        s = Math.Min(Math.Min(256 / r, kernelW), 64);
    }

    private bool IsGlobalPdp(Call ld, Call pdp)
    {
        int filterH = ((TensorConst)pdp[GNNEPdp1.Filter]).Value.ToArray<int>()[0];
        int filterW = ((TensorConst)pdp[GNNEPdp1.Filter]).Value.ToArray<int>()[1];
        GNNEShape inputShape = new GNNEShape(ld.Arguments[0].CheckedShape.ToValueArray());
        if (inputShape[2] == filterH && inputShape[3] == filterW)
        {
            if (inputShape[2] <= 16 && inputShape[3] <= 64)
            {
                return inputShape[2] * inputShape[3] > 256;
            }

            return true;
        }

        return false;
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Call call = (Call)result["call"];
        GNNEPdp1 callOp = (GNNEPdp1)result["callOp"];
        Call ld = (Call)result["ld"];
        Call st = (Call)result["st"];
        return GetReplace(call, callOp, ld, st);
    }
}
