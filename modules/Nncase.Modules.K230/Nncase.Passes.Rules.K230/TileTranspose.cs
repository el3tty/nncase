// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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
public class TileTranspose : RewriteRule<Pattern>
{
    private static int _count = -1;

    public override Pattern Pattern { get; } = FusionPattern.IsGNNETransposeFusion();

    private PrimFunction GetReplace(Call call, GNNETranspose callOp, Call ld, Call st)
    {
        _count++;
        MFU_TRANS_PERMUTE perm = callOp.Perm;
        TileTransposeGlb glb = SearchGlbParameters(ld, st, perm);
        int[] inputShape = ld.Arguments[0].CheckedShape.ToValueArray();
        T.CreateBuffer(new TensorType(ld[GNNELoad.Input].CheckedDataType, inputShape), MemoryLocation.Input,
            out Buffer buffer, "var ddrIf");
        T.CreateBuffer(new TensorType(st.CheckedDataType, st.CheckedShape), MemoryLocation.Output, out Buffer buffer2,
            "var ddrOf");
        List<GnneAction> actions = BuildSchedule(glb, callOp, ld, st, buffer, buffer2);
        ActionToInstruct actionToInstruct = new ActionToInstruct();
        return T.PrimFunc($"TileTranspose_{_count}", K230RtModule.Kind, buffer, buffer2)
            .Body(actionToInstruct.Instructions(actions), I.END(GP_REGISTER.x0, 0)).Build();
    }

    private List<GnneAction> BuildSchedule(TiledGlb glb, GNNETranspose transpose, Call ld, Call st, Buffer ddrIf,
        Buffer ddrOf)
    {
        MFU_TRANS_PERMUTE perm = transpose.Perm;
        List<GnneAction> actions = new List<GnneAction>();
        GprHandler gpr = new GprHandler(GNNEEnv.GprNum);
        SsrHandler ssr = new SsrHandler(GNNEEnv.SsrNum);
        CcrHandler ccrHandler = new CcrHandler();
        GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, glb, ccrHandler, gpr, ssr);
        gnneActionUpdater.UpdateMmuConf();
        int[] lastOutShape = glb.LastOutShape;
        Shape inputShape = ld.CheckedShape;
        IEnumerable<SegmentND> ofmapSegments =
            from glbInputBatch in TileUtilities.SegmentBy(0, lastOutShape[0], inputShape[0].FixedValue)
            from glbInputChannel in TileUtilities.SegmentBy(0, lastOutShape[1], inputShape[1].FixedValue)
            from glbInputRow in TileUtilities.SegmentBy(0, lastOutShape[2], inputShape[2].FixedValue)
            from glbInputColumn in TileUtilities.SegmentBy(0, lastOutShape[3], inputShape[3].FixedValue)
            select new SegmentND(glbInputBatch, glbInputChannel, glbInputRow, glbInputColumn);
        int ifmapIndex = 0;
        foreach (SegmentND ifmapSegment in ofmapSegments)
        {
            SegmentND ofmapSegment = new SegmentND(GNNETypePatternUtility.ApplyPerm1(perm, ifmapSegment.ToArray()));
            List<CcrSet> ifmapCcrsToSet = new List<CcrSet>();
            ifmapCcrsToSet.Add(new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, ifmapIndex)), 1));
            SegmentND ifmap = ifmapSegment;
            List<int> stridesD = new int[3]
            {
                glb.GlbMap[ItemName.Ifmap].Dimensions[1], glb.GlbMap[ItemName.Ifmap].Dimensions[2],
                glb.GlbMap[ItemName.Ifmap].Dimensions[3],
            }.ToList();
            gnneActionUpdater.UpdateLoadIf(ifmap, ld, ifmapIndex, ddrIf, 0, stridesD, ItemName.Ifmap, ifmapCcrsToSet);
            List<CcrSet> ccrsToSet = new List<CcrSet>
            {
                new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, ifmapIndex)), 1),
            };
            List<CcrClr> ccrsToClr = new List<CcrClr>
            {
                new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, ifmapIndex))),
            };
            SegmentND ofmap = ofmapSegment;
            gnneActionUpdater.UpdateMfuTranspose(ifmap, ofmap, ld.CheckedDataType, perm, ifmapIndex, ccrsToSet, ccrsToClr);
            List<CcrClr> ccrsToClr2 = new List<CcrClr>
            {
                new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, ifmapIndex))),
            };
            gnneActionUpdater.UpdateStoreT(ofmap, st, ifmapIndex, ddrOf, 0, null, null, ccrsToClr2);
            ifmapIndex = (ifmapIndex + 1) % 2;
        }

        TileUtilities.Assert(ccrHandler.CcrSanityCheck(), "ccrHandler.CcrSanityCheck()",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileTranspose.cs",
            86);
        return actions;
    }

    private TileTransposeGlb SearchGlbParameters(Call ld, Call st, MFU_TRANS_PERMUTE perm)
    {
        GNNEShape shapeB = new GNNEShape(ld.Arguments[0].CheckedShape.ToValueArray());
        GNNEShape lastOutputShape = new GNNEShape(1, 1, 1, 1);
        for (int axis = 3; axis >= 0; axis--)
        {
            int i1 = axis;
            lastOutputShape[axis] = TileUtilities.SearchAxis(lastOutputShape, shapeB, axis,
                (int newV) => HandleAllocate(ld, st, perm, lastOutputShape.WithIndex(i1, newV)));
        }

        AllocateResult allocateResult = HandleAllocate(ld, st, perm, lastOutputShape, isFinal: true);
        TileUtilities.Assert(allocateResult.IsOk, "allocation.IsOk",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileTranspose.cs",
            105);
        return new TileTransposeGlb(allocateResult.GlbMap, allocateResult.Items, lastOutputShape.Dims,
            GNNEEnv.NPingPongSplit, allocateResult.GlbMap[ItemName.Ifmap], allocateResult.GlbMap[ItemName.Ofmap]);
    }

    private AllocateResult HandleAllocate(Call ld, Call st, MFU_TRANS_PERMUTE perm, GNNEShape shape,
        bool isFinal = false)
    {
        BoxPacker boxPacker = new BoxPacker(16);
        Dictionary<ItemName, TensorOnGlb> glbMap = new Dictionary<ItemName, TensorOnGlb>();
        GNNEShape permutedShape = new GNNEShape(GNNETypePatternUtility.ApplyPerm1(perm, shape.Dims));
        int alignmentFactor = (TileUtilities.GetBytesPerElement(ld.CheckedDataType) == 1) ? 32 : 16;
        int alignedNum = TileUtilities.GetAlignedNum(shape[3], alignmentFactor);
        TensorOnGlb ifmapTensor =
            new TensorOnGlb(new int[4] { shape[0], shape[1], shape[2], alignedNum }, ld.CheckedDataType, 0);
        int ifmapBytes = ifmapTensor.GlbNByte * GNNEEnv.NPingPongSplit;
        ifmapBytes = TileUtilities.GetAlignedNum(ifmapBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        int alignedNum2 = TileUtilities.GetAlignedNum(permutedShape[3], alignmentFactor);
        TensorOnGlb ofmapTensor = new TensorOnGlb(
            new int[4] { permutedShape[0], permutedShape[1], permutedShape[2], alignedNum2 },
            st.CheckedDataType, 0);
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

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Call call = (Call)result["call"];
        GNNETranspose callOp = (GNNETranspose)result["callOp"];
        Call ld = (Call)result["ld"];
        Call st = (Call)result["st"];
        return GetReplace(call, callOp, ld, st);
    }
}
