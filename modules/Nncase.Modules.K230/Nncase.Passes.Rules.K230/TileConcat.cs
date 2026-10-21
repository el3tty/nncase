// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using DryIoc.ImTools;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using Nncase.Runtime.K230;
using Nncase.TIR;
using Nncase.TIR.Instructions;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class TileConcat : RewriteRule<Pattern>
{
    private static int _count = -1;

    private readonly CcrHandler _ccrHandler = new CcrHandler();

    private readonly GprHandler _gpr = new GprHandler();

    private readonly SsrHandler _ssr = new SsrHandler();

    private DataType[]? _inputTypes;

    private DataType? _outputType;

    private Call[]? _lifs;

    private Call? _sof;

    private GNNEShape[]? _inputShapes;

    private GNNEShape? _outputShape;

    private int _minInputIndex;

    private int _axis;

    public override Pattern Pattern { get; } = Nncase.PatternMatch.Utility.IsFusion("fusion", "k230",
        Nncase.PatternMatch.F.K230.IsGNNEStore("st", "stCall", (GNNEStore _) => true,
            Tensors.IsConcat("concat", "concatCall", (Concat _) => true,
                Nncase.PatternMatch.Utility.IsTuple("tuple",
                    Nncase.PatternMatch.Utility.IsVArgsRepeat("tupleInputs",
                        () => Nncase.PatternMatch.Utility.IsWildcard())))),
        Nncase.PatternMatch.Utility.IsVArgsRepeat(() => Nncase.PatternMatch.Utility.IsVar()));

    private PrimFunction GetReplace(IReadOnlyList<Expr> tupleInputs, Call stCall, Concat concat)
    {
        _count++;
        InitParameters(tupleInputs, stCall, concat);
        List<Nncase.TIR.Buffer> buffers = new List<Nncase.TIR.Buffer>();
        for (int i = 0; i < _lifs.Length; i++)
        {
            T.CreateBuffer(new TensorType(_lifs[i][GNNELoad.Input].CheckedDataType, _lifs[i].CheckedShape),
                MemoryLocation.Input, out Nncase.TIR.Buffer buffer, "ddrIf_" + i);
            buffers.Add(buffer);
        }

        T.CreateBuffer(new TensorType(stCall.CheckedDataType, stCall.CheckedShape), MemoryLocation.Output,
            out Nncase.TIR.Buffer buffer2, "var ddrOf");
        List<GnneAction> actions = BuildSchedule(buffers, buffer2);
        ActionToInstruct actionToInstruct = new ActionToInstruct();
        return T.PrimFunc($"TileConcat_{_count}", K230RtModule.Kind,
                buffers.Concat(new Nncase.TIR.Buffer[1] { buffer2 }).ToArray())
            .Body(actionToInstruct.Instructions(actions), I.END(GP_REGISTER.x0, 0)).Build();
    }

    private void InitParameters(IReadOnlyList<Expr> tupleInputs, Call stCall, Concat concat)
    {
        _lifs = tupleInputs.Select((Expr i) => (Call)i).ToArray();
        _sof = stCall;
        _inputTypes = _lifs.Select((Call i) => i.CheckedDataType).ToArray();
        _outputType = _sof.CheckedDataType;
        _inputShapes = _lifs.Select((Call i) => new GNNEShape(i.CheckedShape[0].FixedValue,
            i.CheckedShape[1].FixedValue, i.CheckedShape[2].FixedValue, i.CheckedShape[3].FixedValue)).ToArray();
        _outputShape = new GNNEShape(_sof.CheckedShape[0].FixedValue, _sof.CheckedShape[1].FixedValue,
            _sof.CheckedShape[2].FixedValue, _sof.CheckedShape[3].FixedValue);
        int minDim = _inputShapes.Min((GNNEShape s) => s.Dims[concat.Axis]);
        _minInputIndex = ArrayTools.IndexOf<GNNEShape>(_inputShapes,
            (Func<GNNEShape, bool>)((GNNEShape s) => s.Dims[concat.Axis] == minDim));
        _axis = concat.Axis;
    }

    private TileConcatGlb SearchGlbParameters(int index)
    {
        int i = 1;
        int j = 1;
        int k = 1;
        GNNEShape inputShape = _inputShapes[index];
        int w = inputShape[3];
        int width = inputShape[3];
        int nPingPongSplit = 1;
        AllocateResult allocateResult = HandleAllocate(i, j, k, w, k, width, nPingPongSplit);
        if (!allocateResult.IsOk)
        {
            nPingPongSplit = 1;
        }

        for (; k < inputShape[2] && k < 65535; k++)
        {
            int nextRows = k + 1;
            allocateResult = HandleAllocate(i, j, nextRows, w, nextRows, width, nPingPongSplit);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; j < inputShape[1] && j < 65535; j++)
        {
            allocateResult = HandleAllocate(i, j + 1, k, w, k, width, nPingPongSplit);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; i < inputShape[0] && i < 65535; i++)
        {
            allocateResult = HandleAllocate(i + 1, j, k, w, k, width, nPingPongSplit);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        allocateResult = HandleAllocate(i, j, k, w, k, width, nPingPongSplit, isFinal: true);
        TileUtilities.Assert(allocateResult.IsOk, "allocation.IsOk",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileConcat.cs",
            154);
        GNNEShape glbShape = new GNNEShape(i, j, k, width);
        return new TileConcatGlb(allocateResult.GlbMap, allocateResult.Items, glbShape.Dims, GNNEEnv.NPingPongSplit,
            allocateResult.GlbMap[ItemName.Ifmap], allocateResult.GlbMap[ItemName.Ofmap]);
    }

    private AllocateResult HandleAllocate(int n, int c, int h, int w, int e, int f, int nPingPongSplit,
        bool isFinal = false)
    {
        List<BoxOnGlb> boxes = new List<BoxOnGlb>();
        Dictionary<ItemName, TensorOnGlb> glbMap = new Dictionary<ItemName, TensorOnGlb>();
        int channelsPerPass = (int)System.Math.Ceiling(1.0 * (double)c / (double)nPingPongSplit);
        TensorOnGlb ifmapTensor = new TensorOnGlb(new int[4] { n, channelsPerPass, h, w }, _inputTypes[0], 0);
        int value = ifmapTensor.AllocatedBytes * nPingPongSplit;
        value = TileUtilities.GetAlignedNum(value, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        TensorOnGlb ofmapTensor = new TensorOnGlb(new int[4] { n, channelsPerPass, h, w }, _inputTypes[0], 0);
        int basementSize = SpaceSearcher.GetBasementSize();
        basementSize = TileUtilities.GetAlignedNum(basementSize, GNNEEnv.GlbWidth * GNNEEnv.GlbBankWidth);
        boxes.Add(new BoxOnGlb(new int[2] { GNNEEnv.GlbWidth, basementSize / GNNEEnv.GlbWidth / GNNEEnv.GlbBankWidth },
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

    private List<GnneAction> BuildSchedule(List<Nncase.TIR.Buffer> ddrIfs, Nncase.TIR.Buffer ddrOf)
    {
        List<GnneAction> actions = new List<GnneAction>();
        int[] strides = ((TensorConst)_sof[GNNEStore.Strides]).Value.ToArray<int>();
        Func<int, int, int, Segment1D> func = delegate(int start, int end, int dim)
        {
            int scaledStart = (int)System.Math.Ceiling(1.0 * (double)start / (double)strides[dim]);
            return new Segment1D(
                new System.Range(end: (int)System.Math.Ceiling(1.0 * (double)end / (double)strides[dim]), start: scaledStart),
                new Padding(0, 0));
        };
        int concatOffset = 0;
        int ifmapIndex = 0;
        for (int inputIndex = 0; inputIndex < _lifs.Length; inputIndex++)
        {
            GprHandler gpr = new GprHandler(GNNEEnv.GprNum);
            SsrHandler ssr = new SsrHandler(GNNEEnv.SsrNum);
            CcrHandler ccrHandler = new CcrHandler();
            TiledGlb tiledGlb = SearchGlbParameters(inputIndex);
            GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, tiledGlb, ccrHandler, gpr, ssr);
            gnneActionUpdater.UpdateMmuConf();
            foreach (Segment1D segmentN in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[0],
                         _inputShapes[inputIndex][0]))
            {
                Segment1D shiftedN = func(segmentN.Start, segmentN.End, 0);
                if (_axis == 0)
                {
                    shiftedN = func(shiftedN.Start + concatOffset, shiftedN.End + concatOffset, 0);
                }

                foreach (Segment1D segmentC in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[1],
                             _inputShapes[inputIndex][1]))
                {
                    Segment1D shiftedC = func(segmentC.Start, segmentC.End, 1);
                    if (_axis == 1)
                    {
                        shiftedC = func(shiftedC.Start + concatOffset, shiftedC.End + concatOffset, 1);
                    }

                    foreach (Segment1D segmentH in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[2],
                                 _inputShapes[inputIndex][2]))
                    {
                        Segment1D shiftedH = func(segmentH.Start, segmentH.End, 2);
                        if (_axis == 2)
                        {
                            shiftedH = func(shiftedH.Start + concatOffset, shiftedH.End + concatOffset, 2);
                        }

                        foreach (Segment1D segmentW in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[3],
                                     _inputShapes[inputIndex][3]))
                        {
                            Segment1D shiftedW = func(segmentW.Start, segmentW.End, 3);
                            if (_axis == 3)
                            {
                                shiftedW = func(shiftedW.Start + concatOffset, shiftedW.End + concatOffset, 3);
                            }

                            SegmentND shiftedSegment = new SegmentND(shiftedN, shiftedC, shiftedH, shiftedW);
                            SegmentND slice = new SegmentND(shiftedN, segmentC, segmentH, segmentW);
                            List<CcrSet> ccrsToSet = new List<CcrSet>
                            {
                                new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, ifmapIndex)), 1)
                            };
                            SegmentND tensor = slice;
                            List<int> storeStrides =
                                new int[3] { tensor[1].Length, tensor[2].Length, tensor[3].Length }.ToList();
                            gnneActionUpdater.UpdateLoadIf(tensor, _lifs[inputIndex], ifmapIndex, ddrIfs[inputIndex], 0, storeStrides,
                                ItemName.Ifmap, ccrsToSet);
                            List<CcrClr> ccrsToClr = new List<CcrClr>
                            {
                                new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, ifmapIndex)))
                            };
                            SegmentND ofmapSegment = shiftedSegment;
                            SegmentND storeSegment = shiftedSegment;
                            List<int> stridesD =
                                new int[3] { _outputShape[1], _outputShape[2], _outputShape[3] }.ToList();
                            storeSegment[0] = new Segment1D((storeSegment[0].Start * strides[0])..storeSegment[0].End,
                                new Padding(0, 0));
                            storeSegment[1] = new Segment1D((storeSegment[1].Start * strides[1])..storeSegment[1].End,
                                new Padding(0, 0));
                            storeSegment[2] = new Segment1D((storeSegment[2].Start * strides[2])..storeSegment[2].End,
                                new Padding(0, 0));
                            storeSegment[3] = new Segment1D((storeSegment[3].Start * strides[3])..storeSegment[3].End,
                                new Padding(0, 0));
                            int sliceOffsetInTensor = TileUtilities.GetSliceOffsetInTensor(in tensor, in slice);
                            GNNEShape shape = new GNNEShape(ofmapSegment[0].Length, ofmapSegment[1].Length,
                                ofmapSegment[2].Length, ofmapSegment[3].Length);
                            gnneActionUpdater.UpdateStoreT(ofmapSegment, _sof, ifmapIndex, ddrOf, sliceOffsetInTensor, null,
                                null, ccrsToClr, ItemName.Ofmap, storeStrides, stridesD, shape);
                        }
                    }
                }
            }

            concatOffset += _inputShapes[inputIndex][_axis];
            TileUtilities.Assert(ccrHandler.CcrSanityCheck(), "ccrHandler.CcrSanityCheck()",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileConcat.cs",
                301);
        }

        return actions;
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        IReadOnlyList<Expr> tupleInputs = (IReadOnlyList<Expr>)__result["tupleInputs"];
        Call stCall = (Call)__result["stCall"];
        Concat concat = (Concat)__result["concat"];
        return GetReplace(tupleInputs, stCall, concat);
    }
}
