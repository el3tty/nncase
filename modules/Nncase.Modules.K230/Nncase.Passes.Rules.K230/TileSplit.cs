// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Runtime.K230;
using Nncase.TIR;
using Nncase.TIR.Instructions;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class TileSplit : RewriteRule<Pattern>
{
    private static int _count = -1;

    private readonly CcrHandler _ccrHandler = new CcrHandler();

    private readonly GprHandler _gpr = new GprHandler();

    private readonly SsrHandler _ssr = new SsrHandler();

    private DataType? _inputType;

    private DataType[]? _outputTypes;

    private Call? _lif;

    private Call[]? _sofs;

    private GNNEShape? _inputShape;

    private GNNEShape[]? _outputShapes;

    private int _axis;

    public override Pattern Pattern { get; } = Nncase.PatternMatch.Utility.IsFusion("fusion", "k230",
        Nncase.PatternMatch.Utility.IsTuple(
            "tuple",
            Nncase.PatternMatch.Utility.IsVArgsRepeat("tupleInputs", () => Nncase.PatternMatch.Utility.IsWildcard())),
        Nncase.PatternMatch.Utility.IsVArgsRepeat(() => Nncase.PatternMatch.Utility.IsVar()));

    private PrimFunction GetReplace(IReadOnlyList<Expr> tupleInputs)
    {
        try
        {
            _count++;
            InitParameters(tupleInputs);
            T.CreateBuffer(new TensorType(_sofs[0].CheckedDataType, _lif.CheckedShape), MemoryLocation.Input,
                out Nncase.TIR.Buffer buffer, "var ddrIf");
            List<Nncase.TIR.Buffer> buffers = new List<Nncase.TIR.Buffer>();
            for (int i = 0; i < _sofs.Length; i++)
            {
                T.CreateBuffer(new TensorType(_sofs[i].CheckedDataType, _sofs[i].CheckedShape), MemoryLocation.Output,
                    out Nncase.TIR.Buffer buffer2, "ddrOf_" + i);
                buffers.Add(buffer2);
            }

            List<GnneAction> actions = BuildSchedule(buffer, buffers);
            ActionToInstruct actionToInstruct = new ActionToInstruct();
            return T.PrimFunc($"TileSplit_{_count}", K230RtModule.Kind,
                    new Nncase.TIR.Buffer[1] { buffer }.Concat(buffers).ToArray())
                .Body(actionToInstruct.Instructions(actions), I.END(GP_REGISTER.x0, 0)).Build();
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex.Message);
            return null;
        }
    }

    private void InitParameters(IReadOnlyList<Expr> tupleInputs)
    {
        _sofs = tupleInputs.Select((Expr i) => (Call)i).ToArray();
        Call call = (Call)((Call)_sofs[0][GNNEStore.Input])[GetItem.Input];
        _lif = (Call)call[Split.Input];
        _inputType = _lif.CheckedDataType;
        _outputTypes = _sofs.Select((Call i) => i.CheckedDataType).ToArray();
        _inputShape = new GNNEShape(_lif.CheckedShape[0].FixedValue, _lif.CheckedShape[1].FixedValue,
            _lif.CheckedShape[2].FixedValue, _lif.CheckedShape[3].FixedValue);
        _outputShapes = _sofs.Select((Call i) => new GNNEShape(
            i.CheckedShape[0].FixedValue,
            i.CheckedShape[1].FixedValue, i.CheckedShape[2].FixedValue, i.CheckedShape[3].FixedValue)).ToArray();
        _ = _inputShape.Dims[0];
        _axis = ((TensorConst)call[Split.Axis]).Value.ToScalar<int>();
    }

    private TileSplitGlb SearchGlbParameters(int index)
    {
        int i = 1;
        int j = 1;
        int k = 1;
        GNNEShape outputShape = _outputShapes[index];
        int w = outputShape[3];
        int width = outputShape[3];
        int nPingPongSplit = 1;
        AllocateResult allocateResult = HandleAllocate(i, j, k, w, k, width, nPingPongSplit);
        if (!allocateResult.IsOk)
        {
            nPingPongSplit = 1;
        }

        for (; k < outputShape[2] && k < 65535; k++)
        {
            int nextRows = k + 1;
            allocateResult = HandleAllocate(i, j, nextRows, w, nextRows, width, nPingPongSplit);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; j < outputShape[1] && j < 65535; j++)
        {
            allocateResult = HandleAllocate(i, j + 1, k, w, k, width, nPingPongSplit);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; i < outputShape[0] && i < 65535; i++)
        {
            allocateResult = HandleAllocate(i + 1, j, k, w, k, width, nPingPongSplit);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        allocateResult = HandleAllocate(i, j, k, w, k, width, nPingPongSplit, isFinal: true);
        TileUtilities.Assert(allocateResult.IsOk, "allocation.IsOk",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileSplit.cs",
            154);
        GNNEShape glbShape = new GNNEShape(i, j, k, width);
        return new TileSplitGlb(allocateResult.GlbMap, allocateResult.Items, glbShape.Dims, GNNEEnv.NPingPongSplit,
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

    private List<GnneAction> BuildSchedule(Nncase.TIR.Buffer ddrIf, List<Nncase.TIR.Buffer> ddrOfs)
    {
        List<GnneAction> actions = new List<GnneAction>();
        int[] strides = ((TensorConst)_sofs[0][GNNEStore.Strides]).Value.ToArray<int>();
        Func<int, int, int, Segment1D> func = (start, end, dim) =>
        {
            int scaledStart = (int)Math.Ceiling(1.0 * (double)start / (double)strides[dim]);
            return new Segment1D(
                new System.Range(end: (int)Math.Ceiling(1.0 * (double)end / (double)strides[dim]), start: scaledStart),
                new Padding(0, 0));
        };
        int splitOffset = 0;
        int ifmapIndex = 0;
        for (int outputIndex = 0; outputIndex < _sofs.Length; outputIndex++)
        {
            GprHandler gpr = new GprHandler(GNNEEnv.GprNum);
            SsrHandler ssr = new SsrHandler(GNNEEnv.SsrNum);
            CcrHandler ccrHandler = new CcrHandler();
            TiledGlb tiledGlb = SearchGlbParameters(outputIndex);
            GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, tiledGlb, ccrHandler, gpr, ssr);
            gnneActionUpdater.UpdateMmuConf();
            foreach (Segment1D segmentN in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[0],
                         _outputShapes[outputIndex][0]))
            {
                func(segmentN.Start, segmentN.End, 0);
                if (_axis == 0)
                {
                    func(segmentN.Start + splitOffset, segmentN.End + splitOffset, 0);
                }

                foreach (Segment1D segmentC in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[1],
                             _outputShapes[outputIndex][1]))
                {
                    Segment1D shiftedC = func(segmentC.Start, segmentC.End, 1);
                    if (_axis == 1)
                    {
                        shiftedC = func(segmentC.Start + splitOffset, segmentC.End + splitOffset, 1);
                    }

                    foreach (Segment1D segmentH in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[2],
                                 _outputShapes[outputIndex][2]))
                    {
                        Segment1D shiftedH = func(segmentH.Start, segmentH.End, 2);
                        if (_axis == 2)
                        {
                            shiftedH = func(segmentH.Start + splitOffset, segmentH.End + splitOffset, 2);
                        }

                        foreach (Segment1D segmentW in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[3],
                                     _outputShapes[outputIndex][3]))
                        {
                            Segment1D shiftedW = func(segmentW.Start, segmentW.End, 3);
                            if (_axis == 3)
                            {
                                shiftedW = func(segmentW.Start + splitOffset, segmentW.End + splitOffset, 3);
                            }

                            SegmentND ofmapSegment = new SegmentND(segmentN, segmentC, segmentH, segmentW);
                            SegmentND slice = new SegmentND(segmentN, shiftedC, shiftedH, shiftedW);
                            List<CcrSet> ccrsToSet = new List<CcrSet>
                            {
                                new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, ifmapIndex)), 1),
                            };
                            SegmentND tensor = slice;
                            List<int> stridesS = new int[3] { _inputShape[1], _inputShape[2], _inputShape[3] }.ToList();
                            int sliceOffsetInTensor = TileUtilities.GetSliceOffsetInTensor(in tensor, in slice);
                            GNNEShape shape = new GNNEShape(tensor[0].Length, tensor[1].Length, tensor[2].Length,
                                tensor[3].Length);
                            gnneActionUpdater.UpdateLoadIf(tensor, _lif, ifmapIndex, ddrIf, sliceOffsetInTensor, null,
                                ItemName.Ifmap, ccrsToSet, null, h2c: false, 0, stridesS, shape);
                            List<CcrClr> ccrsToClr = new List<CcrClr>
                            {
                                new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, ifmapIndex))),
                            };
                            SegmentND storeSegment = ofmapSegment;
                            List<int> stridesD = new int[3]
                            {
                                storeSegment[1].Length, storeSegment[2].Length, storeSegment[3].Length,
                            }.ToList();
                            gnneActionUpdater.UpdateStoreT(storeSegment, _sofs[outputIndex], ifmapIndex, ddrOfs[outputIndex], 0, null, null,
                                ccrsToClr, ItemName.Ofmap, null, stridesD);
                        }
                    }
                }
            }

            splitOffset += _outputShapes[outputIndex][_axis];
            TileUtilities.Assert(ccrHandler.CcrSanityCheck(), "ccrHandler.CcrSanityCheck()",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileSplit.cs",
                296);
        }

        return actions;
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        IReadOnlyList<Expr> tupleInputs = (IReadOnlyList<Expr>)result["tupleInputs"];
        return GetReplace(tupleInputs);
    }
}
