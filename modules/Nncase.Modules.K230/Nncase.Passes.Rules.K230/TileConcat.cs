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
        List<Nncase.TIR.Buffer> list = new List<Nncase.TIR.Buffer>();
        for (int i = 0; i < _lifs.Length; i++)
        {
            T.CreateBuffer(new TensorType(_lifs[i][GNNELoad.Input].CheckedDataType, _lifs[i].CheckedShape),
                MemoryLocation.Input, out Nncase.TIR.Buffer buffer, "ddrIf_" + i);
            list.Add(buffer);
        }

        T.CreateBuffer(new TensorType(stCall.CheckedDataType, stCall.CheckedShape), MemoryLocation.Output,
            out Nncase.TIR.Buffer buffer2, "var ddrOf");
        List<GnneAction> actions = BuildSchedule(list, buffer2);
        ActionToInstruct actionToInstruct = new ActionToInstruct();
        return T.PrimFunc($"TileConcat_{_count}", K230RtModule.Kind,
                list.Concat(new Nncase.TIR.Buffer[1] { buffer2 }).ToArray())
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
        GNNEShape gNNEShape = _inputShapes[index];
        int w = gNNEShape[3];
        int num = gNNEShape[3];
        int nPingPongSplit = 1;
        AllocateResult allocateResult = HandleAllocate(i, j, k, w, k, num, nPingPongSplit);
        if (!allocateResult.IsOk)
        {
            nPingPongSplit = 1;
        }

        for (; k < gNNEShape[2] && k < 65535; k++)
        {
            int num2 = k + 1;
            allocateResult = HandleAllocate(i, j, num2, w, num2, num, nPingPongSplit);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; j < gNNEShape[1] && j < 65535; j++)
        {
            allocateResult = HandleAllocate(i, j + 1, k, w, k, num, nPingPongSplit);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; i < gNNEShape[0] && i < 65535; i++)
        {
            allocateResult = HandleAllocate(i + 1, j, k, w, k, num, nPingPongSplit);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        allocateResult = HandleAllocate(i, j, k, w, k, num, nPingPongSplit, isFinal: true);
        TileUtilities.Assert(allocateResult.IsOk, "allocation.IsOk",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileConcat.cs",
            154);
        GNNEShape gNNEShape2 = new GNNEShape(i, j, k, num);
        return new TileConcatGlb(allocateResult.GlbMap, allocateResult.Items, gNNEShape2.Dims, GNNEEnv.NPingPongSplit,
            allocateResult.GlbMap[ItemName.Ifmap], allocateResult.GlbMap[ItemName.Ofmap]);
    }

    private AllocateResult HandleAllocate(int n, int c, int h, int w, int e, int f, int nPingPongSplit,
        bool isFinal = false)
    {
        List<BoxOnGlb> list = new List<BoxOnGlb>();
        Dictionary<ItemName, TensorOnGlb> dictionary = new Dictionary<ItemName, TensorOnGlb>();
        int num = (int)System.Math.Ceiling(1.0 * (double)c / (double)nPingPongSplit);
        TensorOnGlb tensorOnGlb = new TensorOnGlb(new int[4] { n, num, h, w }, _inputTypes[0], 0);
        int value = tensorOnGlb.AllocatedBytes * nPingPongSplit;
        value = TileUtilities.GetAlignedNum(value, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        TensorOnGlb tensorOnGlb2 = new TensorOnGlb(new int[4] { n, num, h, w }, _inputTypes[0], 0);
        int basementSize = SpaceSearcher.GetBasementSize();
        basementSize = TileUtilities.GetAlignedNum(basementSize, GNNEEnv.GlbWidth * GNNEEnv.GlbBankWidth);
        list.Add(new BoxOnGlb(new int[2] { GNNEEnv.GlbWidth, basementSize / GNNEEnv.GlbWidth / GNNEEnv.GlbBankWidth },
            ItemName.Basement));
        list.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.IfmapBankWidth, value / GNNEEnv.IfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.Ifmap));
        AllocateResult allocateResult = SpaceSearcher.TryAllocate(new BoxPacker(16) { Boxes = list });
        if (allocateResult.IsOk)
        {
            tensorOnGlb.Mmu = allocateResult.Items[ItemName.Ifmap];
            tensorOnGlb2.Mmu = allocateResult.Items[ItemName.Ifmap];
        }

        dictionary.Add(ItemName.Ifmap, tensorOnGlb);
        dictionary.Add(ItemName.Ofmap, tensorOnGlb2);
        return new AllocateResult { IsOk = allocateResult.IsOk, Items = allocateResult.Items, GlbMap = dictionary };
    }

    private List<GnneAction> BuildSchedule(List<Nncase.TIR.Buffer> ddrIfs, Nncase.TIR.Buffer ddrOf)
    {
        List<GnneAction> list = new List<GnneAction>();
        int[] strides = ((TensorConst)_sof[GNNEStore.Strides]).Value.ToArray<int>();
        Func<int, int, int, Segment1D> func = delegate(int start, int end, int dim)
        {
            int num4 = (int)System.Math.Ceiling(1.0 * (double)start / (double)strides[dim]);
            return new Segment1D(
                new System.Range(end: (int)System.Math.Ceiling(1.0 * (double)end / (double)strides[dim]), start: num4),
                new Padding(0, 0));
        };
        int num = 0;
        int num2 = 0;
        for (int num3 = 0; num3 < _lifs.Length; num3++)
        {
            GprHandler gpr = new GprHandler(GNNEEnv.GprNum);
            SsrHandler ssr = new SsrHandler(GNNEEnv.SsrNum);
            CcrHandler ccrHandler = new CcrHandler();
            TiledGlb tiledGlb = SearchGlbParameters(num3);
            GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(list, tiledGlb, ccrHandler, gpr, ssr);
            gnneActionUpdater.UpdateMmuConf();
            foreach (Segment1D item in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[0],
                         _inputShapes[num3][0]))
            {
                Segment1D segment1D = func(item.Start, item.End, 0);
                if (_axis == 0)
                {
                    segment1D = func(segment1D.Start + num, segment1D.End + num, 0);
                }

                foreach (Segment1D item2 in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[1],
                             _inputShapes[num3][1]))
                {
                    Segment1D segment1D2 = func(item2.Start, item2.End, 1);
                    if (_axis == 1)
                    {
                        segment1D2 = func(segment1D2.Start + num, segment1D2.End + num, 1);
                    }

                    foreach (Segment1D item3 in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[2],
                                 _inputShapes[num3][2]))
                    {
                        Segment1D segment1D3 = func(item3.Start, item3.End, 2);
                        if (_axis == 2)
                        {
                            segment1D3 = func(segment1D3.Start + num, segment1D3.End + num, 2);
                        }

                        foreach (Segment1D item4 in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[3],
                                     _inputShapes[num3][3]))
                        {
                            Segment1D segment1D4 = func(item4.Start, item4.End, 3);
                            if (_axis == 3)
                            {
                                segment1D4 = func(segment1D4.Start + num, segment1D4.End + num, 3);
                            }

                            SegmentND segmentND = new SegmentND(segment1D, segment1D2, segment1D3, segment1D4);
                            SegmentND slice = new SegmentND(segment1D, item2, item3, item4);
                            List<CcrSet> ccrsToSet = new List<CcrSet>
                            {
                                new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, num2)), 1)
                            };
                            SegmentND tensor = slice;
                            List<int> list2 =
                                new int[3] { tensor[1].Length, tensor[2].Length, tensor[3].Length }.ToList();
                            gnneActionUpdater.UpdateLoadIf(tensor, _lifs[num3], num2, ddrIfs[num3], 0, list2,
                                ItemName.Ifmap, ccrsToSet);
                            List<CcrClr> ccrsToClr = new List<CcrClr>
                            {
                                new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, num2)))
                            };
                            SegmentND segmentND2 = segmentND;
                            SegmentND segmentND3 = segmentND;
                            List<int> stridesD =
                                new int[3] { _outputShape[1], _outputShape[2], _outputShape[3] }.ToList();
                            segmentND3[0] = new Segment1D((segmentND3[0].Start * strides[0])..segmentND3[0].End,
                                new Padding(0, 0));
                            segmentND3[1] = new Segment1D((segmentND3[1].Start * strides[1])..segmentND3[1].End,
                                new Padding(0, 0));
                            segmentND3[2] = new Segment1D((segmentND3[2].Start * strides[2])..segmentND3[2].End,
                                new Padding(0, 0));
                            segmentND3[3] = new Segment1D((segmentND3[3].Start * strides[3])..segmentND3[3].End,
                                new Padding(0, 0));
                            int sliceOffsetInTensor = TileUtilities.GetSliceOffsetInTensor(in tensor, in slice);
                            GNNEShape shape = new GNNEShape(segmentND2[0].Length, segmentND2[1].Length,
                                segmentND2[2].Length, segmentND2[3].Length);
                            gnneActionUpdater.UpdateStoreT(segmentND2, _sof, num2, ddrOf, sliceOffsetInTensor, null,
                                null, ccrsToClr, ItemName.Ofmap, list2, stridesD, shape);
                        }
                    }
                }
            }

            num += _inputShapes[num3][_axis];
            TileUtilities.Assert(ccrHandler.CcrSanityCheck(), "ccrHandler.CcrSanityCheck()",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileConcat.cs",
                301);
        }

        return list;
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        IReadOnlyList<Expr> tupleInputs = (IReadOnlyList<Expr>)__result["tupleInputs"];
        Call stCall = (Call)__result["stCall"];
        Concat concat = (Concat)__result["concat"];
        return GetReplace(tupleInputs, stCall, concat);
    }
}
