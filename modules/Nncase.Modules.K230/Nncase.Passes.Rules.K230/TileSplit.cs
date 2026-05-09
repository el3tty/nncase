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
        Nncase.PatternMatch.Utility.IsTuple("tuple",
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
            List<Nncase.TIR.Buffer> list = new List<Nncase.TIR.Buffer>();
            for (int i = 0; i < _sofs.Length; i++)
            {
                T.CreateBuffer(new TensorType(_sofs[i].CheckedDataType, _sofs[i].CheckedShape), MemoryLocation.Output,
                    out Nncase.TIR.Buffer buffer2, "ddrOf_" + i);
                list.Add(buffer2);
            }

            List<GnneAction> actions = BuildSchedule(buffer, list);
            ActionToInstruct actionToInstruct = new ActionToInstruct();
            return T.PrimFunc($"TileSplit_{_count}", K230RtModule.Kind,
                    new Nncase.TIR.Buffer[1] { buffer }.Concat(list).ToArray())
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
        _outputShapes = _sofs.Select((Call i) => new GNNEShape(i.CheckedShape[0].FixedValue,
            i.CheckedShape[1].FixedValue, i.CheckedShape[2].FixedValue, i.CheckedShape[3].FixedValue)).ToArray();
        _ = _inputShape.Dims[0];
        _axis = ((TensorConst)call[Split.Axis]).Value.ToScalar<int>();
    }

    private TileSplitGlb SearchGlbParameters(int index)
    {
        int i = 1;
        int j = 1;
        int k = 1;
        GNNEShape gNNEShape = _outputShapes[index];
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
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileSplit.cs",
            154);
        GNNEShape gNNEShape2 = new GNNEShape(i, j, k, num);
        return new TileSplitGlb(allocateResult.GlbMap, allocateResult.Items, gNNEShape2.Dims, GNNEEnv.NPingPongSplit,
            allocateResult.GlbMap[ItemName.Ifmap], allocateResult.GlbMap[ItemName.Ofmap]);
    }

    private AllocateResult HandleAllocate(int n, int c, int h, int w, int e, int f, int nPingPongSplit,
        bool isFinal = false)
    {
        List<BoxOnGlb> list = new List<BoxOnGlb>();
        Dictionary<ItemName, TensorOnGlb> dictionary = new Dictionary<ItemName, TensorOnGlb>();
        int num = (int)Math.Ceiling(1.0 * (double)c / (double)nPingPongSplit);
        TensorOnGlb tensorOnGlb = new TensorOnGlb(new int[4] { n, num, h, w }, _inputType, 0);
        int value = tensorOnGlb.AllocatedBytes * nPingPongSplit;
        value = TileUtilities.GetAlignedNum(value, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        TensorOnGlb tensorOnGlb2 = new TensorOnGlb(new int[4] { n, num, h, w }, _inputType, 0);
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

    private List<GnneAction> BuildSchedule(Nncase.TIR.Buffer ddrIf, List<Nncase.TIR.Buffer> ddrOfs)
    {
        List<GnneAction> list = new List<GnneAction>();
        int[] strides = ((TensorConst)_sofs[0][GNNEStore.Strides]).Value.ToArray<int>();
        Func<int, int, int, Segment1D> func = delegate(int start, int end, int dim)
        {
            int num4 = (int)Math.Ceiling(1.0 * (double)start / (double)strides[dim]);
            return new Segment1D(
                new System.Range(end: (int)Math.Ceiling(1.0 * (double)end / (double)strides[dim]), start: num4),
                new Padding(0, 0));
        };
        int num = 0;
        int num2 = 0;
        for (int num3 = 0; num3 < _sofs.Length; num3++)
        {
            GprHandler gpr = new GprHandler(GNNEEnv.GprNum);
            SsrHandler ssr = new SsrHandler(GNNEEnv.SsrNum);
            CcrHandler ccrHandler = new CcrHandler();
            TiledGlb tiledGlb = SearchGlbParameters(num3);
            GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(list, tiledGlb, ccrHandler, gpr, ssr);
            gnneActionUpdater.UpdateMmuConf();
            foreach (Segment1D item in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[0],
                         _outputShapes[num3][0]))
            {
                func(item.Start, item.End, 0);
                if (_axis == 0)
                {
                    func(item.Start + num, item.End + num, 0);
                }

                foreach (Segment1D item2 in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[1],
                             _outputShapes[num3][1]))
                {
                    Segment1D segment1D = func(item2.Start, item2.End, 1);
                    if (_axis == 1)
                    {
                        segment1D = func(item2.Start + num, item2.End + num, 1);
                    }

                    foreach (Segment1D item3 in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[2],
                                 _outputShapes[num3][2]))
                    {
                        Segment1D segment1D2 = func(item3.Start, item3.End, 2);
                        if (_axis == 2)
                        {
                            segment1D2 = func(item3.Start + num, item3.End + num, 2);
                        }

                        foreach (Segment1D item4 in TileUtilities.GetSegmentStartEndLength(0, tiledGlb.LastOutShape[3],
                                     _outputShapes[num3][3]))
                        {
                            Segment1D segment1D3 = func(item4.Start, item4.End, 3);
                            if (_axis == 3)
                            {
                                segment1D3 = func(item4.Start + num, item4.End + num, 3);
                            }

                            SegmentND segmentND = new SegmentND(item, item2, item3, item4);
                            SegmentND slice = new SegmentND(item, segment1D, segment1D2, segment1D3);
                            List<CcrSet> ccrsToSet = new List<CcrSet>
                            {
                                new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, num2)), 1)
                            };
                            SegmentND tensor = slice;
                            List<int> stridesS = new int[3] { _inputShape[1], _inputShape[2], _inputShape[3] }.ToList();
                            int sliceOffsetInTensor = TileUtilities.GetSliceOffsetInTensor(in tensor, in slice);
                            GNNEShape shape = new GNNEShape(tensor[0].Length, tensor[1].Length, tensor[2].Length,
                                tensor[3].Length);
                            gnneActionUpdater.UpdateLoadIf(tensor, _lif, num2, ddrIf, sliceOffsetInTensor, null,
                                ItemName.Ifmap, ccrsToSet, null, h2c: false, 0, stridesS, shape);
                            List<CcrClr> ccrsToClr = new List<CcrClr>
                            {
                                new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, num2)))
                            };
                            SegmentND segmentND2 = segmentND;
                            List<int> stridesD = new int[3]
                            {
                                segmentND2[1].Length, segmentND2[2].Length, segmentND2[3].Length
                            }.ToList();
                            gnneActionUpdater.UpdateStoreT(segmentND2, _sofs[num3], num2, ddrOfs[num3], 0, null, null,
                                ccrsToClr, ItemName.Ofmap, null, stridesD);
                        }
                    }
                }
            }

            num += _outputShapes[num3][_axis];
            TileUtilities.Assert(ccrHandler.CcrSanityCheck(), "ccrHandler.CcrSanityCheck()",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileSplit.cs",
                296);
        }

        return list;
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        IReadOnlyList<Expr> tupleInputs = (IReadOnlyList<Expr>)__result["tupleInputs"];
        return GetReplace(tupleInputs);
    }
}
