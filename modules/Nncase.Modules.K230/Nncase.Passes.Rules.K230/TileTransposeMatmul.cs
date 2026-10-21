// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.K230;
using Nncase.PatternMatch;
using Nncase.Runtime.K230;
using Nncase.TIR;
using Nncase.TIR.Instructions;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class TileTransposeMatmul : RewriteRule<Pattern>
{
    private static int _count = -1;

    private readonly List<Tuple<SegmentND, TensorStat>>[] _weightRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _ofmapRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _g2LIfRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _g2RWRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _l2GOfRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _l2RIf2Rec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _g2RWSliceRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly Dictionary<Var, Nncase.TIR.Buffer> _ifBufferMap =
        new Dictionary<Var, Nncase.TIR.Buffer>(ReferenceEqualityComparer.Instance);

    private readonly List<WeightGroupHandler> _weightGroups = new List<WeightGroupHandler>(2);

    private CcrHandler _ccrHandler = new CcrHandler();

    private GprHandler _gpr = new GprHandler();

    private SsrHandler _ssr = new SsrHandler();

    private GlbSearchStrategy _strategy;

    private DataType? _inputType;

    private DataType? _outputType;

    private DataType? _weightType;

    private Call? _conv;

    private Call? _pool;

    private Call? _dw;

    private Call? _act1;

    private int _icPerGroup;

    private int _ocPerGroup;

    private int _groupPerPass;

    private Call? _lif;

    private Call? _lif2;

    private Call? _lw;

    private Call? _lact;

    private Call? _lwQarg;

    private Call? _sof;

    private bool _weightPreloadEn;

    private bool _h2C;

    private GNNEShape? _inputShape;

    private GNNEShape? _outputShape;

    private GNNEShape? _convOutputShape;

    private GNNEShape? _weightsShape;

    private Padding? _paddingH;

    private Padding? _paddingW;

    private int _strideH;

    private int _strideW;

    private int _dilationH;

    private int _dilationW;

    private int _groups;

    private int _fusedKernelH;

    private int _fusedKernelW;

    private Padding? _fusedPaddingH;

    private Padding? _fusedPaddingW;

    private int _fusedStrideH;

    private int _fusedStrideW;

    private int _fusedDilationH;

    private int _fusedDilationW;

    private Tuple<int, int>? _weightSplitPattern;

    private bool _tp1;

    private bool _tp2;

    public override OrPattern Pattern { get; } = Nncase.PatternMatch.Utility.IsAlt(FusionPattern.IsL1Pdp0DwFusion(),
        FusionPattern.IsL1Pdp0ReduceFusion(), FusionPattern.IsL1Act1Fusion(), FusionPattern.IsGNNEConvFusion());

    private PrimFunction GetReplace(Call call, Call ld, Call st, Fusion fusion)
    {
        _count++;
        InitParameters(call, ld, st);
        if (!_tp1 || !_tp2)
        {
            return null;
        }

        TileTransposeMatmulGlb glb = SearchGlbParameters();
        int[] inputShape = ld.Arguments[0].CheckedShape.ToValueArray();
        T.CreateBuffer(new TensorType(ld[GNNELoad.Input].CheckedDataType, inputShape), MemoryLocation.Input,
            out Nncase.TIR.Buffer ddrIf, "var ddrIf");
        List<Nncase.TIR.Buffer> buffers = new List<Nncase.TIR.Buffer> { ddrIf };
        _ifBufferMap.Add((Var)ld[GNNELoad.Input], ddrIf);
        T.CreateBuffer(new TensorType(st.CheckedDataType, st.CheckedShape), MemoryLocation.Output,
            out Nncase.TIR.Buffer ddrOf, "var ddrOf");
        byte[] weightBytes = ((TensorConst)((Call)call[GNNEConv2D.Weights])[GNNELoadW.Input]).Value.BytesBuffer.ToArray();
        byte[] weightBytesCopy = new byte[weightBytes.Length];
        Array.Copy(weightBytes, weightBytesCopy, weightBytesCopy.Length);
        T.AttachBuffer(
            Const.FromTensor(Tensor.FromBytes(DataTypes.UInt8, weightBytesCopy.ToArray(), new int[1] { weightBytesCopy.Length })),
            out Nncase.TIR.Buffer ddrW, "var ddrW");
        T.AttachBuffer((TensorConst)((Call)call[GNNEConv2D.WeightsBias])[GNNELoadW.Input],
            out Nncase.TIR.Buffer ddrWQarg, "var ddrWQarg");
        T.AttachBuffer((TensorConst)((Call)call[GNNEConv2D.Act])[GNNELoadW.Input], out Nncase.TIR.Buffer ddrAct,
            "var ddrAct");
        Nncase.TIR.Buffer ddrDW = null;
        Nncase.TIR.Buffer ddrDWQarg = null;
        Nncase.TIR.Buffer ddrDWAct = null;
        Nncase.TIR.Buffer ddrAct1 = null;
        Nncase.TIR.Buffer ddrPdpAct = null;
        Nncase.TIR.Buffer ddrIf2 = null;
        if ((object)_dw != null)
        {
            byte[] dwWeightBytes = ((TensorConst)((Call)_dw[GNNEPdp0DW.Weights])[GNNELoadW.Input]).Value.BytesBuffer.ToArray();
            byte[] dwWeightBytesCopy = new byte[dwWeightBytes.Length];
            Array.Copy(dwWeightBytes, dwWeightBytesCopy, dwWeightBytesCopy.Length);
            T.AttachBuffer(
                Const.FromTensor(Tensor.FromBytes(DataTypes.UInt8, dwWeightBytesCopy.ToArray(), new int[1] { dwWeightBytesCopy.Length })),
                out ddrDW, "ddrDW");
            T.AttachBuffer((TensorConst)((Call)_dw[GNNEPdp0DW.WeightsBias])[GNNELoadW.Input], out ddrDWQarg, "ddrDWQarg");
            T.AttachBuffer((TensorConst)((Call)_dw[GNNEPdp0DW.Act])[GNNELoadW.Input], out ddrDWAct, "ddrDWAct");
        }

        if ((object)_pool != null)
        {
            T.AttachBuffer((TensorConst)((Call)_pool[GNNEPdp0Reduce.Act])[GNNELoadW.Input], out ddrPdpAct, "ddrPdpAct");
        }

        if ((object)_act1 != null)
        {
            T.AttachBuffer((TensorConst)((Call)_act1[GNNEActivation.Act])[GNNELoadW.Input], out ddrAct1, "ddrAct1");
            if ((object)_lif2 != null)
            {
                if (!(_lif2[GNNELoad.Input] is TensorConst))
                {
                    ddrIf2 = ddrIf;
                    if (_lif2[GNNELoad.Input] != _lif[GNNELoad.Input])
                    {
                        T.CreateBuffer(new TensorType(_lif2[GNNELoad.Input].CheckedDataType, _lif2.CheckedShape),
                            MemoryLocation.Input, out ddrIf2, "ddrIf2");
                        buffers.Add(ddrIf2);
                        _ifBufferMap.Add((Var)_lif2[GNNELoad.Input], ddrIf2);
                    }
                }
                else
                {
                    T.AttachBuffer((TensorConst)_lif2[GNNELoad.Input], out ddrIf2, "ddrIf2");
                }
            }
        }

        buffers = ArrayExtensions.AsValueEnumerable<Var>(fusion.Parameters)
            .Select<Nncase.TIR.Buffer>((Func<Var, Nncase.TIR.Buffer>)((Var v) => _ifBufferMap[v])).ToList();
        buffers.Add(ddrOf);
        ItemRecStatusInit();
        BuildIfFirstSchedule(glb, call, ld, st, ddrIf, ddrOf, ddrW, ddrWQarg, ddrAct, ddrDW, ddrDWQarg, ddrDWAct,
            ddrAct1, ddrPdpAct, ddrIf2, weightGroupOnly: true, _weightGroups[0]);
        List<GnneAction> actionsIfFirst = BuildIfFirstSchedule(glb, call, ld, st, ddrIf, ddrOf, ddrW, ddrWQarg, ddrAct,
            ddrDW, ddrDWQarg, ddrDWAct, ddrAct1, ddrPdpAct, ddrIf2, weightGroupOnly: false, _weightGroups[0]);
        ScheduleStrategy index = ScheduleStrategy.IfFirstSchedule;
        List<GnneAction> actions = actionsIfFirst;
        Span<byte> bytesBuffer = ((TensorConst)((Call)ddrW.MemSpan.Start)[DDrOf.Input]).Value.BytesBuffer;
        ArrangeWeights(_weightType, _weightsShape, bytesBuffer, _weightGroups[(int)index]);
        ActionToInstruct actionToInstruct = new ActionToInstruct();
        return T.PrimFunc($"TileTransposeMatmul_{_count}", K230RtModule.Kind, buffers.ToArray())
            .Body(actionToInstruct.Instructions(actions), I.END(GP_REGISTER.x0, 0)).Build();
    }

    private List<GnneAction> BuildIfFirstSchedule(TiledGlb glb, Call convCall, Call ld, Call st,
        Nncase.TIR.Buffer ddrIf, Nncase.TIR.Buffer ddrOf, Nncase.TIR.Buffer ddrW, Nncase.TIR.Buffer ddrWQarg,
        Nncase.TIR.Buffer ddrAct, Nncase.TIR.Buffer ddrDW, Nncase.TIR.Buffer ddrDWQarg, Nncase.TIR.Buffer ddrDWAct,
        Nncase.TIR.Buffer ddrAct1, Nncase.TIR.Buffer ddrPdpAct, Nncase.TIR.Buffer ddrIf2, bool weightGroupOnly,
        WeightGroupHandler weightGroup)
    {
        List<GnneAction> actions = new List<GnneAction>();
        _gpr = new GprHandler(GNNEEnv.GprNum);
        _ssr = new SsrHandler(GNNEEnv.SsrNum);
        _ccrHandler = new CcrHandler();
        GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, glb, _ccrHandler, _gpr, _ssr);
        if (!weightGroupOnly)
        {
            UpdateCcrRecStat();
        }

        if (!weightGroupOnly)
        {
            gnneActionUpdater.UpdateMmuConf();
        }

        int pingPongIndex = 0;
        int ifmapSlot = -1;
        int weightSlot = -1;
        _ = GNNEEnv.NPingPongSplit;
        List<Segment1D> segmentsN =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[0], _outputShape[0]);
        List<Segment1D> segmentsH =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[2], _outputShape[2]);
        List<Segment1D> segmentsW =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[3], _outputShape[3]);
        List<Segment1D> ocSegments = new List<Segment1D>();
        List<Segment1D> icSegments = new List<Segment1D>();
        if (glb.LastOutShape[1] >= _ocPerGroup)
        {
            ocSegments = TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[1], _outputShape[1]);
            icSegments = TileUtilities.GetSegmentStartEndLength(0, glb.GlbMap[ItemName.Ifmap].Dimensions[1], _inputShape[1]);
            if (_tp1)
            {
                icSegments = TileUtilities.GetSegmentStartEndLength(0, glb.GlbMap[ItemName.Ifmap2].Dimensions[3],
                    _inputShape[1]);
            }
        }
        else
        {
            for (int groupIndex = 0; groupIndex < _groups; groupIndex++)
            {
                List<Segment1D> groupOcSegments =
                    TileUtilities.GetSegmentStartEndLength(groupIndex * _ocPerGroup, glb.LastOutShape[1], (groupIndex + 1) * _ocPerGroup);
                List<Segment1D> groupIcSegments = TileUtilities.GetSegmentStartEndLength(groupIndex * _icPerGroup,
                    glb.GlbMap[ItemName.Ifmap].Dimensions[1], (groupIndex + 1) * _icPerGroup);
                if (_tp1)
                {
                    groupIcSegments = TileUtilities.GetSegmentStartEndLength(groupIndex * _icPerGroup,
                        glb.GlbMap[ItemName.Ifmap2].Dimensions[3], (groupIndex + 1) * _icPerGroup);
                }

                ocSegments.AddRange(groupOcSegments);
                icSegments.AddRange(groupIcSegments);
            }
        }

        if (!weightGroupOnly)
        {
            if (_weightType == DataTypes.UInt8 || _weightType == DataTypes.Int16)
            {
                gnneActionUpdater.UpdateLoadWQarg(_lwQarg, weightGroup, ddrWQarg);
            }

            gnneActionUpdater.UpdateLoadAct(_lact, ddrAct);
            if ((object)_dw != null)
            {
                gnneActionUpdater.UpdateLoadDw(_dw, weightGroup, ddrDW);
                if (_dw[GNNEPdp0DW.Weights].CheckedDataType == DataTypes.UInt8)
                {
                    gnneActionUpdater.UpdateLoadDwQarg(_dw, weightGroup, ddrDWQarg);
                }
            }

            if ((object)_act1 != null)
            {
                gnneActionUpdater.UpdateLoadAct((Call)_act1[GNNEActivation.Act], ddrAct1, ItemName.MfuAct1);
            }

            if ((object)_dw != null)
            {
                gnneActionUpdater.UpdateLoadAct((Call)_dw[GNNEPdp0DW.Act], ddrDWAct, ItemName.DwAct1);
            }
            else if ((object)_pool != null)
            {
                gnneActionUpdater.UpdateLoadAct((Call)_pool[GNNEPdp0Reduce.Act], ddrPdpAct, ItemName.PdpAct1);
            }
        }

        Segment1D emptySegment = new Segment1D(0..0, new Padding(0, 0));
        List<SegmentND> segmentHistory = new List<SegmentND>();
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        List<SegmentND> ifmapHistory = segmentHistory;
        segmentHistory = new List<SegmentND>();
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        List<SegmentND> weightHistory = segmentHistory;
        int l1Pp = 0;
        _weightSplitPattern = new Tuple<int, int>(0, 0);
        foreach (Segment1D segmentN in segmentsN)
        {
            foreach (Segment1D segmentH in segmentsH)
            {
                Segment1D convInputRowSegment = TileUtilities.GetInputRowSegment(segmentH.Start, segmentH.Length,
                    _convOutputShape[2], _fusedKernelH, _fusedStrideH, _fusedDilationH, in _fusedPaddingH);
                Segment1D ifmapRowSegment = TileUtilities.GetInputRowSegment(convInputRowSegment.Start,
                    convInputRowSegment.Length, _inputShape[2], _weightsShape[2], _strideH, _dilationH, in _paddingH);
                foreach (Segment1D segmentW in segmentsW)
                {
                    Segment1D convInputColumnSegment = TileUtilities.GetInputColumnSegment(segmentW.Start, segmentW.Length,
                        _convOutputShape[3], _fusedKernelW, _fusedStrideW, _fusedDilationW, in _fusedPaddingW);
                    Segment1D ifmapColumnSegment = TileUtilities.GetInputColumnSegment(convInputColumnSegment.Start,
                        convInputColumnSegment.Length, _inputShape[3], _weightsShape[3], _strideW, _dilationW,
                        in _paddingW);
                    foreach (Segment1D ocSegment in ocSegments)
                    {
                        SegmentND outputSegment = new SegmentND(segmentN, ocSegment, segmentH, segmentW);
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
                                new Segment1D(0.._weightsShape[2], new Padding(0, 0)),
                                new Segment1D(0.._weightsShape[3], new Padding(0, 0)));
                            if (weightHistory[0] == weightSegment)
                            {
                                weightSlot = 0;
                            }
                            else if (weightHistory[1] == weightSegment)
                            {
                                weightSlot = 1;
                            }
                            else
                            {
                                weightSlot = (weightSlot + 1) % 2;
                                weightGroup.Current_aligned_offset_init();
                                if (weightGroupOnly)
                                {
                                    _weightRec[weightSlot].Add(new Tuple<SegmentND, TensorStat>(weightSegment,
                                        new TensorStat(isFirstSlice: false, isLastSlice: false)));
                                }

                                weightHistory[weightSlot] = weightSegment;
                            }

                            SegmentND ifmapSegment = new SegmentND(segmentN, icSegment, ifmapRowSegment, ifmapColumnSegment);
                            SegmentND ifmap2Segment = ifmapSegment;
                            if (_tp1)
                            {
                                ifmap2Segment = new SegmentND(segmentN, ifmapRowSegment, ifmapColumnSegment, icSegment);
                                SegmentND ifmap = new SegmentND(ifmapColumnSegment, icSegment, segmentN, ifmapRowSegment);
                                if (ifmapHistory[0] == ifmap2Segment)
                                {
                                    ifmapSlot = 0;
                                }
                                else if (ifmapHistory[1] == ifmap2Segment)
                                {
                                    ifmapSlot = 1;
                                }
                                else
                                {
                                    ifmapSlot = (ifmapSlot + 1) % 2;
                                    if (!weightGroupOnly && _g2LIfRec[ifmapSlot].Count > 0 &&
                                        _g2LIfRec[ifmapSlot][0].Item1 == ifmap2Segment)
                                    {
                                        TensorStat loadStat = _g2LIfRec[ifmapSlot][0].Item2;
                                        int ifmapCcrCount = ((loadStat != null && loadStat.IsFirstSlice && loadStat.IsLastSlice) ? 1 : 2);
                                        List<CcrSet> ccrsToSet = new List<CcrSet>
                                        {
                                            new CcrSet(
                                                _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap2, ifmapSlot)),
                                                ifmapCcrCount)
                                        };
                                        List<int> stridesS = new List<int> { 1, _inputShape[0], _inputShape[1] };
                                        List<int> ifmapStrides = new List<int>
                                        {
                                            glb.GlbMap[ItemName.Ifmap2].Dimensions[1],
                                            glb.GlbMap[ItemName.Ifmap2].Dimensions[2],
                                            glb.GlbMap[ItemName.Ifmap2].Dimensions[3]
                                        };
                                        GNNEShape shape = new GNNEShape(ifmap2Segment[0].Length, ifmap2Segment[1].Length,
                                            ifmap2Segment[2].Length, ifmap2Segment[3].Length);
                                        gnneActionUpdater.UpdateLoadIf(ifmap, _lif, ifmapSlot, ddrIf, 0, ifmapStrides,
                                            ItemName.Ifmap2, ccrsToSet, null, _h2C,
                                            ((TensorConst)convCall[GNNEConv2D.DeqBias]).Value.ToScalar<int>(), stridesS,
                                            shape);
                                    }

                                    ifmapHistory[ifmapSlot] = ifmap2Segment;
                                }
                            }
                            else if (ifmapHistory[0] == ifmapSegment)
                            {
                                ifmapSlot = 0;
                            }
                            else if (ifmapHistory[1] == ifmapSegment)
                            {
                                ifmapSlot = 1;
                            }
                            else
                            {
                                ifmapSlot = (ifmapSlot + 1) % 2;
                                if (!weightGroupOnly && _g2LIfRec[ifmapSlot].Count > 0 &&
                                    _g2LIfRec[ifmapSlot][0].Item1 == ifmapSegment)
                                {
                                    TensorStat loadStat = _g2LIfRec[ifmapSlot][0].Item2;
                                    int ifmap2CcrCount = ((loadStat != null && loadStat.IsFirstSlice && loadStat.IsLastSlice) ? 1 : 2);
                                    List<CcrSet> ifmap2CcrsToSet = new List<CcrSet>
                                    {
                                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap, ifmapSlot)),
                                            ifmap2CcrCount)
                                    };
                                    List<int> ifmap2Strides = new List<int>
                                    {
                                        glb.GlbMap[ItemName.Ifmap].Dimensions[1],
                                        glb.GlbMap[ItemName.Ifmap].Dimensions[2],
                                        glb.GlbMap[ItemName.Ifmap].Dimensions[3]
                                    };
                                    gnneActionUpdater.UpdateLoadIf(ifmapSegment, _lif, ifmapSlot, ddrIf, 0, ifmap2Strides,
                                        ItemName.Ifmap, ifmap2CcrsToSet, null, _h2C,
                                        ((TensorConst)convCall[GNNEConv2D.DeqBias]).Value.ToScalar<int>());
                                }

                                ifmapHistory[ifmapSlot] = ifmapSegment;
                            }

                            if (!weightGroupOnly && (object)_act1 != null &&
                                _act1[GNNEActivation.InputB] != None.Default && icSegment.End == _inputShape[1])
                            {
                                ifmap2Segment = outputSegment;
                                TensorStat loadStat = _l2RIf2Rec[pingPongIndex][0].Item2;
                                int ifmap2CcrCount = ((loadStat != null && loadStat.IsFirstSlice && loadStat.IsLastSlice) ? 1 : 2);
                                List<CcrSet> ccrsToSet3 = new List<CcrSet>
                                {
                                    new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap2, pingPongIndex)),
                                        ifmap2CcrCount)
                                };
                                List<int> stridesD3 = new List<int>
                                {
                                    glb.GlbMap[ItemName.Ifmap2].Dimensions[1],
                                    glb.GlbMap[ItemName.Ifmap2].Dimensions[2],
                                    glb.GlbMap[ItemName.Ifmap2].Dimensions[3]
                                };
                                gnneActionUpdater.UpdateLoadIf(ifmap2Segment, _lif2, pingPongIndex, ddrIf2, 0, stridesD3,
                                    ItemName.Ifmap2, ccrsToSet3);
                            }

                            BuildL1Schedule(gnneActionUpdater, glb, ifmapSegment, weightSegment, outputSegment, ifmapSlot, pingPongIndex, weightSlot,
                                ifmap2Segment, weightGroupOnly, weightGroup, l1Pp, ddrW, ddrOf);
                        }
                    }
                }
            }
        }

        TileUtilities.Assert(_ccrHandler.CcrSanityCheck(), "_ccrHandler.CcrSanityCheck()",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileTransposeMatmul.cs",
            466);
        return actions;
    }

    private List<GnneAction> BuildWFirstSchedule(TiledGlb glb, Call convCall, Call ld, Call st, Nncase.TIR.Buffer ddrIf,
        Nncase.TIR.Buffer ddrOf, Nncase.TIR.Buffer ddrW, Nncase.TIR.Buffer ddrWQarg, Nncase.TIR.Buffer ddrAct,
        Nncase.TIR.Buffer ddrDW, Nncase.TIR.Buffer ddrDWQarg, Nncase.TIR.Buffer ddrDWAct, Nncase.TIR.Buffer ddrAct1,
        Nncase.TIR.Buffer ddrPdpAct, Nncase.TIR.Buffer ddrIf2, bool weightGroupOnly, WeightGroupHandler weightGroup)
    {
        List<GnneAction> actions = new List<GnneAction>();
        _gpr = new GprHandler(GNNEEnv.GprNum);
        _ssr = new SsrHandler(GNNEEnv.SsrNum);
        _ccrHandler = new CcrHandler();
        GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, glb, _ccrHandler, _gpr, _ssr);
        if (!weightGroupOnly)
        {
            UpdateCcrRecStat();
        }

        if (!weightGroupOnly)
        {
            gnneActionUpdater.UpdateMmuConf();
        }

        int pingPongIndex = 0;
        int ifmapSlot = -1;
        int weightSlot = -1;
        int nPingPongSplit = GNNEEnv.NPingPongSplit;
        List<Segment1D> segmentsN =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[0], _outputShape[0]);
        List<Segment1D> segmentsH =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[2], _outputShape[2]);
        List<Segment1D> segmentsW =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[3], _outputShape[3]);
        List<Segment1D> ocSegments = new List<Segment1D>();
        List<Segment1D> icSegments = new List<Segment1D>();
        if (glb.LastOutShape[1] >= _ocPerGroup)
        {
            ocSegments = TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[1], _outputShape[1]);
            icSegments = TileUtilities.GetSegmentStartEndLength(0, glb.GlbMap[ItemName.Ifmap].Dimensions[1], _inputShape[1]);
        }
        else
        {
            for (int groupIndex = 0; groupIndex < _groups; groupIndex++)
            {
                List<Segment1D> groupOcSegments =
                    TileUtilities.GetSegmentStartEndLength(groupIndex * _ocPerGroup, glb.LastOutShape[1], (groupIndex + 1) * _ocPerGroup);
                List<Segment1D> groupIcSegments = TileUtilities.GetSegmentStartEndLength(groupIndex * _icPerGroup,
                    glb.GlbMap[ItemName.Ifmap].Dimensions[1], (groupIndex + 1) * _icPerGroup);
                ocSegments.AddRange(groupOcSegments);
                icSegments.AddRange(groupIcSegments);
            }
        }

        if (!weightGroupOnly)
        {
            if (_weightType == DataTypes.UInt8 || _weightType == DataTypes.Int16)
            {
                gnneActionUpdater.UpdateLoadWQarg(_lwQarg, weightGroup, ddrWQarg);
            }

            gnneActionUpdater.UpdateLoadAct(_lact, ddrAct);
            if ((object)_dw != null)
            {
                gnneActionUpdater.UpdateLoadDw(_dw, weightGroup, ddrDW);
                if (_dw[GNNEPdp0DW.Weights].CheckedDataType == DataTypes.UInt8)
                {
                    gnneActionUpdater.UpdateLoadDwQarg(_dw, weightGroup, ddrDWQarg);
                }
            }

            if ((object)_act1 != null)
            {
                gnneActionUpdater.UpdateLoadAct((Call)_act1[GNNEActivation.Act], ddrAct1, ItemName.MfuAct1);
            }

            if ((object)_dw != null)
            {
                gnneActionUpdater.UpdateLoadAct((Call)_dw[GNNEPdp0DW.Act], ddrDWAct, ItemName.DwAct1);
            }
            else if ((object)_pool != null)
            {
                gnneActionUpdater.UpdateLoadAct((Call)_pool[GNNEPdp0Reduce.Act], ddrPdpAct, ItemName.PdpAct1);
            }
        }

        Segment1D emptySegment = new Segment1D(0..0, new Padding(0, 0));
        List<SegmentND> segmentHistory = new List<SegmentND>();
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        List<SegmentND> ifmapHistory = segmentHistory;
        segmentHistory = new List<SegmentND>();
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        List<SegmentND> weightHistory = segmentHistory;
        int l1Pp = 0;
        _weightSplitPattern = new Tuple<int, int>(0, 0);
        foreach (Segment1D ocSegment in ocSegments)
        {
            int icGroupStart = ocSegment.Start / _ocPerGroup * _icPerGroup;
            int icGroupEnd = (ocSegment.End - 1) / _ocPerGroup * _icPerGroup + _icPerGroup;
            foreach (Segment1D segmentN in segmentsN)
            {
                foreach (Segment1D segmentH in segmentsH)
                {
                    Segment1D convInputRowSegment = TileUtilities.GetInputRowSegment(segmentH.Start, segmentH.Length,
                        _convOutputShape[2], _fusedKernelH, _fusedStrideH, _fusedDilationH, in _fusedPaddingH);
                    Segment1D ifmapRowSegment = TileUtilities.GetInputRowSegment(convInputRowSegment.Start,
                        convInputRowSegment.Length, _inputShape[2], _weightsShape[2], _strideH, _dilationH, in _paddingH);
                    foreach (Segment1D segmentW in segmentsW)
                    {
                        Segment1D convInputColumnSegment = TileUtilities.GetInputColumnSegment(segmentW.Start, segmentW.Length,
                            _convOutputShape[3], _fusedKernelW, _fusedStrideW, _fusedDilationW, in _fusedPaddingW);
                        Segment1D ifmapColumnSegment = TileUtilities.GetInputColumnSegment(convInputColumnSegment.Start,
                            convInputColumnSegment.Length, _inputShape[3], _weightsShape[3], _strideW, _dilationW,
                            in _paddingW);
                        SegmentND outputSegment = new SegmentND(segmentN, ocSegment, segmentH, segmentW);
                        foreach (Segment1D icSegment in icSegments)
                        {
                            if (icSegment.Start < icGroupStart || icSegment.End > icGroupEnd)
                            {
                                continue;
                            }

                            int icStartInGroup = icSegment.Start % _icPerGroup;
                            int icEndInGroup = Math.Min(icStartInGroup + icSegment.Length, _icPerGroup);
                            SegmentND weightSegment = new SegmentND(ocSegment, new Segment1D(icStartInGroup..icEndInGroup, new Padding(0, 0)),
                                new Segment1D(0.._weightsShape[2], new Padding(0, 0)),
                                new Segment1D(0.._weightsShape[3], new Padding(0, 0)));
                            if (weightHistory[0] == weightSegment)
                            {
                                weightSlot = 0;
                            }
                            else if (weightHistory[1] == weightSegment)
                            {
                                weightSlot = 1;
                            }
                            else
                            {
                                weightSlot = (weightSlot + 1) % 2;
                                weightGroup.Current_aligned_offset_init();
                                if (weightGroupOnly)
                                {
                                    _weightRec[weightSlot].Add(new Tuple<SegmentND, TensorStat>(weightSegment,
                                        new TensorStat(isFirstSlice: false, isLastSlice: false)));
                                }

                                weightHistory[weightSlot] = weightSegment;
                            }

                            SegmentND ifmapSegment = new SegmentND(segmentN, icSegment, ifmapRowSegment, ifmapColumnSegment);
                            if (ifmapHistory[0] == ifmapSegment)
                            {
                                ifmapSlot = 0;
                            }
                            else if (ifmapHistory[1] == ifmapSegment)
                            {
                                ifmapSlot = 1;
                            }
                            else
                            {
                                ifmapSlot = (ifmapSlot + 1) % 2;
                                if (!weightGroupOnly && _g2LIfRec[ifmapSlot].Count > 0 &&
                                    _g2LIfRec[ifmapSlot][0].Item1 == ifmapSegment)
                                {
                                    TensorStat loadStat = _g2LIfRec[ifmapSlot][0].Item2;
                                    int ifmapCcrCount = ((loadStat != null && loadStat.IsFirstSlice && loadStat.IsLastSlice) ? 1 : 2);
                                    List<CcrSet> ccrsToSet = new List<CcrSet>
                                    {
                                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap, ifmapSlot)),
                                            ifmapCcrCount)
                                    };
                                    List<int> ifmapStrides = new List<int>
                                    {
                                        glb.GlbMap[ItemName.Ifmap].Dimensions[1],
                                        glb.GlbMap[ItemName.Ifmap].Dimensions[2],
                                        glb.GlbMap[ItemName.Ifmap].Dimensions[3]
                                    };
                                    gnneActionUpdater.UpdateLoadIf(ifmapSegment, _lif, ifmapSlot, ddrIf, 0, ifmapStrides,
                                        ItemName.Ifmap, ccrsToSet, null, _h2C,
                                        ((TensorConst)convCall[GNNEConv2D.DeqBias]).Value.ToScalar<int>());
                                }

                                ifmapHistory[ifmapSlot] = ifmapSegment;
                            }

                            SegmentND ifmap2Segment = outputSegment;
                            if (!weightGroupOnly && (object)_act1 != null)
                            {
                                Call call = (Call)_act1[GNNEActivation.InputB];
                                if (((object)call == null || !(call.Target is Uninitialized)) &&
                                    icSegment.End == _inputShape[1])
                                {
                                    TensorStat loadStat = _l2RIf2Rec[pingPongIndex][0].Item2;
                                    int ifmap2CcrCount = ((loadStat != null && loadStat.IsFirstSlice && loadStat.IsLastSlice) ? 1 : 2);
                                    List<CcrSet> ifmap2CcrsToSet = new List<CcrSet>
                                    {
                                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap2, pingPongIndex)),
                                            ifmap2CcrCount)
                                    };
                                    List<int> ifmap2Strides = new List<int>
                                    {
                                        glb.GlbMap[ItemName.Ifmap2].Dimensions[1],
                                        glb.GlbMap[ItemName.Ifmap2].Dimensions[2],
                                        glb.GlbMap[ItemName.Ifmap2].Dimensions[3]
                                    };
                                    gnneActionUpdater.UpdateLoadIf(ifmap2Segment, _lif2, pingPongIndex, ddrIf2, 0, ifmap2Strides,
                                        ItemName.Ifmap2, ifmap2CcrsToSet);
                                }
                            }

                            BuildL1Schedule(gnneActionUpdater, glb, ifmapSegment, weightSegment, outputSegment, ifmapSlot, pingPongIndex, weightSlot,
                                ifmap2Segment, weightGroupOnly, weightGroup, l1Pp, ddrW, ddrOf);
                        }

                        if (!weightGroupOnly)
                        {
                            bool needsOfmapFake = !_ofmapRec[pingPongIndex][0].Item2.IsLastSlice;
                            _ofmapRec[pingPongIndex].RemoveAt(0);
                            List<CcrSet> ofmapFakeCcrsToSet = new List<CcrSet>();
                            if ((needsOfmapFake ? 1 : 0) > (false ? 1 : 0))
                            {
                                ofmapFakeCcrsToSet.Add(new CcrSet(
                                    _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.OfmapFake, pingPongIndex)), 1));
                            }

                            List<CcrClr> ccrsToClr = new List<CcrClr>
                            {
                                new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, pingPongIndex)))
                            };
                            gnneActionUpdater.UpdateStoreT(outputSegment, _sof, pingPongIndex, ddrOf, 0, null, ofmapFakeCcrsToSet, ccrsToClr);
                        }
                        else
                        {
                            _ofmapRec[pingPongIndex].Add(new Tuple<SegmentND, TensorStat>(outputSegment,
                                new TensorStat(isFirstSlice: false, isLastSlice: false)));
                        }

                        pingPongIndex = (pingPongIndex + 1) % nPingPongSplit;
                    }
                }
            }
        }

        TileUtilities.Assert(_ccrHandler.CcrSanityCheck(), "_ccrHandler.CcrSanityCheck()",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileTransposeMatmul.cs",
            703);
        return actions;
    }

    private TileTransposeMatmulGlb SearchGlbParameters()
    {
        int batch = 1;
        int channels = 1;
        int kernelH = _weightsShape[2];
        int kernelW = _weightsShape[3];
        int ofmapRows = 1;
        int ofmapCols = Math.Min(GNNEEnv.PuWidth, _outputShape[3]);
        if (ofmapCols < GNNEEnv.PuWidth)
        {
            ofmapRows = Math.Min((int)Math.Ceiling(1.0 * (double)GNNEEnv.PuWidth / (double)ofmapCols), _outputShape[2]);
        }

        int ifmapRows = SpaceSearcher.GetInputHeight(
            SpaceSearcher.GetInputHeight(ofmapRows, _convOutputShape[2], _fusedKernelH, _outputShape[2], _fusedStrideH,
                _fusedDilationH, in _fusedPaddingH), _inputShape[2] + _paddingH.Sum(), kernelH, _convOutputShape[2],
            _strideH, _dilationH, new Padding(0, 0));
        int ifmapCols = SpaceSearcher.GetInputHeight(
            SpaceSearcher.GetInputHeight(ofmapCols, _convOutputShape[3], _fusedKernelW, _outputShape[3], _fusedStrideW,
                _fusedDilationW, in _fusedPaddingW), _inputShape[3] + _paddingW.Sum(), kernelW, _convOutputShape[3],
            _strideW, _dilationW, new Padding(0, 0));
        int outChannels = 1;
        AllocateResult allocateResult;
        for (; channels < _groupPerPass * _icPerGroup; channels++)
        {
            allocateResult = HandleAllocate(batch, channels + 1, ifmapRows, ifmapCols, kernelH, kernelW, outChannels, ofmapRows, ofmapCols);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; outChannels < GNNEEnv.PuWidth && outChannels < _groupPerPass * _ocPerGroup; outChannels++)
        {
            allocateResult = HandleAllocate(batch, channels, ifmapRows, ifmapCols, kernelH, kernelW, outChannels + 1, ofmapRows, ofmapCols);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        if (_strategy == GlbSearchStrategy.IfFirst)
        {
            if (_groupPerPass == 1)
            {
                while (outChannels < _outputShape[1] && (kernelH != _weightsShape[2] || kernelW != _weightsShape[3] ||
                                               channels < _weightsShape[1] || outChannels < _outputShape[1] / 2))
                {
                    int nextOutChannels = outChannels + 1;
                    int nextChannels = channels;
                    if (outChannels >= _ocPerGroup)
                    {
                        nextOutChannels = outChannels + _ocPerGroup;
                        nextChannels = channels + _icPerGroup;
                    }

                    allocateResult = HandleAllocate(batch, nextChannels, ifmapRows, ifmapCols, kernelH, kernelW, nextOutChannels, ofmapRows, ofmapCols);
                    if (!allocateResult.IsOk)
                    {
                        break;
                    }

                    outChannels = nextOutChannels;
                    channels = nextChannels;
                }
            }
            else
            {
                while (outChannels < _outputShape[1] && (kernelH != _weightsShape[2] || kernelW != _weightsShape[3] ||
                                               channels < _weightsShape[1] || outChannels < _weightsShape[0] / 2))
                {
                    int nextOutChannels = Math.Min(outChannels + _groupPerPass * _ocPerGroup, _outputShape[1]);
                    int nextChannels = Math.Min(channels + _groupPerPass * _icPerGroup, _inputShape[1]);
                    allocateResult = HandleAllocate(batch, nextChannels, ifmapRows, ifmapCols, kernelH, kernelW, nextOutChannels, ofmapRows, ofmapCols);
                    if (!allocateResult.IsOk)
                    {
                        break;
                    }

                    outChannels = nextOutChannels;
                    channels = nextChannels;
                }
            }

            while (ofmapRows < _outputShape[2])
            {
                int nextOfmapRows = ofmapRows + 1;
                int nextIfmapRows = SpaceSearcher.GetInputHeight(
                    SpaceSearcher.GetInputHeight(nextOfmapRows, _convOutputShape[2], _fusedKernelH, _outputShape[2],
                        _fusedStrideH, _fusedDilationH, in _fusedPaddingH), _inputShape[2] + _paddingH.Sum(), kernelH,
                    _convOutputShape[2], _strideH, _dilationH, new Padding(0, 0));
                allocateResult = HandleAllocate(batch, channels, nextIfmapRows, ifmapCols, kernelH, kernelW, outChannels, nextOfmapRows, ofmapCols);
                if (!allocateResult.IsOk)
                {
                    break;
                }

                ofmapRows = nextOfmapRows;
                ifmapRows = nextIfmapRows;
            }

            while (ofmapCols < _outputShape[3])
            {
                int nextOfmapCols = ofmapCols + 1;
                int nextIfmapCols = SpaceSearcher.GetInputHeight(
                    SpaceSearcher.GetInputHeight(nextOfmapCols, _convOutputShape[3], _fusedKernelW, _outputShape[3],
                        _fusedStrideW, _fusedDilationW, in _fusedPaddingW), _inputShape[3] + _paddingW.Sum(), kernelW,
                    _convOutputShape[3], _strideW, _dilationW, new Padding(0, 0));
                allocateResult = HandleAllocate(batch, channels, ifmapRows, nextIfmapCols, kernelH, kernelW, outChannels, ofmapRows, nextOfmapCols);
                if (!allocateResult.IsOk)
                {
                    break;
                }

                ofmapCols = nextOfmapCols;
                ifmapCols = nextIfmapCols;
            }

            for (; batch < _outputShape[0]; batch++)
            {
                allocateResult = HandleAllocate(batch + 1, channels, ifmapRows, ifmapCols, kernelH, kernelW, outChannels, ofmapRows, ofmapCols);
                if (!allocateResult.IsOk)
                {
                    break;
                }
            }
        }
        else if (_strategy == GlbSearchStrategy.WFirst)
        {
            while (ofmapRows < _outputShape[2])
            {
                int nextOfmapRows = ofmapRows + 1;
                int nextIfmapRows = SpaceSearcher.GetInputHeight(
                    SpaceSearcher.GetInputHeight(nextOfmapRows, _convOutputShape[2], _fusedKernelH, _outputShape[2],
                        _fusedStrideH, _fusedDilationH, in _fusedPaddingH), _inputShape[2] + _paddingH.Sum(), kernelH,
                    _convOutputShape[2], _strideH, _dilationH, new Padding(0, 0));
                allocateResult = HandleAllocate(batch, channels, nextIfmapRows, ifmapCols, kernelH, kernelW, outChannels, nextOfmapRows, ofmapCols);
                if (!allocateResult.IsOk)
                {
                    break;
                }

                ofmapRows = nextOfmapRows;
                ifmapRows = nextIfmapRows;
            }

            while (ofmapCols < _outputShape[3])
            {
                int nextOfmapCols = ofmapCols + 1;
                int nextIfmapCols = SpaceSearcher.GetInputHeight(
                    SpaceSearcher.GetInputHeight(nextOfmapCols, _convOutputShape[3], _fusedKernelW, _outputShape[3],
                        _fusedStrideW, _fusedDilationW, in _fusedPaddingW), _inputShape[3] + _paddingW.Sum(), kernelW,
                    _convOutputShape[3], _strideW, _dilationW, new Padding(0, 0));
                allocateResult = HandleAllocate(batch, channels, ifmapRows, nextIfmapCols, kernelH, kernelW, outChannels, ofmapRows, nextOfmapCols);
                if (!allocateResult.IsOk)
                {
                    break;
                }

                ofmapCols = nextOfmapCols;
                ifmapCols = nextIfmapCols;
            }

            for (;
                 batch < _outputShape[0] && (channels != _inputShape[1] || ifmapRows != _inputShape[2] || ifmapCols != _inputShape[3] ||
                                         batch < _outputShape[0] / 2);
                 batch++)
            {
                allocateResult = HandleAllocate(batch + 1, channels, ifmapRows, ifmapCols, kernelH, kernelW, outChannels, ofmapRows, ofmapCols);
                if (!allocateResult.IsOk)
                {
                    break;
                }
            }

            if (_groupPerPass == 1)
            {
                while (outChannels < _outputShape[1])
                {
                    int nextOutChannels = outChannels + 1;
                    int nextChannels = channels;
                    if (outChannels >= _ocPerGroup)
                    {
                        nextOutChannels = outChannels + _ocPerGroup;
                        nextChannels = channels + _icPerGroup;
                    }

                    allocateResult = HandleAllocate(batch, nextChannels, ifmapRows, ifmapCols, kernelH, kernelW, nextOutChannels, ofmapRows, ofmapCols);
                    if (!allocateResult.IsOk)
                    {
                        break;
                    }

                    outChannels = nextOutChannels;
                    channels = nextChannels;
                }
            }
            else
            {
                while (outChannels < _outputShape[1])
                {
                    int nextOutChannels = Math.Min(outChannels + _groupPerPass * _ocPerGroup, _outputShape[1]);
                    int nextChannels = Math.Min(channels + _groupPerPass * _icPerGroup, _inputShape[1]);
                    allocateResult = HandleAllocate(batch, nextChannels, ifmapRows, ifmapCols, kernelH, kernelW, nextOutChannels, ofmapRows, ofmapCols);
                    if (!allocateResult.IsOk)
                    {
                        break;
                    }

                    outChannels = nextOutChannels;
                    channels = nextChannels;
                }
            }
        }

        if (outChannels % GNNEEnv.PuWidth != 0 && outChannels > GNNEEnv.PuWidth && outChannels <= _ocPerGroup)
        {
            outChannels = outChannels / GNNEEnv.PuWidth * GNNEEnv.PuWidth;
        }

        allocateResult = HandleAllocate(batch, channels, ifmapRows, ifmapCols, kernelH, kernelW, outChannels, ofmapRows, ofmapCols, isFinal: true);
        TileUtilities.Assert(allocateResult.IsOk, "allocation.IsOk",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileTransposeMatmul.cs",
            970);
        GNNEShape glbShape = new GNNEShape(batch, outChannels, ofmapRows, ofmapCols);
        return new TileTransposeMatmulGlb(allocateResult.GlbMap, allocateResult.Items, glbShape.Dims,
            GNNEEnv.NPingPongSplit, allocateResult.GlbMap[ItemName.Ifmap], allocateResult.GlbMap[ItemName.Ofmap]);
    }

    private AllocateResult HandleAllocate(int n, int c, int h, int w, int r, int s, int m, int e, int f,
        bool isFinal = false)
    {
        List<float> glbUsage = new List<float> { 0f, 0f };
        AllocateResult boxes = GetBoxes(n, c, h, w, r, s, m, e, f, glbUsage);
        BoxPacker bp = new BoxPacker(16) { Boxes = boxes.Boxes };
        AllocateResult allocation = SpaceSearcher.TryAllocate(bp);
        allocation.GlbMap = boxes.GlbMap;
        if (!allocation.IsOk)
        {
            return allocation;
        }

        foreach (KeyValuePair<ItemName, TensorOnGlb> entry in boxes.GlbMap.Where<KeyValuePair<ItemName, TensorOnGlb>>((
                     KeyValuePair<ItemName, TensorOnGlb> map) => allocation.Items.ContainsKey(map.Key)))
        {
            entry.Value.Mmu = allocation.Items[entry.Key];
        }

        return allocation;
    }

    private List<int> L1Search(TiledGlb glb, SegmentND weight, SegmentND psum)
    {
        int puHeight = Math.Min(GNNEEnv.PuHeight, _groupPerPass * _icPerGroup);
        int puWidth = Math.Min(GNNEEnv.PuWidth, _groupPerPass * _ocPerGroup);
        int h = 1;
        int w = 1;
        int e = 1;
        int f = 1;
        int fStep = 1;
        int r = ((_weightSplitPattern.Item1 == 0 || _dilationH > 3) ? 1 : _weightSplitPattern.Item1);
        int s = ((_weightSplitPattern.Item2 == 0 || _dilationW > 3) ? 1 : _weightSplitPattern.Item2);
        int hConvOut = SpaceSearcher.GetInputHeight(e, _convOutputShape[2], _fusedKernelH, _outputShape[2],
            _fusedStrideH, _fusedDilationH, in _fusedPaddingH);
        h = SpaceSearcher.GetInputHeight(hConvOut, _inputShape[2] + _paddingH.Sum(), r, _convOutputShape[2], _strideH,
            _dilationH, new Padding(0, 0));
        int wConvOut = SpaceSearcher.GetInputHeight(f, _convOutputShape[3], _fusedKernelW, _outputShape[3],
            _fusedStrideW, _fusedDilationW, in _fusedPaddingW);
        w = SpaceSearcher.GetInputHeight(wConvOut, _inputShape[3] + _paddingW.Sum(), s, _convOutputShape[3], _strideW,
            _dilationW, new Padding(0, 0));
        int psumPingPangSplit = 1;
        if ((object)_pool != null || (object)_dw != null || (object)_act1 != null)
        {
            psumPingPangSplit = 2;
        }

        int ifBytesPerElementGlb = TileUtilities.GetBytesPerElement(_inputType);
        if (!HandleL1Allocate(h, w, Math.Max(e, hConvOut), Math.Max(f, wConvOut), psumPingPangSplit,
                ifBytesPerElementGlb))
        {
            throw new NotSupportedException("exceeds L1 size");
        }

        while (f < psum[3].Length && e < psum[2].Length &&
               h * w * ifBytesPerElementGlb < GNNEEnv.IfL1Size / GNNEEnv.PuHeight / 2)
        {
            if (f <= e)
            {
                if (!IncreaseFBy())
                {
                    break;
                }
            }
            else if (!IncreaseEBy())
            {
                break;
            }
        }

        while (f < psum[3].Length && h * w * ifBytesPerElementGlb < GNNEEnv.IfL1SizePerChan / 2 && IncreaseFBy())
        {
        }

        while (e < psum[2].Length && h * w * ifBytesPerElementGlb < GNNEEnv.IfL1SizePerChan / 2 && IncreaseEBy())
        {
        }

        if (_weightSplitPattern.Equals(new Tuple<int, int>(0, 0)))
        {
            while (s < weight[3].Length && r < weight[2].Length && s < 8 && r < 31 && _dilationW <= 3 &&
                   _dilationH <= 3)
            {
                if (s <= r)
                {
                    if (!IncreaseSBy())
                    {
                        break;
                    }
                }
                else if (!IncreaseRBy())
                {
                    break;
                }
            }

            while (s < weight[3].Length && s < 8 && _dilationW <= 3 && IncreaseSBy())
            {
            }

            while (r < weight[2].Length && r < 31 && _dilationH <= 3 && IncreaseRBy())
            {
            }
        }

        while (f < psum[3].Length && e < psum[2].Length)
        {
            if (f <= e)
            {
                if (!IncreaseFBy())
                {
                    break;
                }
            }
            else if (!IncreaseEBy())
            {
                break;
            }
        }

        while (f < psum[3].Length && IncreaseFBy())
        {
        }

        while (e < psum[2].Length && IncreaseEBy())
        {
        }

        if (_dilationW > 3)
        {
            r = 1;
        }

        if (_dilationH > 3)
        {
            s = 1;
        }

        _weightSplitPattern = new Tuple<int, int>(r, s);
        return new List<int>
        {
            puHeight,
            puWidth,
            e,
            f,
            r,
            s
        };

        bool IncreaseEBy()
        {
            int nextE = e + 1;
            int inputHeight = SpaceSearcher.GetInputHeight(nextE, _convOutputShape[2], _fusedKernelH, _outputShape[2],
                _fusedStrideH, _fusedDilationH, in _fusedPaddingH);
            int inputHeight2 = SpaceSearcher.GetInputHeight(inputHeight, _inputShape[2] + _paddingH.Sum(), r,
                _convOutputShape[2], _strideH, _dilationH, new Padding(0, 0));
            bool allocated = HandleL1Allocate(inputHeight2, w, Math.Max(nextE, inputHeight), Math.Max(f, wConvOut),
                psumPingPangSplit, ifBytesPerElementGlb);
            if (!allocated)
            {
                return allocated;
            }

            e = nextE;
            h = inputHeight2;
            hConvOut = inputHeight;
            return allocated;
        }

        bool IncreaseFBy()
        {
            int nextF = f + fStep;
            int inputHeight = SpaceSearcher.GetInputHeight(nextF, _convOutputShape[3], _fusedKernelW, _outputShape[3],
                _fusedStrideW, _fusedDilationW, in _fusedPaddingW);
            int inputHeight2 = SpaceSearcher.GetInputHeight(inputHeight, _inputShape[3] + _paddingW.Sum(), s,
                _convOutputShape[3], _strideW, _dilationW, new Padding(0, 0));
            bool allocated = HandleL1Allocate(h, inputHeight2, Math.Max(e, hConvOut), Math.Max(nextF, inputHeight),
                psumPingPangSplit, ifBytesPerElementGlb);
            if (!allocated)
            {
                return allocated;
            }

            f = nextF;
            w = inputHeight2;
            wConvOut = inputHeight;
            return allocated;
        }

        bool IncreaseRBy()
        {
            int nextR = r + 1;
            int inputHeight = SpaceSearcher.GetInputHeight(e, _convOutputShape[2], _fusedKernelH, _outputShape[2],
                _fusedStrideH, _fusedDilationH, in _fusedPaddingH);
            int inputHeight2 = SpaceSearcher.GetInputHeight(inputHeight, _inputShape[2] + _paddingH.Sum(), nextR,
                _convOutputShape[2], _strideH, _dilationH, new Padding(0, 0));
            bool allocated = HandleL1Allocate(inputHeight2, w, Math.Max(e, inputHeight), Math.Max(f, wConvOut),
                psumPingPangSplit, ifBytesPerElementGlb);
            if (allocated)
            {
                r = nextR;
                h = inputHeight2;
                hConvOut = inputHeight;
            }

            return allocated;
        }

        bool IncreaseSBy()
        {
            int nextS = s + 1;
            int inputHeight = SpaceSearcher.GetInputHeight(f, _convOutputShape[3], _fusedKernelW, _outputShape[3],
                _fusedStrideW, _fusedDilationW, in _fusedPaddingW);
            int inputHeight2 = SpaceSearcher.GetInputHeight(inputHeight, _inputShape[3] + _paddingW.Sum(), nextS,
                _convOutputShape[3], _strideW, _dilationW, new Padding(0, 0));
            bool allocated = HandleL1Allocate(h, inputHeight2, Math.Max(e, hConvOut), Math.Max(f, inputHeight),
                psumPingPangSplit, ifBytesPerElementGlb);
            if (allocated)
            {
                s = nextS;
                w = inputHeight2;
                wConvOut = inputHeight;
            }

            return allocated;
        }
    }

    private bool HandleL1Allocate(int h, int w, int e, int f, int psumPingPangSplit, int ifBytesPerElementGlb)
    {
        if (e * f > GNNEEnv.PsumL1ElePerChan / psumPingPangSplit)
        {
            return false;
        }

        if (GNNEEnv.PuHeight * h * w * ifBytesPerElementGlb > GNNEEnv.IfL1Size)
        {
            return false;
        }

        return true;
    }

    private void BuildL1Schedule(GnneActionUpdater actionUpdater, TiledGlb glb, SegmentND ifmap, SegmentND weight,
        SegmentND psum, int iPp, int ofPp, int wPp, SegmentND ifmap2, bool weightGroupOnly,
        WeightGroupHandler weightGroup, int l1Pp, Nncase.TIR.Buffer ddrW, Nncase.TIR.Buffer ddrOf)
    {
        List<int> l1Params = L1Search(glb, weight, psum);
        List<Segment1D> psumRowSegments =
            TileUtilities.GetSegmentStartEndLength(psum[2].Start, l1Params[2], psum[2].End);
        List<Segment1D> psumColSegments =
            TileUtilities.GetSegmentStartEndLength(psum[3].Start, l1Params[3], psum[3].End);
        List<Segment1D> psumBatchSegments = TileUtilities.GetSegmentStartEndLength(psum[0].Start, 1, psum[0].End);
        List<Segment1D> kernelRowSegments =
            TileUtilities.GetSegmentStartEndLength(weight[2].Start, l1Params[4], weight[2].End);
        List<Segment1D> kernelColSegments =
            TileUtilities.GetSegmentStartEndLength(weight[3].Start, l1Params[5], weight[3].End);
        List<List<Segment1D>> l1McSeg = GetL1McSeg(ifmap, psum, l1Params[1], l1Params[0]);
        List<Segment1D> ocSegments = l1McSeg[0];
        List<Segment1D> icSegments = l1McSeg[1];
        int chunkSize = l1Params[4];
        int chunkSize2 = Math.Min(GNNEEnv.PuKernelSpad / 2, l1Params[5]);
        if (_dilationH > 1)
        {
            chunkSize = 1;
        }

        if (_dilationW > 1)
        {
            chunkSize2 = 1;
        }

        int pingPongIndex = 0;
        foreach (Segment1D ocSegment in ocSegments)
        {
            foreach (Segment1D batchSegment in psumBatchSegments)
            {
                foreach (Segment1D rowSegment in psumRowSegments)
                {
                    Segment1D convInputRowSegment = TileUtilities.GetInputRowSegment(rowSegment.Start, rowSegment.Length,
                        _convOutputShape[2], _fusedKernelH, _fusedStrideH, _fusedDilationH, in _fusedPaddingH);
                    Segment1D ifmapRowSegment = TileUtilities.GetInputRowSegment(convInputRowSegment.Start,
                        convInputRowSegment.Length, _inputShape[2], _weightsShape[2], _strideH, _dilationH, in _paddingH);
                    foreach (Segment1D colSegment in psumColSegments)
                    {
                        Segment1D convInputColumnSegment = TileUtilities.GetInputColumnSegment(colSegment.Start, colSegment.Length,
                            _convOutputShape[3], _fusedKernelW, _fusedStrideW, _fusedDilationW, in _fusedPaddingW);
                        Segment1D ifmapColumnSegment = TileUtilities.GetInputColumnSegment(convInputColumnSegment.Start,
                            convInputColumnSegment.Length, _inputShape[3], _weightsShape[3], _strideW, _dilationW,
                            in _paddingW);
                        SegmentND l2GOf = new SegmentND(batchSegment, ocSegment, rowSegment, colSegment);
                        SegmentND convSegment = new SegmentND(batchSegment, ocSegment, convInputRowSegment, convInputColumnSegment);
                        int icGroupStart = ocSegment.Start / _ocPerGroup * _icPerGroup;
                        int icGroupEnd = (ocSegment.End - 1) / _ocPerGroup * _icPerGroup + _icPerGroup;
                        Segment1D icSegmentInGroup = new Segment1D(0..0, new Padding(0, 0));
                        bool isFirstLoop = true;
                        foreach (Segment1D icSegment in icSegments)
                        {
                            if (icSegment.Start < icGroupStart || icSegment.End > icGroupEnd)
                            {
                                continue;
                            }

                            int icStartInGroup = icSegment.Start % _icPerGroup;
                            icSegmentInGroup = new Segment1D(
                                new System.Range(end: Math.Min(icStartInGroup + icSegment.Length, _icPerGroup), start: icStartInGroup),
                                new Padding(0, 0));
                            foreach (Segment1D kernelRowSegment in kernelRowSegments)
                            {
                                foreach (Segment1D kernelColSegment in kernelColSegments)
                                {
                                    Segment1D ifmapIcSegment = icSegment;
                                    List<Segment1D> kernelRowSlices =
                                        TileUtilities.GetSegmentStartEndLength(kernelRowSegment.Start, chunkSize, kernelRowSegment.End);
                                    List<Segment1D> kernelColSlices =
                                        TileUtilities.GetSegmentStartEndLength(kernelColSegment.Start, chunkSize2, kernelColSegment.End);
                                    Segment1D ifmapBatchSegment = batchSegment;
                                    Segment1D weightOcSegment = ocSegment;
                                    SegmentND ifmapTile = new SegmentND(ifmapBatchSegment, ifmapIcSegment, ifmapRowSegment,
                                        ifmapColumnSegment);
                                    SegmentND weightTile = new SegmentND(ocSegment, icSegmentInGroup, kernelRowSegment, kernelColSegment);
                                    foreach (Segment1D kernelRowSlice in kernelRowSlices)
                                    {
                                        foreach (Segment1D kernelColSlice in kernelColSlices)
                                        {
                                            SegmentND weightSlice = new SegmentND(weightOcSegment, icSegmentInGroup, kernelRowSlice, kernelColSlice);
                                            SegmentND slice = TileUtilities.ShiftInputTensor(in ifmapTile, in weightSlice,
                                                _weightsShape[2], _weightsShape[3], _strideH, _strideW, _dilationH,
                                                _dilationW);
                                            int weightPasses = ((!(_weightType == DataTypes.Int16)) ? 1 : 2);
                                            int inputPasses = ((!(_inputType == DataTypes.Int16)) ? 1 : 2);
                                            for (int weightPass = 0; weightPass < weightPasses; weightPass++)
                                            {
                                                for (int inputPass = 0; inputPass < inputPasses; inputPass++)
                                                {
                                                    if (!weightGroupOnly)
                                                    {
                                                        _weightPreloadEn = false;
                                                        if (!_weightPreloadEn)
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
                                                                weightSlice == g2RWSliceRecStat.Item1 && g2RWRecStat.Item1 == weightRecStat.Item1,
                                                                "l2RW == g2RWSliceRecStat.Item1 && g2RWRecStat.Item1 == weightRecStat.Item1",
                                                                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileTransposeMatmul.cs",
                                                                1376);
                                                            if (g2RWSliceRecStat.Item2.IsFirstSlice)
                                                            {
                                                                int clearWeightFakeCcr = ((!weightRecStat.Item2.IsFirstSlice &&
                                                                             g2RWRecStat.Item2.IsFirstSlice)
                                                                    ? 1
                                                                    : 0);
                                                                List<CcrClr> weightFakeCcrsToClr = new List<CcrClr>();
                                                                if (clearWeightFakeCcr > 0)
                                                                {
                                                                    weightFakeCcrsToClr.Add(new CcrClr(
                                                                        _ccrHandler.GetCcrItem(
                                                                            _ccrHandler.GetName(ItemName.WeightFake,
                                                                                wPp))));
                                                                }

                                                                List<CcrSet> ccrsToSet = new List<CcrSet>
                                                                {
                                                                    new CcrSet(
                                                                        _ccrHandler.GetCcrItem(
                                                                            _ccrHandler.GetName(ItemName.Weight,
                                                                                (wPp << 1) +
                                                                                (g2RWSliceRecStat.Item2.SliceIdx & 1))), 1)
                                                                };
                                                                actionUpdater.UpdateLoadW(weightSlice, _lw, weightGroup, wPp,
                                                                    ddrW, ccrsToSet, weightFakeCcrsToClr, 0, _h2C, 0, ItemName.Weight,
                                                                    weightPass);
                                                            }

                                                            int setWeightFakeCcr = 0;
                                                            if (g2RWRecStat.Item2.IsLastSlice && _weightRec[wPp].Count > 0)
                                                            {
                                                                setWeightFakeCcr = 1;
                                                            }

                                                            List<CcrSet> weightFakeCcrsToSet = new List<CcrSet>();
                                                            if (setWeightFakeCcr > 0)
                                                            {
                                                                weightFakeCcrsToSet.Add(new CcrSet(
                                                                    _ccrHandler.GetCcrItem(
                                                                        _ccrHandler.GetName(ItemName.WeightFake, wPp)),
                                                                    1));
                                                            }

                                                            List<CcrClr> weightCcrsToClr = new List<CcrClr>();
                                                            if (g2RWSliceRecStat.Item2.IsFirstSlice)
                                                            {
                                                                weightCcrsToClr.Add(new CcrClr(
                                                                    _ccrHandler.GetCcrItem(
                                                                        _ccrHandler.GetName(ItemName.Weight,
                                                                            (wPp << 1) +
                                                                            (g2RWSliceRecStat.Item2.SliceIdx & 1)))));
                                                            }

                                                            actionUpdater.UpdateG2RW(weightSlice, weightGroup, _ocPerGroup, _lw,
                                                                wPp, weightPass, weightFakeCcrsToSet, weightCcrsToClr, 0, 0, ItemName.Weight,
                                                                ItemName.WQarg, _h2C);
                                                        }
                                                        else
                                                        {
                                                            actionUpdater.UpdateG2RW(weightSlice, weightGroup, _ocPerGroup, _lw,
                                                                wPp, weightPass, null, null, 0, 0, ItemName.Weight,
                                                                ItemName.WQarg, _h2C);
                                                        }
                                                    }
                                                    else
                                                    {
                                                        weightGroup.UpdateWeightGroup(weightSlice);
                                                        if (!_weightPreloadEn)
                                                        {
                                                            _g2RWRec[wPp].Add(new Tuple<SegmentND, TensorStat>(weight,
                                                                new TensorStat(isFirstSlice: false,
                                                                    isLastSlice: false)));
                                                            _g2RWSliceRec[wPp].Add(new Tuple<SegmentND, TensorStat>(weightSlice,
                                                                new TensorStat(isFirstSlice: false, isLastSlice: false,
                                                                    -1)));
                                                        }
                                                    }

                                                    bool hasIfmapLoad = false;
                                                    SegmentND shiftedIfmap = TileUtilities.ShiftInputTensor(in ifmapTile, in weightTile,
                                                        _weightsShape[2], _weightsShape[3], _strideH, _strideW,
                                                        _dilationH, _dilationW);
                                                    if (shiftedIfmap[0].Length > 0 && shiftedIfmap[1].Length > 0 &&
                                                        shiftedIfmap[2].Length > 0 && shiftedIfmap[3].Length > 0)
                                                    {
                                                        hasIfmapLoad = true;
                                                        if (_tp1)
                                                        {
                                                            if (inputPass != 1 && weightPass != 1)
                                                            {
                                                                ifmap2 = new SegmentND(ifmap[0], ifmap[2], ifmap[3],
                                                                    ifmap[1]);
                                                                if (!weightGroupOnly)
                                                                {
                                                                    int ifmapStatCount = _g2LIfRec[iPp][0].Item2.Stat_cnt();
                                                                    _g2LIfRec[iPp].RemoveAt(0);
                                                                    List<CcrClr> ifmap2CcrsToClr = new List<CcrClr>();
                                                                    if (ifmapStatCount > 0)
                                                                    {
                                                                        ifmap2CcrsToClr.Add(new CcrClr(
                                                                            _ccrHandler.GetCcrItem(
                                                                                _ccrHandler.GetName(ItemName.Ifmap2,
                                                                                    iPp))));
                                                                    }

                                                                    SegmentND transposedIfmap = new SegmentND(shiftedIfmap[0],
                                                                        shiftedIfmap[2], shiftedIfmap[3], shiftedIfmap[1]);
                                                                    List<CcrSet> ifmapTransposeCcrsToSet = new List<CcrSet>();
                                                                    ifmapTransposeCcrsToSet.Add(new CcrSet(
                                                                        _ccrHandler.GetCcrItem(
                                                                            _ccrHandler.GetName(ItemName.Ifmap, pingPongIndex)),
                                                                        1));
                                                                    int dim = (transposedIfmap[0].Start - ifmap2[0].Start) %
                                                                              ifmap2[0].Length;
                                                                    int dim2 = (transposedIfmap[1].Start - ifmap2[1].Start) %
                                                                               ifmap2[1].Length;
                                                                    int dim3 = (transposedIfmap[2].Start - ifmap2[2].Start) %
                                                                               ifmap2[2].Length;
                                                                    int dim4 = (transposedIfmap[3].Start - ifmap2[3].Start) %
                                                                               ifmap2[3].Length;
                                                                    int addr = glb.GlbMap[ItemName.Ifmap2]
                                                                        .GetAddr(dim, dim2, dim3, dim4);
                                                                    int offsetD = 0;
                                                                    actionUpdater.UpdateMfuTranspose(transposedIfmap,
                                                                        shiftedIfmap, _inputType, MFU_TRANS_PERMUTE.NWCH,
                                                                        iPp, ifmapTransposeCcrsToSet, ifmap2CcrsToClr, addr, offsetD,
                                                                        ItemName.Ifmap2, ItemName.Ifmap, pingPongIndex);
                                                                    ifmap2CcrsToClr = new List<CcrClr>();
                                                                    ifmap2CcrsToClr.Add(new CcrClr(
                                                                        _ccrHandler.GetCcrItem(
                                                                            _ccrHandler.GetName(ItemName.Ifmap, pingPongIndex))));
                                                                    actionUpdater.UpdateG2LIf(shiftedIfmap, ifmap, _lif,
                                                                        pingPongIndex, null, ifmap2CcrsToClr, 0, DataTypes.UInt8,
                                                                        ItemName.Ifmap, restored: false, 0, 0, 0, _h2C,
                                                                        _weightsShape[2], _strideH, slice_from: false);
                                                                }
                                                                else
                                                                {
                                                                    _g2LIfRec[iPp]
                                                                        .Add(new Tuple<SegmentND, TensorStat>(ifmap2,
                                                                            new TensorStat(isFirstSlice: false,
                                                                                isLastSlice: false)));
                                                                }
                                                            }

                                                            if (inputPasses == 2)
                                                            {
                                                                if (inputPass == 1)
                                                                {
                                                                    pingPongIndex = (pingPongIndex + 1) % 2;
                                                                }
                                                            }
                                                            else
                                                            {
                                                                pingPongIndex = (pingPongIndex + 1) % 2;
                                                            }
                                                        }
                                                        else if (!weightGroupOnly)
                                                        {
                                                            int ifmapStatCount2 = _g2LIfRec[iPp][0].Item2.Stat_cnt();
                                                            _g2LIfRec[iPp].RemoveAt(0);
                                                            List<CcrClr> ifmapCcrsToClr = new List<CcrClr>();
                                                            if (ifmapStatCount2 > 0)
                                                            {
                                                                ifmapCcrsToClr.Add(new CcrClr(
                                                                    _ccrHandler.GetCcrItem(
                                                                        _ccrHandler.GetName(ItemName.Ifmap, iPp))));
                                                            }

                                                            actionUpdater.UpdateG2LIf(shiftedIfmap, ifmap, _lif, iPp,
                                                                null, ifmapCcrsToClr, 0, DataTypes.UInt8, ItemName.Ifmap,
                                                                restored: false, 0, 0, 0, _h2C, _weightsShape[2],
                                                                _strideH);
                                                        }
                                                        else
                                                        {
                                                            _g2LIfRec[iPp].Add(new Tuple<SegmentND, TensorStat>(ifmap,
                                                                new TensorStat(isFirstSlice: false,
                                                                    isLastSlice: false)));
                                                        }
                                                    }

                                                    if (!weightGroupOnly)
                                                    {
                                                        actionUpdater.UpdateL2RIf(slice, shiftedIfmap, _strideH, _strideW,
                                                            _icPerGroup, _lif, 0, inputPass,
                                                            ((TensorConst)_conv[GNNEConv2D.DeqBias]).Value
                                                            .ToScalar<int>(), DataTypes.UInt8, _h2C, _weightsShape[2]);
                                                    }

                                                    bool releaseIf = false;
                                                    int isLastPass = 0;
                                                    if (kernelColSlice == kernelColSlices[
                                                            kernelColSlices.Count - 1])
                                                    {
                                                        if (kernelRowSlice == kernelRowSlices[
                                                                kernelRowSlices.Count - 1] && weightPass == weightPasses - 1)
                                                        {
                                                            isLastPass = ((inputPass == inputPasses - 1) ? 1 : 0);
                                                        }
                                                    }

                                                    if (((uint)isLastPass & (hasIfmapLoad ? 1u : 0u)) != 0)
                                                    {
                                                        releaseIf = true;
                                                        hasIfmapLoad = false;
                                                    }

                                                    bool loopStart = false;
                                                    if (isFirstLoop && weightPass == 0 && inputPass == 0)
                                                    {
                                                        loopStart = true;
                                                        isFirstLoop = false;
                                                    }

                                                    bool isLastKernelSlice = icSegmentInGroup.End == _weightsShape[1] &&
                                                                 kernelColSlice.End == weight[3].End &&
                                                                 kernelRowSlice.End == weight[2].End && weightPass == weightPasses - 1 &&
                                                                 inputPass == inputPasses - 1;
                                                    DataType ofType = _outputType;
                                                    ACT0_OUTPUT_DEST act0OutputDest = ACT0_OUTPUT_DEST.dm;
                                                    if ((object)_pool != null || (object)_dw != null ||
                                                        (object)_act1 != null)
                                                    {
                                                        act0OutputDest = ACT0_OUTPUT_DEST.psum;
                                                        ofType = _conv.CheckedDataType;
                                                    }

                                                    if (!weightGroupOnly)
                                                    {
                                                        int shift = ((TensorConst)_conv[GNNEConv2D.ShiftBits]).Value
                                                            .ToScalar<int>();
                                                        int setOfmapCcr = 0;
                                                        if (isLastKernelSlice && act0OutputDest == ACT0_OUTPUT_DEST.dm)
                                                        {
                                                            setOfmapCcr = 1;
                                                        }

                                                        List<CcrSet> ofmapCcrsToSet = new List<CcrSet>();
                                                        if (setOfmapCcr > 0)
                                                        {
                                                            ofmapCcrsToSet.Add(new CcrSet(
                                                                _ccrHandler.GetCcrItem(
                                                                    _ccrHandler.GetName(ItemName.Ofmap, ofPp)), 1));
                                                        }

                                                        int offsetL2GOf = 0;
                                                        actionUpdater.UpdateR2LPsum(shift, convSegment, convSegment, psum,
                                                            ofPp, l1Pp, act0OutputDest, releaseIf, Math.Max(weightPass, inputPass),
                                                            TcuComputeMode.NormalConv2d, loopStart, isLastKernelSlice, _strideH,
                                                            _strideW, _ocPerGroup, _inputType, _weightType, ofType,
                                                            _lact.CheckedDataType, ofmapCcrsToSet, null, 0, offsetL2GOf, null,
                                                            ItemName.Act, l2g_of_slice_from: false);
                                                        if (isLastKernelSlice && act0OutputDest == ACT0_OUTPUT_DEST.dm)
                                                        {
                                                            SegmentND transposedConvSegment = new SegmentND(convSegment[0],
                                                                convSegment[2], convSegment[3], convSegment[1]);
                                                            List<CcrClr> ofmapCcrsToClr = new List<CcrClr>();
                                                            ofmapCcrsToClr.Add(new CcrClr(
                                                                _ccrHandler.GetCcrItem(
                                                                    _ccrHandler.GetName(ItemName.Ofmap, ofPp))));
                                                            List<CcrSet> ofmap2CcrsToSet = new List<CcrSet>();
                                                            ofmap2CcrsToSet.Add(new CcrSet(
                                                                _ccrHandler.GetCcrItem(
                                                                    _ccrHandler.GetName(ItemName.Ofmap2, ofPp)), 1));
                                                            int offsetS = 0;
                                                            int offsetD2 = 0;
                                                            actionUpdater.UpdateMfuTranspose(convSegment, transposedConvSegment,
                                                                _outputType, MFU_TRANS_PERMUTE.NHWC, ofPp, ofmap2CcrsToSet,
                                                                ofmapCcrsToClr, offsetS, offsetD2, ItemName.Ofmap,
                                                                ItemName.Ofmap2);
                                                            List<CcrClr> ofmap2CcrsToClr = new List<CcrClr>();
                                                            ofmap2CcrsToClr.Add(new CcrClr(
                                                                _ccrHandler.GetCcrItem(
                                                                    _ccrHandler.GetName(ItemName.Ofmap2, ofPp))));
                                                            SegmentND ofmapStDdr = new SegmentND(transposedConvSegment[2],
                                                                transposedConvSegment[3], transposedConvSegment[0], transposedConvSegment[1]);
                                                            List<int> stridesD = new List<int>
                                                            {
                                                                ddrOf.FixedDimensions(2),
                                                                ddrOf.FixedDimensions(0),
                                                                ddrOf.FixedDimensions(1)
                                                            };
                                                            GNNEShape shape = new GNNEShape(transposedConvSegment[0].Length,
                                                                transposedConvSegment[1].Length, transposedConvSegment[2].Length,
                                                                transposedConvSegment[3].Length);
                                                            actionUpdater.UpdateStoreT(transposedConvSegment, _sof, ofPp, ddrOf, 0,
                                                                ofmapStDdr, null, ofmap2CcrsToClr, ItemName.Ofmap2, null,
                                                                stridesD, shape);
                                                            ofPp = (ofPp + 1) % 2;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (!weightGroupOnly)
                        {
                            if (((object)_pool == null && (object)_dw == null && (object)_act1 == null) ||
                                icSegmentInGroup.End != _weightsShape[1])
                            {
                                continue;
                            }

                            SegmentND l2RDw = new SegmentND();
                            SegmentND l2RIf = new SegmentND();
                            List<CcrClr> ifmap2CcrsToClr2 = new List<CcrClr>();
                            if ((object)_act1 != null && _act1[GNNEActivation.InputB] != None.Default)
                            {
                                l2RIf = convSegment;
                                int ifmap2StatCount = _l2RIf2Rec[ofPp][0].Item2.Stat_cnt();
                                _l2RIf2Rec[ofPp].RemoveAt(0);
                                for (int ccrIndex = 0; ccrIndex < ifmap2StatCount; ccrIndex++)
                                {
                                    ifmap2CcrsToClr2.Add(new CcrClr(
                                        _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap2, ofPp))));
                                }
                            }

                            if ((object)_dw != null)
                            {
                                int[] subArray = _dw[GNNEPdp0DW.Weights].CheckedShape.ToValueArray()[2..];
                                l2RDw = new SegmentND(new Segment1D(0..1, new Padding(0, 0)), ocSegment,
                                    new Segment1D(0..subArray[0], new Padding(0, 0)),
                                    new Segment1D(0..subArray[1], new Padding(0, 0)));
                            }

                            int psumSetOfmapCcr = 0;
                            int psumClearOfmapFake = 0;
                            Tuple<SegmentND, TensorStat> l2GOfRecStat = _l2GOfRec[ofPp][0];
                            _l2GOfRec[ofPp].RemoveAt(0);
                            if (l2GOfRecStat.Item2.IsFirstSlice && !_ofmapRec[ofPp][0].Item2.IsFirstSlice)
                            {
                                psumClearOfmapFake = 1;
                            }

                            if (l2GOfRecStat.Item2.IsLastSlice)
                            {
                                psumSetOfmapCcr = 1;
                            }

                            List<CcrSet> psumOfmapCcrsToSet = new List<CcrSet>();
                            if (psumSetOfmapCcr > 0)
                            {
                                psumOfmapCcrsToSet.Add(new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, ofPp)),
                                    1));
                            }

                            List<CcrClr> psumOfmapFakeCcrsToClr = new List<CcrClr>();
                            if (psumClearOfmapFake > 0)
                            {
                                psumOfmapFakeCcrsToClr.Add(new CcrClr(
                                    _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.OfmapFake, ofPp))));
                            }

                            actionUpdater.UpdateL2RPsum(_dw, _pool, _act1, l2RIf, ifmap2, l2RDw, convSegment, l2GOf, psum,
                                ofPp, l1Pp, ACT0_OUTPUT_DEST.dm, TcuComputeMode.NormalConv2d, _ocPerGroup, weightGroup,
                                ifmap2CcrsToClr2, psumOfmapCcrsToSet, psumOfmapFakeCcrsToClr);
                            l1Pp = (l1Pp + 1) % 2;
                        }
                        else
                        {
                            if ((object)_act1 != null && icSegmentInGroup.End == _weightsShape[1] &&
                                _act1[GNNEActivation.InputB] != None.Default)
                            {
                                _l2RIf2Rec[ofPp].Add(new Tuple<SegmentND, TensorStat>(ifmap2,
                                    new TensorStat(isFirstSlice: false, isLastSlice: false)));
                            }

                            if (((object)_pool != null || (object)_dw != null || (object)_act1 != null) &&
                                icSegmentInGroup.End == _weightsShape[1])
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

    private List<List<Segment1D>> GetL1McSeg(SegmentND ifmap, SegmentND psum, int mInloop, int cInloop)
    {
        int groupCount = Math.Max(ifmap[1].Length / _icPerGroup, 1);
        List<Segment1D> ocSegments = new List<Segment1D>();
        List<Segment1D> icSegments = new List<Segment1D>();
        if (groupCount >= 2 && _groupPerPass < 2)
        {
            for (int groupIndex = 0; groupIndex < groupCount; groupIndex++)
            {
                List<Segment1D> groupOcSegments =
                    TileUtilities.GetSegmentStartEndLength(psum[1].Start + groupIndex * _ocPerGroup, mInloop,
                        psum[1].Start + (groupIndex + 1) * _ocPerGroup);
                List<Segment1D> groupIcSegments =
                    TileUtilities.GetSegmentStartEndLength(ifmap[1].Start + groupIndex * _icPerGroup, cInloop,
                        ifmap[1].Start + (groupIndex + 1) * _icPerGroup);
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

    private AllocateResult GetBoxes(int n, int c, int h, int w, int r, int s, int m, int e, int f, List<float> glbUsage)
    {
        List<BoxOnGlb> boxes = new List<BoxOnGlb>();
        Dictionary<ItemName, TensorOnGlb> glbMap = new Dictionary<ItemName, TensorOnGlb>();
        List<float> glbUsageBytes = new List<float> { 0f, 0f };
        int nPingPongSplit = GNNEEnv.NPingPongSplit;
        if (_h2C)
        {
            h += _paddingH.Sum();
        }

        TensorOnGlb ifmapTransposedTensor = new TensorOnGlb(new int[4] { 0, 0, 0, 0 }, DataTypes.UInt8, 0);
        int ifmapTransposedBytes = 0;
        if (_tp1)
        {
            int alignmentFactor = ((TileUtilities.GetBytesPerElement(_inputType) == 1) ? 32 : 16);
            int alignedNum = TileUtilities.GetAlignedNum(c, alignmentFactor);
            ifmapTransposedTensor = new TensorOnGlb(new int[4] { n, h, w, alignedNum }, _inputType, 0);
            ifmapTransposedBytes = ifmapTransposedTensor.AllocatedBytes * nPingPongSplit;
            glbUsageBytes[0] += ifmapTransposedBytes;
            ifmapTransposedBytes = TileUtilities.GetAlignedNum(ifmapTransposedBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
            glbUsageBytes[1] += ifmapTransposedBytes;
        }

        int ifmapWidth = w;
        int ifmapAlignment = ((TileUtilities.GetBytesPerElement(_inputType) == 1) ? 32 : 16);
        if (_tp1)
        {
            ifmapWidth = TileUtilities.GetAlignedNum(GNNEEnv.IfL1Size / GNNEEnv.PuHeight, ifmapAlignment);
        }

        TensorOnGlb ifmapTensor =
            new TensorOnGlb(new int[4] { n, _tp1 ? GNNEEnv.PuHeight : c, h, ifmapWidth }, _inputType, 0);
        int ifmapBytes = ifmapTensor.AllocatedBytes * nPingPongSplit;
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
        int ofmapWidth = f;
        int ofmapAlignment = ((TileUtilities.GetBytesPerElement(_outputType) == 1) ? 32 : 16);
        if (_tp2)
        {
            ofmapWidth = TileUtilities.GetAlignedNum(GNNEEnv.IfL1Size / GNNEEnv.PuHeight, ofmapAlignment);
        }

        TensorOnGlb ofmapTensor =
            new TensorOnGlb(new int[4] { n, _tp2 ? GNNEEnv.PuWidth : m, e, ofmapWidth }, _outputType, 0);
        int ofmapBytes = ofmapTensor.AllocatedBytes * GNNEEnv.NPingPongSplit;
        glbUsageBytes[0] += ofmapBytes;
        ofmapBytes = TileUtilities.GetAlignedNum(ofmapBytes, GNNEEnv.OfmapBankWidth * GNNEEnv.GlbBankWidth);
        glbUsageBytes[1] += ofmapBytes;
        TensorOnGlb ofmap2Tensor = new TensorOnGlb(new int[4] { n, e, ofmapWidth, GNNEEnv.PuWidth }, _outputType, 0);
        int ofmap2Bytes = ofmapTensor.AllocatedBytes * GNNEEnv.NPingPongSplit;
        glbUsageBytes[0] += ofmap2Bytes;
        ofmap2Bytes = TileUtilities.GetAlignedNum(ofmap2Bytes, GNNEEnv.OfmapBankWidth * GNNEEnv.GlbBankWidth);
        glbUsageBytes[1] += ofmap2Bytes;
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
        if (_tp1)
        {
            boxes.Add(new BoxOnGlb(
                new int[2] { GNNEEnv.IfmapBankWidth, ifmapTransposedBytes / GNNEEnv.IfmapBankWidth / GNNEEnv.GlbBankWidth },
                ItemName.Ifmap2));
            glbMap.Add(ItemName.Ifmap2, ifmapTransposedTensor);
        }

        if (_tp2)
        {
            boxes.Add(new BoxOnGlb(
                new int[2] { GNNEEnv.OfmapBankWidth, ofmap2Bytes / GNNEEnv.OfmapBankWidth / GNNEEnv.GlbBankWidth },
                ItemName.Ofmap2));
            glbMap.Add(ItemName.Ofmap2, ofmap2Tensor);
        }

        if (_weightType == DataTypes.UInt8 || _weightType == DataTypes.Int16)
        {
            int[] wQargShape = new int[4] { 1, 1, 1, 0 };
            wQargShape[3] = _convOutputShape[1];
            TensorOnGlb wQargTensor = new TensorOnGlb(wQargShape, _lwQarg.CheckedDataType, 0);
            int wQargSize = SpaceSearcher.GetWQargSize(_convOutputShape[1],
                TileUtilities.GetBytesPerElement(_lwQarg.CheckedDataType));
            if (_inputShape[1] == _outputShape[1] && _outputShape[1] == _groups && _groups != 1)
            {
                wQargSize = SpaceSearcher.GetWQargSize(
                    (int)Math.Ceiling(1.0 * (double)_convOutputShape[1] / (double)GNNEEnv.PuHeight) * GNNEEnv.PuWidth,
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

        if ((object)_dw != null)
        {
            int fusedKernelH = _fusedKernelH;
            int fusedKernelW = _fusedKernelW;
            int dwChannels = _outputShape[1];
            DataType dwWeightType = _dw[GNNEPdp0DW.Weights].CheckedDataType;
            TensorOnGlb dwWeightTensor =
                new TensorOnGlb(new int[4] { dwChannels, 1, fusedKernelH, fusedKernelW }, dwWeightType, 0);
            int weightSize = SpaceSearcher.GetWeightSize(fusedKernelH, fusedKernelW,
                TileUtilities.GetAlignedNum(dwChannels, GNNEEnv.PuWidth), 1,
                TileUtilities.GetBytesPerElement(dwWeightType));
            glbUsageBytes[0] += weightSize;
            weightSize = TileUtilities.GetAlignedNum(weightSize, GNNEEnv.WBankWidth * GNNEEnv.GlbBankWidth);
            glbUsageBytes[1] += weightSize;
            boxes.Add(new BoxOnGlb(
                new int[2] { GNNEEnv.WBankWidth, weightSize / GNNEEnv.WBankWidth / GNNEEnv.GlbBankWidth },
                ItemName.DwWeight));
            glbMap.Add(ItemName.DwWeight, dwWeightTensor);
            TensorOnGlb dwActTensor = new TensorOnGlb(new int[4] { 1, 1, dwChannels, GNNEEnv.ActNumPerChan },
                _dw[GNNEPdp0DW.Act].CheckedDataType, 0);
            int actSize2 = SpaceSearcher.GetActSize(dwChannels, GNNEEnv.ActNumPerChan,
                TileUtilities.GetBytesPerElement(_dw[GNNEPdp0DW.Act].CheckedDataType));
            glbUsageBytes[0] += actSize2;
            actSize2 = TileUtilities.GetAlignedNum(actSize2, GNNEEnv.ActBankWidth * GNNEEnv.GlbBankWidth);
            glbUsageBytes[1] += actSize2;
            boxes.Add(new BoxOnGlb(
                new int[2] { GNNEEnv.ActBankWidth, actSize2 / GNNEEnv.ActBankWidth / GNNEEnv.GlbBankWidth },
                ItemName.DwAct1));
            glbMap.Add(ItemName.DwAct1, dwActTensor);
            if (dwWeightType == DataTypes.UInt8 || dwWeightType == DataTypes.Int16)
            {
                int[] dwQargShape = new int[4] { 1, 1, 1, 0 };
                dwQargShape[3] = dwChannels;
                TensorOnGlb dwQargTensor = new TensorOnGlb(dwQargShape, dwWeightType, 0);
                int wQargSize2 = SpaceSearcher.GetWQargSize(dwChannels, TileUtilities.GetBytesPerElement(dwWeightType));
                glbUsageBytes[0] += wQargSize2;
                wQargSize2 = TileUtilities.GetAlignedNum(wQargSize2, GNNEEnv.WQargBankWidth * GNNEEnv.GlbBankWidth);
                glbUsageBytes[1] += wQargSize2;
                boxes.Add(new BoxOnGlb(
                    new int[2] { GNNEEnv.WQargBankWidth, wQargSize2 / GNNEEnv.WQargBankWidth / GNNEEnv.GlbBankWidth },
                    ItemName.DwQarg));
                glbMap.Add(ItemName.DwQarg, dwQargTensor);
            }
        }

        TensorOnGlb ifmap2Tensor = new TensorOnGlb(new int[4] { 0, 0, 0, 0 }, DataTypes.Float16, 0);
        if ((object)_act1 != null && !_tp1)
        {
            bool is16Segments = ((TensorConst)_act1[GNNEActivation.Is16Segments]).Value.ToScalar<bool>();
            int actChannels = (is16Segments ? 1 : _outputShape[1]);
            int actNumPerChan = (is16Segments ? 49 : GNNEEnv.ActNumPerChan);
            TensorOnGlb act1Tensor = new TensorOnGlb(new int[4] { 1, 1, actChannels, actNumPerChan },
                _act1[GNNEActivation.Act].CheckedDataType, 0);
            int actSize3 = SpaceSearcher.GetActSize(actChannels, actNumPerChan,
                TileUtilities.GetBytesPerElement(_act1[GNNEActivation.Act].CheckedDataType));
            glbUsageBytes[0] += actSize3;
            actSize3 = TileUtilities.GetAlignedNum(actSize3, GNNEEnv.ActBankWidth * GNNEEnv.GlbBankWidth);
            glbUsageBytes[1] += actSize3;
            boxes.Add(new BoxOnGlb(
                new int[2] { GNNEEnv.ActBankWidth, actSize3 / GNNEEnv.ActBankWidth / GNNEEnv.GlbBankWidth },
                ItemName.MfuAct1));
            glbMap.Add(ItemName.MfuAct1, act1Tensor);
            if (_act1[GNNEActivation.InputB] != None.Default)
            {
                Call inputACall = (Call)_act1[GNNEActivation.InputA];
                Call inputBCall = (Call)_act1[GNNEActivation.InputB];
                DataType dataType = DataTypes.Float16;
                if ((object)inputACall != null && inputACall.Target is GNNELoad)
                {
                    dataType = inputACall.CheckedDataType;
                }
                else if ((object)inputBCall != null && inputBCall.Target is GNNELoad)
                {
                    dataType = inputBCall.CheckedDataType;
                }

                int ifmap2E = e;
                int ifmap2F = f;
                int alignElements = 32 / TileUtilities.GetBytesPerElement(dataType);
                if (e * f % alignElements != 0)
                {
                    int eLimit = e + alignElements - 1;
                    int fLimit = f + alignElements - 1;
                    long bestArea = 4294836225L;
                    for (int i = e; i < eLimit; i++)
                    {
                        for (int j = f; j < fLimit; j++)
                        {
                            int area = i * j;
                            if (area % alignElements == 0 && area < bestArea)
                            {
                                bestArea = area;
                                ifmap2E = i;
                                ifmap2F = j;
                            }
                        }
                    }
                }

                ifmap2Tensor = new TensorOnGlb(new int[4] { n, m, ifmap2E, ifmap2F }, dataType, 0);
                int ifmap2Bytes = ifmap2Tensor.AllocatedBytes * GNNEEnv.NPingPongSplit;
                glbUsageBytes[0] += ifmap2Bytes;
                ifmap2Bytes = TileUtilities.GetAlignedNum(ifmap2Bytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
                glbUsageBytes[1] += ifmap2Bytes;
                boxes.Add(new BoxOnGlb(
                    new int[2] { GNNEEnv.IfmapBankWidth, ifmap2Bytes / GNNEEnv.IfmapBankWidth / GNNEEnv.GlbBankWidth },
                    ItemName.Ifmap2));
            }
        }

        if (!_tp1)
        {
            glbMap.Add(ItemName.Ifmap2, ifmap2Tensor);
        }

        if ((object)_pool != null)
        {
            TensorOnGlb pdpActTensor = new TensorOnGlb(new int[4] { 1, 1, _outputShape[1], GNNEEnv.ActNumPerChan },
                _pool[GNNEPdp0Reduce.Act].CheckedDataType, 0);
            int actSize4 = SpaceSearcher.GetActSize(_outputShape[1], GNNEEnv.ActNumPerChan,
                TileUtilities.GetBytesPerElement(_pool[GNNEPdp0Reduce.Act].CheckedDataType));
            glbUsageBytes[0] += actSize4;
            actSize4 = TileUtilities.GetAlignedNum(actSize4, GNNEEnv.ActBankWidth * GNNEEnv.GlbBankWidth);
            glbUsageBytes[1] += actSize4;
            boxes.Add(new BoxOnGlb(
                new int[2] { GNNEEnv.ActBankWidth, actSize4 / GNNEEnv.ActBankWidth / GNNEEnv.GlbBankWidth },
                ItemName.PdpAct1));
            glbMap.Add(ItemName.PdpAct1, pdpActTensor);
        }

        glbUsage[0] = glbUsageBytes[0] / (float)GNNEEnv.GlbSize;
        glbUsage[1] = glbUsageBytes[1] / (float)GNNEEnv.GlbSize;
        return new AllocateResult { Boxes = boxes, GlbMap = glbMap };
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

            if (_l2RIf2Rec[i] != null)
            {
                _l2RIf2Rec[i].Clear();
            }
            else
            {
                _l2RIf2Rec[i] = new List<Tuple<SegmentND, TensorStat>>();
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

    private void InitParameters(Call convNode, Call ld, Call st)
    {
        _weightSplitPattern = new Tuple<int, int>(0, 0);
        _conv = convNode;
        _lif = ld;
        _lw = (Call)_conv[GNNEConv2D.Weights];
        _lact = (Call)_conv[GNNEConv2D.Act];
        _lwQarg = (Call)_conv[GNNEConv2D.WeightsBias];
        _sof = st;
        _inputShape = new GNNEShape(1, _lif.CheckedShape[1].FixedValue, 1, _lif.CheckedShape[0].FixedValue);
        _outputShape = new GNNEShape(1, _conv.CheckedShape[1].FixedValue, 1, _conv.CheckedShape[0].FixedValue);
        _convOutputShape = _outputShape;
        _weightsShape = new GNNEShape(_lw.CheckedShape[0].FixedValue, _lw.CheckedShape[1].FixedValue,
            _lw.CheckedShape[2].FixedValue, _lw.CheckedShape[3].FixedValue);
        int[] paddings = ((TensorConst)_conv[GNNEConv2D.Padding]).Value.ToArray<int>();
        _paddingH = new Padding(paddings[0], paddings[1]);
        _paddingW = new Padding(paddings[2], paddings[3]);
        _strideH = ((TensorConst)_conv[GNNEConv2D.Stride]).Value.ToArray<int>()[0];
        _strideW = ((TensorConst)_conv[GNNEConv2D.Stride]).Value.ToArray<int>()[1];
        _dilationH = ((TensorConst)_conv[GNNEConv2D.Dilation]).Value.ToArray<int>()[0];
        _dilationW = ((TensorConst)_conv[GNNEConv2D.Dilation]).Value.ToArray<int>()[1];
        _groups = ((TensorConst)_conv[GNNEConv2D.Groups]).Value.ToScalar<int>();
        _icPerGroup = _inputShape[1] / _groups;
        _ocPerGroup = _outputShape[1] / _groups;
        _groupPerPass = Math.Min(Math.Max(Math.Min(GNNEEnv.PuHeight / _icPerGroup, GNNEEnv.PuWidth / _ocPerGroup), 1),
            _groups);
        _pool = ((st[GNNEStore.Input] is Call storePool && storePool.Target is GNNEPdp0Reduce)
            ? ((Call)st[GNNEStore.Input])
            : null);
        _dw = ((st[GNNEStore.Input] is Call storeDw && storeDw.Target is GNNEPdp0DW) ? ((Call)st[GNNEStore.Input]) : null);
        _act1 = ((st[GNNEStore.Input] is Call storeAct1 && storeAct1.Target is GNNEActivation)
            ? ((Call)st[GNNEStore.Input])
            : null);
        if ((object)_act1 != null && _act1[GNNEActivation.InputB] != None.Default)
        {
            if (_act1[GNNEActivation.InputA] is Call inputACall && inputACall.Target is GNNELoad)
            {
                _lif2 = _act1[GNNEActivation.InputA] as Call;
            }
            else
            {
                _lif2 = _act1[GNNEActivation.InputB] as Call;
            }
        }
        else
        {
            _lif2 = null;
        }

        _strategy = DetermineSearchStrategy();
        _inputType = _lif.CheckedDataType;
        _weightType = _lw.CheckedDataType;
        _weightGroups.Add(new WeightGroupHandler(_weightType, _weightType));
        _weightGroups.Add(new WeightGroupHandler(_weightType, _weightType));
        _outputType = _conv.CheckedDataType;
        if ((object)_pool != null)
        {
            _outputType = _pool.CheckedDataType;
        }

        if ((object)_dw != null)
        {
            _outputType = _dw.CheckedDataType;
        }

        if ((object)_act1 != null)
        {
            _outputType = _act1.CheckedDataType;
        }

        _fusedKernelH = 1;
        _fusedKernelW = 1;
        _fusedPaddingH = new Padding(0, 0);
        _fusedPaddingW = new Padding(0, 0);
        _fusedStrideH = 1;
        _fusedStrideW = 1;
        _fusedDilationH = 1;
        _fusedDilationW = 1;
        if ((object)_pool != null)
        {
            _outputShape = new GNNEShape(_pool.CheckedShape[0].FixedValue, _pool.CheckedShape[1].FixedValue,
                _pool.CheckedShape[2].FixedValue, _pool.CheckedShape[3].FixedValue);
            _fusedKernelH = ((TensorConst)_pool[GNNEPdp0Reduce.Filter]).Value.ToArray<int>()[0];
            _fusedKernelW = ((TensorConst)_pool[GNNEPdp0Reduce.Filter]).Value.ToArray<int>()[1];
            paddings = ((TensorConst)_pool[GNNEPdp0Reduce.Padding]).Value.ToArray<int>();
            _fusedPaddingH = new Padding(paddings[0], paddings[1]);
            _fusedPaddingW = new Padding(paddings[2], paddings[3]);
            _fusedStrideH = ((TensorConst)_pool[GNNEPdp0Reduce.Stride]).Value.ToArray<int>()[0];
            _fusedStrideW = ((TensorConst)_pool[GNNEPdp0Reduce.Stride]).Value.ToArray<int>()[1];
        }
        else if ((object)_dw != null)
        {
            _outputShape = new GNNEShape(_dw.CheckedShape[0].FixedValue, _dw.CheckedShape[1].FixedValue,
                _dw.CheckedShape[2].FixedValue, _dw.CheckedShape[3].FixedValue);
            _fusedKernelH = _dw[GNNEPdp0DW.Weights].CheckedShape[2].FixedValue;
            _fusedKernelW = _dw[GNNEPdp0DW.Weights].CheckedShape[3].FixedValue;
            paddings = ((TensorConst)_dw[GNNEPdp0DW.Padding]).Value.ToArray<int>();
            _fusedPaddingH = new Padding(paddings[0], paddings[1]);
            _fusedPaddingW = new Padding(paddings[2], paddings[3]);
            _fusedStrideH = ((TensorConst)_dw[GNNEPdp0DW.Stride]).Value.ToArray<int>()[0];
            _fusedStrideW = ((TensorConst)_dw[GNNEPdp0DW.Stride]).Value.ToArray<int>()[1];
            _fusedDilationH = ((TensorConst)_dw[GNNEPdp0DW.Dilation]).Value.ToArray<int>()[0];
            _fusedDilationW = ((TensorConst)_dw[GNNEPdp0DW.Dilation]).Value.ToArray<int>()[1];
        }
        else if ((object)_act1 != null)
        {
            _outputShape = new GNNEShape(_act1.CheckedShape[0].FixedValue, _act1.CheckedShape[1].FixedValue,
                _act1.CheckedShape[2].FixedValue, _act1.CheckedShape[3].FixedValue);
            TileUtilities.Assert(_outputShape.Dims.SequenceEqual(_convOutputShape.Dims),
                "_outputShape.Dims.SequenceEqual(_convOutputShape.Dims)",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileTransposeMatmul.cs",
                2144);
        }

        _convOutputShape[2] = TileUtilities.GetInputRowSegment(0, _outputShape[2], _convOutputShape[2], _fusedKernelH,
            _fusedStrideH, _fusedDilationH, in _fusedPaddingH).Length;
        _convOutputShape[3] = TileUtilities.GetInputColumnSegment(0, _outputShape[3], _convOutputShape[3],
            _fusedKernelW, _fusedStrideW, _fusedDilationW, in _fusedPaddingW).Length;
        bool isDepthwise = _inputShape[1] == _outputShape[1] && _outputShape[1] == _groups && _groups != 1;
        _h2C = _weightsShape[1] * _weightsShape[2] <= GNNEEnv.PuHeight && _weightsShape[2] != 1 &&
               _inputShape[2] > 200 && _inputShape[3] > 200 && !isDepthwise;
        if (_lif.CheckedShape[0].FixedValue > 1 && _conv.CheckedShape[2].FixedValue == 1 &&
            _conv.CheckedShape[3].FixedValue == 1 && _lif.CheckedShape[2].FixedValue == 1 &&
            _lif.CheckedShape[3].FixedValue == 1 && _weightsShape[2] == 1 && _weightsShape[3] == 1 && _strideH == 1 &&
            _strideW == 1 && _dilationH == 1 && _dilationW == 1 && _paddingH.Sum() == 0 && _paddingW.Sum() == 0 &&
            _groups == 1)
        {
            _tp1 = true;
            _tp2 = true;
        }
    }

    private GlbSearchStrategy DetermineSearchStrategy()
    {
        if (_inputShape.Size > _weightsShape.Size)
        {
            return GlbSearchStrategy.IfFirst;
        }

        return GlbSearchStrategy.WFirst;
    }

    private ScheduleStrategy DetermineScheduleStrategy(List<GnneAction> actionsIfFirst, List<GnneAction> actionsWFirst)
    {
        DdrBandwidth[] bandwidth = new DdrBandwidth[2] { new DdrBandwidth(), new DdrBandwidth() };
        foreach (GnneAction action in actionsIfFirst)
        {
            bandwidth[0].CalcBw(action);
        }

        foreach (GnneAction action in actionsWFirst)
        {
            bandwidth[1].CalcBw(action);
        }

        long ifFirstBandwidth = bandwidth[0].TotalBw[0];
        long wFirstBandwidth = bandwidth[1].TotalBw[0];
        if (ifFirstBandwidth >= wFirstBandwidth)
        {
            return ScheduleStrategy.WFirstSchedule;
        }

        return ScheduleStrategy.IfFirstSchedule;
    }

    private void UpdateCcrRecStat()
    {
        List<Tuple<SegmentND, TensorStat>>[] recArrays = _g2LIfRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in recArrays)
        {
            for (int sliceIndex = 0; sliceIndex < recList.Count; sliceIndex++)
            {
                if (sliceIndex == 0)
                {
                    recList[sliceIndex].Item2.IsFirstSlice = true;
                }

                if (sliceIndex == recList.Count - 1)
                {
                    recList[sliceIndex].Item2.IsLastSlice = true;
                }

                if (sliceIndex < recList.Count - 1 && recList[sliceIndex].Item1 != recList[sliceIndex + 1].Item1)
                {
                    recList[sliceIndex].Item2.IsLastSlice = true;
                    recList[sliceIndex + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        recArrays = _g2RWRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in recArrays)
        {
            for (int sliceIndex = 0; sliceIndex < recList.Count; sliceIndex++)
            {
                if (sliceIndex == 0)
                {
                    recList[sliceIndex].Item2.IsFirstSlice = true;
                }

                if (sliceIndex == recList.Count - 1)
                {
                    recList[sliceIndex].Item2.IsLastSlice = true;
                }

                if (sliceIndex < recList.Count - 1 && recList[sliceIndex].Item1 != recList[sliceIndex + 1].Item1)
                {
                    recList[sliceIndex].Item2.IsLastSlice = true;
                    recList[sliceIndex + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        recArrays = _l2GOfRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in recArrays)
        {
            for (int sliceIndex = 0; sliceIndex < recList.Count; sliceIndex++)
            {
                if (sliceIndex == 0)
                {
                    recList[sliceIndex].Item2.IsFirstSlice = true;
                }

                if (sliceIndex == recList.Count - 1)
                {
                    recList[sliceIndex].Item2.IsLastSlice = true;
                }

                if (sliceIndex < recList.Count - 1 && !(recList[sliceIndex].Item1 == recList[sliceIndex + 1].Item1))
                {
                    recList[sliceIndex].Item2.IsLastSlice = true;
                    recList[sliceIndex + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        recArrays = _l2RIf2Rec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in recArrays)
        {
            for (int sliceIndex = 0; sliceIndex < recList.Count; sliceIndex++)
            {
                if (sliceIndex == 0)
                {
                    recList[sliceIndex].Item2.IsFirstSlice = true;
                }

                if (sliceIndex == recList.Count - 1)
                {
                    recList[sliceIndex].Item2.IsLastSlice = true;
                }

                if (sliceIndex < recList.Count - 1 && recList[sliceIndex].Item1 != recList[sliceIndex + 1].Item1)
                {
                    recList[sliceIndex].Item2.IsLastSlice = true;
                    recList[sliceIndex + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        recArrays = _weightRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in recArrays)
        {
            if (recList.Count > 0)
            {
                recList[0].Item2.IsFirstSlice = true;
                recList[recList.Count - 1].Item2.IsLastSlice = true;
            }
        }

        recArrays = _ofmapRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in recArrays)
        {
            if (recList.Count > 0)
            {
                recList[0].Item2.IsFirstSlice = true;
                recList[recList.Count - 1].Item2.IsLastSlice = true;
            }
        }

        for (int recIndex = 0; recIndex < _g2RWSliceRec.Length; recIndex++)
        {
            List<SegmentND> uniqueSlices = new List<SegmentND>();
            List<SegmentND> wSlices = new List<SegmentND>();
            List<Tuple<SegmentND, TensorStat>> sliceRecs = new List<Tuple<SegmentND, TensorStat>>();
            for (int recSliceIndex = 0; recSliceIndex < _g2RWRec[recIndex].Count; recSliceIndex++)
            {
                if (_g2RWRec[recIndex][recSliceIndex].Item2.IsFirstSlice)
                {
                    uniqueSlices.Clear();
                    wSlices.Clear();
                    sliceRecs.Clear();
                }

                if (!uniqueSlices.Contains(_g2RWSliceRec[recIndex][recSliceIndex].Item1))
                {
                    uniqueSlices.Add(_g2RWSliceRec[recIndex][recSliceIndex].Item1);
                }

                wSlices.Add(_g2RWSliceRec[recIndex][recSliceIndex].Item1);
                sliceRecs.Add(_g2RWSliceRec[recIndex][recSliceIndex]);
                if (!_g2RWRec[recIndex][recSliceIndex].Item2.IsLastSlice)
                {
                    continue;
                }

                foreach (SegmentND slice in uniqueSlices)
                {
                    sliceRecs[wSlices.FindIndex((SegmentND x) => x == slice)].Item2.IsFirstSlice = true;
                    sliceRecs[wSlices.FindLastIndex((SegmentND g2RwSliceIdx) => g2RwSliceIdx == slice)].Item2.IsLastSlice =
                        true;
                }

                int cnt;
                for (cnt = 0; cnt < wSlices.Count; cnt++)
                {
                    sliceRecs[cnt].Item2.SliceIdx = uniqueSlices.FindIndex((SegmentND x) => x == wSlices[cnt]);
                    _g2RWSliceRec[recIndex][recSliceIndex - wSlices.Count + 1 + cnt] = sliceRecs[cnt];
                }
            }

            if (!(_weightType == DataTypes.Int16))
            {
                continue;
            }

            for (int mirrorIndex = 1; mirrorIndex < _g2RWSliceRec[recIndex].Count; mirrorIndex++)
            {
                if (_g2RWSliceRec[recIndex][_g2RWSliceRec[recIndex].Count - mirrorIndex - 1].Item2.IsFirstSlice)
                {
                    _g2RWSliceRec[recIndex][_g2RWSliceRec[recIndex].Count - mirrorIndex].Item2.IsFirstSlice = true;
                }

                if (_g2RWSliceRec[recIndex][mirrorIndex].Item2.IsLastSlice)
                {
                    _g2RWSliceRec[recIndex][mirrorIndex - 1].Item2.IsLastSlice = true;
                }
            }

            for (int int16SliceIndex = 0; int16SliceIndex < _g2RWSliceRec[recIndex].Count; int16SliceIndex++)
            {
                _g2RWSliceRec[recIndex][int16SliceIndex].Item2.SliceIdx = _g2RWSliceRec[recIndex][int16SliceIndex].Item2.SliceIdx * 2 + (int16SliceIndex & 1);
            }
        }
    }

    private void ArrangeWeights(DataType weightsType, GNNEShape weightsShape, Span<byte> oldWeights,
        WeightGroupHandler weightGroup)
    {
        int bytesPerElement = TileUtilities.GetBytesPerElement(weightsType);
        List<SegmentND> weightSlices = weightGroup.WeightGroupSlice();
        byte[] arranged = new byte[oldWeights.Length];
        if (_h2C)
        {
            int expectedOffset = 0;
            int writeIndex = 0;
            foreach (SegmentND weightSlice in weightSlices)
            {
                TileUtilities.Assert(expectedOffset == weightGroup.WeightGroupOffset(weightSlice) * bytesPerElement,
                    "offset == weightGroup.WeightGroupOffset(slice) * bytesPerElement",
                    "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileTransposeMatmul.cs",
                    2399);
                for (int byteIndex = 0; byteIndex < bytesPerElement; byteIndex++)
                {
                    for (int outChannel = 0; outChannel < weightSlice[0].Length; outChannel++)
                    {
                        for (int kernelW = 0; kernelW < weightSlice[3].Length; kernelW++)
                        {
                            for (int kernelH = 0; kernelH < weightSlice[2].Length; kernelH++)
                            {
                                for (int inChannel = 0; inChannel < weightSlice[1].Length; inChannel++)
                                {
                                    int srcIndex = ((((outChannel + weightSlice[1].Start) * weightsShape[1] + inChannel + weightSlice[1].Start) *
                                                    weightsShape[2] + kernelH + weightSlice[2].Start) * weightsShape[3] + kernelW +
                                                weightSlice[3].Start) *
                                               bytesPerElement;
                                    arranged[writeIndex++] = oldWeights[srcIndex + byteIndex];
                                    expectedOffset++;
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            int expectedOffset = 0;
            int writeIndex = 0;
            foreach (SegmentND weightSlice in weightSlices)
            {
                TileUtilities.Assert(expectedOffset == weightGroup.WeightGroupOffset(weightSlice) * bytesPerElement,
                    "offset == weightGroup.WeightGroupOffset(slice) * bytesPerElement",
                    "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileTransposeMatmul.cs",
                    2426);
                for (int byteIndex = 0; byteIndex < bytesPerElement; byteIndex++)
                {
                    for (int outChannel = 0; outChannel < weightSlice[0].Length; outChannel++)
                    {
                        for (int kernelH = 0; kernelH < weightSlice[2].Length; kernelH++)
                        {
                            for (int kernelW = 0; kernelW < weightSlice[3].Length; kernelW++)
                            {
                                for (int inChannel = 0; inChannel < weightSlice[1].Length; inChannel++)
                                {
                                    int srcIndex =
                                        ((((outChannel + weightSlice[0].Start) * weightsShape[1] + inChannel + weightSlice[1].Start) *
                                             weightsShape[2] + kernelH + weightSlice[2].Start) * weightsShape[3] + kernelW +
                                         weightSlice[3].Start) * bytesPerElement;
                                    arranged[writeIndex++] = oldWeights[srcIndex + byteIndex];
                                    expectedOffset++;
                                }
                            }
                        }
                    }
                }
            }
        }

        arranged.CopyTo(oldWeights);
    }

    private void ArrangeDwWeights(DataType weightsType, GNNEShape weightsShape, Span<byte> oldWeights,
        WeightGroupHandler weightGroup)
    {
        int bytesPerElement = TileUtilities.GetBytesPerElement(weightsType);
        int kernelPosition = 0;
        byte[] arranged = new byte[oldWeights.Length];
        foreach (SegmentND dwSlice in weightGroup.DwGroupSlice())
        {
            int start = dwSlice[1].Start;
            int end = dwSlice[1].End;
            int[] dwShape = ((TensorConst)_dw[GNNEPdp0DW.Weights]).Value.ToArray<int>();
            SegmentND dwSegment = new SegmentND(..1, start..end, ..dwShape[0], ..dwShape[1]);
            for (int i = 0; i < dwSegment[0].Length; i++)
            {
                for (int j = 0; j < dwSegment[2].Length; j++)
                {
                    for (int k = 0; k < dwSegment[3].Length; k++)
                    {
                        for (int l = 0; l < dwSegment[1].Length; l++)
                        {
                            int srcIndex =
                                ((((i + dwSegment[0].Start) * weightsShape[1] + l + dwSegment[1].Start) *
                                     weightsShape[2] + j + dwSegment[2].Start) * weightsShape[3] + k +
                                 dwSegment[3].Start) * bytesPerElement;
                            for (int m = 0; m < bytesPerElement; m++)
                            {
                                arranged[kernelPosition * GNNEEnv.PuWidth + l] = oldWeights[srcIndex + m];
                            }
                        }

                        kernelPosition++;
                    }
                }
            }
        }

        arranged.CopyTo(oldWeights);
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Call call = (Call)__result["call"];
        Call ld = (Call)__result["ld"];
        Call st = (Call)__result["st"];
        Fusion fusion = (Fusion)__result["fusion"];
        return GetReplace(call, ld, st, fusion);
    }
}
