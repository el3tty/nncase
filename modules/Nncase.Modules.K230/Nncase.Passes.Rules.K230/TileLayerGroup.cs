// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.IR.Tensors;
using Nncase.Runtime.K230;
using Nncase.TIR;
using Nncase.TIR.Builders;
using Nncase.TIR.Instructions;

namespace Nncase.Passes.Rules.K230;

public class TileLayerGroup
{
    private enum L1FusedType
    {
        NoFused,
        FusedPool,
        FusedAct1,
        FusedDw
    }

    private static int _count = -1;

    private readonly ILogger<TileLayerGroup> _logger =
        CompileSessionScope.GetCurrentThrowIfNull().GetRequiredService<ILogger<TileLayerGroup>>();

    private readonly Dictionary<Call, WeightGroupHandler> _weightGroups = new Dictionary<Call, WeightGroupHandler>();

    private readonly Dictionary<Call, Tuple<int, int>> _weightSplitPattern = new Dictionary<Call, Tuple<int, int>>();

    private readonly List<Tuple<Call, int>> _nodesQueNeedClearFake = new List<Tuple<Call, int>>();

    private readonly Dictionary<Var, Nncase.TIR.Buffer> _ifBufferMap =
        new Dictionary<Var, Nncase.TIR.Buffer>(ReferenceEqualityComparer.Instance);

    private readonly Dictionary<Call, L1FusedType> _l1FusedInfos = new Dictionary<Call, L1FusedType>();

    private CcrHandler _ccrHandler = new CcrHandler();

    private GprHandler _gpr = new GprHandler();

    private SsrHandler _ssr = new SsrHandler();

    private NodeInfo _preNi;

    private NodeInfo _ni;

    private NodeInfo _l1FuseNi;

    private bool _l1Fused;

    private bool _swapAB;

    private bool _h2C;

    private int _memsetValue;

    private SegmentND _ifmap;

    private SegmentND _ifmap2;

    private int _ifmapOffset;

    private int _ifmap2Offset;

    private ItemName _src2ItemName;

    private SegmentND _ofmap;

    private int _ofmapOffset;

    private SegmentND _ofmapSt;

    private SegmentND _ifmapLd;

    private SegmentND _ofmapConv;

    private SegmentND _ifmapA;

    private SegmentND _ifmapB;

    private DataType _inputType;

    private DataType _outputType;

    private DataType _weightType;

    private DataType _if2Type;

    private Call _conv;

    private Call _pool;

    private Call _dw;

    private Call _act1;

    private Call _resize;

    private int _icPerGroup;

    private int _ocPerGroup;

    private int _groupPerPass;

    private Call? _lif;

    private Call? _lw;

    private Call? _lact;

    private Call? _lwQarg;

    private Call? _sof;

    private int[]? _inputShape;

    private int[]? _outputShape;

    private int[]? _convOutputShape;

    private int[]? _weightsShape;

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

    private Call? _lif2;

    private Call? _pdp1;

    private Call? _transpose;

    private Call? _cat;

    private WeightGroupHandler _weightGroup = new WeightGroupHandler(DataTypes.UInt8, DataTypes.UInt8);

    private SegmentND? _weight;

    private bool _isGlobalPdp;

    private Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>> _nodesWeightRec =
        new Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>>();

    private Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>> _nodesOfmapRec =
        new Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>>();

    private Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>> _nodesG2LIfRec =
        new Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>>();

    private Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>> _nodesG2RWRec =
        new Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>>();

    private Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>> _nodesL2GOfRec =
        new Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>>();

    private Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>> _nodesL2RIf2Rec =
        new Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>>();

    private Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>> _nodesG2RWSliceRec =
        new Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>>();

    private Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>> _nodesAi2dIfRec =
        new Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>>();

    private Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>> _nodesAi2dOfRec =
        new Dictionary<Call, List<List<Tuple<SegmentND, TensorStat>>>>();

    private List<List<Tuple<Call, int>>> _nodesQuenesAsW = new List<List<Tuple<Call, int>>>(2)
    {
        new List<Tuple<Call, int>>(), new List<Tuple<Call, int>>()
    };

    private int _if1BufIdx = -1;

    private int _if2BufIdx = -1;

    private int _ofBufIdx = -1;

    private int _weightBufIdx = -1;

    public PrimFunction BuildSchedule(FusionInfo fusionInfo)
    {
        _count++;
        TiledGlb glb = new TiledGlb();
        InitParameters(fusionInfo);
        GetSliceInfo(fusionInfo, glb, out List<List<NodeInfo>> currSliceInfo,
            out List<Dictionary<Call, NodeInfo>> preSliceInfo);
        List<Sequential> instructions = new List<Sequential>();
        List<Nncase.TIR.Buffer> ifBuffers = new List<Nncase.TIR.Buffer>();
        List<Nncase.TIR.Buffer> ofBuffers = new List<Nncase.TIR.Buffer>();
        _gpr = new GprHandler(GNNEEnv.GprNum);
        _ssr = new SsrHandler(GNNEEnv.SsrNum);
        _ccrHandler = new CcrHandler();
        instructions.Add(BuildMmu(glb));
        ItemRecStatusInit(currSliceInfo);
        for (int sliceIdx = 0; sliceIdx < currSliceInfo.Count; sliceIdx++)
        {
            List<NodeInfo> sliceNodes = currSliceInfo[sliceIdx];
            Dictionary<Call, NodeInfo> sliceInfo = preSliceInfo[sliceIdx];
            for (int layerIdx = 0; layerIdx < sliceNodes.Count; layerIdx += ((!_l1Fused) ? 1 : 2))
            {
                UpdateL2FusePara(fusionInfo, sliceNodes[layerIdx], sliceInfo, glb, weightGroupOnly: true, layerIdx == 0);
                ItemRecStatusUpdate();
                if ((object)_conv != null)
                {
                    BuildConv2d(glb, weightGroupOnly: true, ifBuffers);
                }

                if ((object)_resize != null)
                {
                    BuildResize(glb, sliceNodes[sliceIdx], weightGroupOnly: true);
                }
            }
        }

        UpdateCcrRecStat();
        UpdateAi2dCcrRecStat();
        List<NodeInfo> fusedNodes = fusionInfo.FusedNodes;
        _logger.LogTrace(
            $"{fusedNodes[fusedNodes.Count - 1].Op} -> slice: {currSliceInfo.Count}, layer: {currSliceInfo[0].Count}");
        for (int sliceIdx = 0; sliceIdx < currSliceInfo.Count; sliceIdx++)
        {
            List<NodeInfo> sliceNodes = currSliceInfo[sliceIdx];
            Dictionary<Call, NodeInfo> preSlice = preSliceInfo[sliceIdx];
            bool isFirstSlice = sliceIdx == 0;
            List<Nncase.TIR.Buffer> ifBuffersCopy = new List<Nncase.TIR.Buffer>(ifBuffers);
            List<Nncase.TIR.Buffer> ofBuffersCopy = new List<Nncase.TIR.Buffer>(ofBuffers);
            SegmentND lastOfmap = sliceNodes[sliceNodes.Count - 1].Ofmap;
            _logger.LogTrace(
                $"dim2-> start: {lastOfmap[2].Start}, end: {lastOfmap[2].End}, dim3-> start: {lastOfmap[3].Start}, end: {lastOfmap[3].End}");
            for (int layerIdx = 0; layerIdx < sliceNodes.Count; layerIdx += ((!_l1Fused) ? 1 : 2))
            {
                UpdateL2FusePara(fusionInfo, sliceNodes[layerIdx], preSlice, glb, weightGroupOnly: false, layerIdx == 0);
                if ((object)_lif != null)
                {
                    int iPp = 0;
                    instructions.Add(BuildLoadIf(glb, iPp, ifBuffers, isFirstSlice, ifBuffersCopy));
                }

                if ((object)_conv != null)
                {
                    instructions.Add(BuildConv2d(glb, weightGroupOnly: false, ifBuffers, isFirstSlice, ifBuffersCopy));
                }

                if ((object)_act1 != null && !_l1Fused)
                {
                    instructions.Add(BuildAct1(glb, ifBuffers, isFirstSlice, isFirstSlice, ifBuffersCopy));
                }

                if ((object)_pdp1 != null)
                {
                    instructions.Add(BuildPdp1(glb));
                }

                if ((object)_transpose != null)
                {
                    instructions.Add(BuildTranspose(glb));
                }

                _ = _cat;
                if ((object)_resize != null)
                {
                    BuildResize(glb, sliceNodes[sliceIdx], weightGroupOnly: false);
                }

                if ((object)_sof != null)
                {
                    int ofPp = 0;
                    instructions.Add(BuildStore(glb, ofPp, ofBuffers, preSliceInfo[sliceIdx], isFirstSlice, ofBuffersCopy, fusionInfo.Fusion));
                }
            }

            TileUtilities.Assert(ifBuffersCopy.Count == 0 && ofBuffersCopy.Count == 0,
                "ifBuffersCopy.Count == 0 && ofBuffersCopy.Count == 0",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                260);
        }

        TileUtilities.Assert(_ccrHandler.CcrSanityCheck(), "_ccrHandler.CcrSanityCheck()",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
            263);
        List<Nncase.TIR.Buffer> ddrBuffers = fusionInfo.Inputs.Select((Var v) => _ifBufferMap[v]).ToList();
        ddrBuffers.AddRange(ofBuffers);
        ISequentialBuilder<PrimFunction> sequentialBuilder =
            T.PrimFunc($"TileLayerGroup_{_count}", K230RtModule.Kind, ddrBuffers.ToArray());
        object[] exprOrBuilders = instructions.ToArray();
        return sequentialBuilder.Body(exprOrBuilders).Body(I.END(GP_REGISTER.x0)).Build();
    }

    private Sequential BuildMmu(TiledGlb glb)
    {
        GnneActionUpdater gnneActionUpdater =
            new GnneActionUpdater(new List<GnneAction>(), glb, _ccrHandler, _gpr, _ssr);
        gnneActionUpdater.UpdateMmuConf();
        return new ActionToInstruct().Instructions(gnneActionUpdater.Actions);
    }

    private Sequential BuildLoadIf(TiledGlb glb, int iPp, List<Nncase.TIR.Buffer> ifBuffers, bool isFirstSlice,
        List<Nncase.TIR.Buffer> ifBuffersCopy)
    {
        GnneActionUpdater gnneActionUpdater =
            new GnneActionUpdater(new List<GnneAction>(), glb, _ccrHandler, _gpr, _ssr);
        List<CcrSet> ccrSetsTmp;
        List<CcrClr> ccrClrsTmp;
        if (_lif[GNNELoad.Input] is Call concatCall && concatCall.Target is Concat)
        {
            Expr[] concatInputs = ((Nncase.IR.Tuple)concatCall.Arguments[Concat.Input.Index]).Fields.ToArray();
            List<Nncase.TIR.Buffer> concatBuffers = new List<Nncase.TIR.Buffer>(concatInputs.Length);
            int channelOffset = 0;
            for (int i = 0; i < concatInputs.Length; i++)
            {
                Expr expr = concatInputs[i];
                if (isFirstSlice)
                {
                    T.CreateBuffer(new TensorType(expr.CheckedDataType, expr.CheckedShape), MemoryLocation.Input,
                        out Nncase.TIR.Buffer ddrBuffer, "ddrIf_" + i);
                    concatBuffers.Add(ddrBuffer);
                    _ifBufferMap.Add((Var)expr, ddrBuffer);
                    ifBuffers.Add(ddrBuffer);
                }
                else
                {
                    concatBuffers.Add(ifBuffersCopy[0]);
                    ifBuffersCopy.RemoveAt(0);
                }

                int[] inputShape = expr.CheckedShape.ToValueArray();
                GetCcrSetAndClrVec(_ni).Deconstruct<List<CcrSet>, List<CcrClr>>(out ccrSetsTmp, out ccrClrsTmp);
                List<CcrSet> ccrsToSet = ccrSetsTmp;
                List<CcrClr> ccrsToClr = ccrClrsTmp;
                List<int> stridesD = new int[3]
                {
                    glb.GlbMap[ItemName.Ofmap].Dimensions[1], glb.GlbMap[ItemName.Ofmap].Dimensions[2],
                    glb.GlbMap[ItemName.Ofmap].Dimensions[3]
                }.ToList();
                List<int> stridesS = new int[3] { inputShape[1], inputShape[2], inputShape[3] }.ToList();
                SegmentND fullChannelSegment = new SegmentND(_ofmapSt);
                fullChannelSegment[1] = new Segment1D(0..inputShape[1], new Padding(0, 0));
                SegmentND slice = new SegmentND(fullChannelSegment);
                slice[1] = new Segment1D(channelOffset..(channelOffset + inputShape[1]), new Padding(0, 0));
                gnneActionUpdater.UpdateLoadIf(fullChannelSegment, _lif, iPp, concatBuffers[i],
                    _ni.Nb.OfmapOffset + TileUtilities.GetSliceOffsetInTensor(in _ofmapSt, in slice) *
                    TileUtilities.GetBytesPerElement(_lif.CheckedDataType), stridesD, ItemName.Ifmap,
                    (i == concatInputs.Length - 1) ? ccrsToSet : null, (i == 0) ? ccrsToClr : null, _h2C, _memsetValue, stridesS,
                    new GNNEShape(inputShape), inputShape);
                channelOffset += inputShape[1];
            }
        }
        else
        {
            Nncase.TIR.Buffer ddrBuffer;
            if (isFirstSlice)
            {
                Call loadSource = _lif;
                if (_lif[GNNELoad.Input] is Call reshapeCall && reshapeCall.Target is Reshape)
                {
                    loadSource = reshapeCall;
                }

                T.CreateBuffer(new TensorType(_lif[GNNELoad.Input].CheckedDataType, loadSource.CheckedShape),
                    MemoryLocation.Input, out ddrBuffer, "ddrIf");
                ifBuffers.Add(ddrBuffer);
                _ifBufferMap.Add(
                    ((object)loadSource != null && loadSource.Target is GNNELoad)
                        ? ((Var)loadSource[GNNELoad.Input])
                        : ((Var)loadSource[Reshape.Input]), ddrBuffer);
            }
            else
            {
                ddrBuffer = ifBuffersCopy[0];
                ifBuffersCopy.RemoveAt(0);
            }

            GetCcrSetAndClrVec(_ni).Deconstruct<List<CcrSet>, List<CcrClr>>(out ccrSetsTmp, out ccrClrsTmp);
            List<CcrSet> ccrsToSet = ccrSetsTmp;
            List<CcrClr> ccrsToClr = ccrClrsTmp;
            List<int> stridesD2 = new int[3]
            {
                glb.GlbMap[ItemName.Ofmap].Dimensions[1], glb.GlbMap[ItemName.Ofmap].Dimensions[2],
                glb.GlbMap[ItemName.Ofmap].Dimensions[3]
            }.ToList();
            gnneActionUpdater.UpdateLoadIf(_ofmapSt, _lif, iPp, ddrBuffer, _ni.Nb.OfmapOffset, stridesD2, ItemName.Ifmap,
                ccrsToSet, ccrsToClr, _h2C, _memsetValue, null, null, _lif.CheckedShape.ToValueArray());
        }

        return new ActionToInstruct().Instructions(gnneActionUpdater.Actions);
    }

    private Sequential BuildStore(TiledGlb glb, int ofPp, List<Nncase.TIR.Buffer> ofBuffers,
        Dictionary<Call, NodeInfo> preSliceInfo, bool isFirstSlice, List<Nncase.TIR.Buffer> ofBuffersCopy,
        Fusion fusion)
    {
        GnneActionUpdater gnneActionUpdater =
            new GnneActionUpdater(new List<GnneAction>(), glb, _ccrHandler, _gpr, _ssr);
        Nncase.TIR.Buffer buffer;
        if (isFirstSlice)
        {
            T.CreateBuffer(new TensorType(_sof.CheckedDataType, fusion.Body.CheckedShape), MemoryLocation.Output,
                out buffer, "ddrOf");
            ofBuffers.Add(buffer);
        }
        else
        {
            buffer = ofBuffersCopy[0];
            ofBuffersCopy.RemoveAt(0);
        }

        var (ccrsToSet, ccrsToClr) = GetCcrSetAndClrVec(_ni);
        gnneActionUpdater.UpdateStoreT(_ofmapSt, _sof, ofPp, buffer,
            preSliceInfo[_ni.Op[GNNEStore.Input] as Call].Nb.OfmapOffset, null, ccrsToSet, ccrsToClr, ItemName.Ifmap);
        return new ActionToInstruct().Instructions(gnneActionUpdater.Actions);
    }

    private Sequential BuildConv2d(TiledGlb glb, bool weightGroupOnly, List<Nncase.TIR.Buffer> ifBuffers,
        bool firstSlice = false, List<Nncase.TIR.Buffer>? ifBuffersCopy = null)
    {
        GnneActionUpdater gnneActionUpdater =
            new GnneActionUpdater(new List<GnneAction>(), glb, _ccrHandler, _gpr, _ssr);
        byte[] weightBytes = ((TensorConst)((Call)_ni.Op[GNNEConv2D.Weights])[GNNELoadW.Input]).Value.BytesBuffer.ToArray();
        byte[] weightBytesCopy = new byte[weightBytes.Length];
        Array.Copy(weightBytes, weightBytesCopy, weightBytesCopy.Length);
        T.AttachBuffer(
            Const.FromTensor(Tensor.FromBytes(DataTypes.UInt8, weightBytesCopy.ToArray(), new int[1] { weightBytesCopy.Length })),
            out Nncase.TIR.Buffer ddrW, "var ddrW");
        T.AttachBuffer((TensorConst)((Call)_ni.Op[GNNEConv2D.WeightsBias])[GNNELoadW.Input],
            out Nncase.TIR.Buffer ddrWQarg, "var ddrWQarg");
        T.AttachBuffer((TensorConst)((Call)_ni.Op[GNNEConv2D.Act])[GNNELoadW.Input], out Nncase.TIR.Buffer ddrAct,
            "var ddrAct");
        Nncase.TIR.Buffer ddrDw = null;
        Nncase.TIR.Buffer ddrDwQarg = null;
        Nncase.TIR.Buffer ddrDwAct = null;
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
                out ddrDw, "ddrDW");
            T.AttachBuffer((TensorConst)((Call)_dw[GNNEPdp0DW.WeightsBias])[GNNELoadW.Input], out ddrDwQarg, "ddrDWQarg");
            T.AttachBuffer((TensorConst)((Call)_dw[GNNEPdp0DW.Act])[GNNELoadW.Input], out ddrDwAct, "ddrDWAct");
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
                    if (!weightGroupOnly && firstSlice)
                    {
                        T.CreateBuffer(new TensorType(_lif2[GNNELoad.Input].CheckedDataType, _lif2.CheckedShape),
                            MemoryLocation.Input, out ddrIf2, "ddrIf2");
                        ifBuffers.Add(ddrIf2);
                        _ifBufferMap.Add((Var)_lif2[GNNELoad.Input], ddrIf2);
                    }
                    else if (!weightGroupOnly && !firstSlice)
                    {
                        ddrIf2 = ifBuffersCopy[0];
                        ifBuffersCopy.RemoveAt(0);
                    }
                }
                else
                {
                    T.AttachBuffer((TensorConst)_lif2[GNNELoad.Input], out ddrIf2, "ddrIf2");
                }
            }
        }

        BuildScheduleConv(gnneActionUpdater, glb, ddrW, ddrWQarg, ddrAct, ddrDw, ddrDwQarg, ddrDwAct, ddrAct1, ddrPdpAct,
            ddrIf2, weightGroupOnly, firstSlice);
        if (weightGroupOnly)
        {
            return null;
        }

        Span<byte> bytesBuffer = ddrW.Const().Value.BytesBuffer;
        ArrangeWeights(_weightType, _weightsShape, bytesBuffer, _weightGroup);
        if ((object)_dw != null)
        {
            ArrangeDwWeights(_dw[GNNEPdp0DW.Weights].CheckedDataType,
                _dw[GNNEPdp0DW.Weights].CheckedShape.ToValueArray(), ddrDw.Const().Value.BytesBuffer, _weightGroup);
        }

        return new ActionToInstruct().Instructions(gnneActionUpdater.Actions);
    }

    private Sequential BuildAct1(TiledGlb glb, List<Nncase.TIR.Buffer> ifBuffers, bool firstSlice, bool isFirstSlice,
        List<Nncase.TIR.Buffer> ifBuffersCopy)
    {
        GnneActionUpdater gnneActionUpdater =
            new GnneActionUpdater(new List<GnneAction>(), glb, _ccrHandler, _gpr, _ssr);
        Nncase.TIR.Buffer ddrIf2 = null;
        T.AttachBuffer((TensorConst)((Call)_act1[GNNEActivation.Act])[GNNELoadW.Input], out Nncase.TIR.Buffer ddrAct1,
            "var ddrAct1");
        if ((object)_lif2 != null)
        {
            if (!(_lif2[GNNELoad.Input] is TensorConst))
            {
                if (isFirstSlice)
                {
                    T.CreateBuffer(new TensorType(_lif2[GNNELoad.Input].CheckedDataType, _lif2.CheckedShape),
                        MemoryLocation.Input, out ddrIf2, "ddrIf2");
                    ifBuffers.Add(ddrIf2);
                    _ifBufferMap.Add((Var)_lif2[GNNELoad.Input], ddrIf2);
                }
                else
                {
                    ddrIf2 = ifBuffersCopy[0];
                    ifBuffersCopy.RemoveAt(0);
                }
            }
            else
            {
                T.AttachBuffer((TensorConst)_lif2[GNNELoad.Input], out ddrIf2, "ddrIf2");
            }
        }

        BuildScheduleAct1(gnneActionUpdater, glb, ddrAct1, ddrIf2, firstSlice);
        return new ActionToInstruct().Instructions(gnneActionUpdater.Actions);
    }

    private Sequential BuildPdp1(TiledGlb glb)
    {
        GnneActionUpdater gnneActionUpdater =
            new GnneActionUpdater(new List<GnneAction>(), glb, _ccrHandler, _gpr, _ssr);
        BuildSchedulePdp1(gnneActionUpdater, glb);
        return new ActionToInstruct().Instructions(gnneActionUpdater.Actions);
    }

    private Sequential BuildTranspose(TiledGlb glb)
    {
        GnneActionUpdater gnneActionUpdater =
            new GnneActionUpdater(new List<GnneAction>(), glb, _ccrHandler, _gpr, _ssr);
        BuildScheduleTranspose(gnneActionUpdater, glb);
        return new ActionToInstruct().Instructions(gnneActionUpdater.Actions);
    }

    private Sequential BuildResize(TiledGlb glb, NodeInfo currNode, bool weightGroupOnly, bool firstSlice = false)
    {
        GnneActionUpdater gnneActionUpdater =
            new GnneActionUpdater(new List<GnneAction>(), glb, _ccrHandler, _gpr, _ssr);
        BuildScheduleResize(gnneActionUpdater, glb, weightGroupOnly, currNode);
        if (!weightGroupOnly)
        {
            return new ActionToInstruct().Instructions(gnneActionUpdater.Actions);
        }

        return null;
    }

    private void BuildScheduleConv(GnneActionUpdater actionUpdater, TiledGlb glb, Nncase.TIR.Buffer ddrW,
        Nncase.TIR.Buffer ddrWQarg, Nncase.TIR.Buffer ddrAct, Nncase.TIR.Buffer ddrDw, Nncase.TIR.Buffer ddrDWQarg,
        Nncase.TIR.Buffer ddrDWAct, Nncase.TIR.Buffer ddrAct1, Nncase.TIR.Buffer ddrPdpAct, Nncase.TIR.Buffer ddrIf2,
        bool weightGroupOnly, bool firstSlice = false)
    {
        int iPp = 0;
        int ofPp = 0;
        int wPp = 0;
        if (!weightGroupOnly && firstSlice)
        {
            if (_weightType == DataTypes.UInt8 || _weightType == DataTypes.Int16)
            {
                List<CcrSet> wQargSets = new List<CcrSet>();
                List<CcrClr> wQargClrs = null;
                wQargSets.Add(new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.WQarg)), 1));
                actionUpdater.UpdateLoadWQarg(_lwQarg, _weightGroup, ddrWQarg, _ni.Nb.WeightQargOffset, wQargSets,
                    wQargClrs);
            }

            List<CcrSet> actSets = new List<CcrSet>();
            List<CcrClr> actClrs = new List<CcrClr>();
            actSets.Add(new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Act)), 1));
            actionUpdater.UpdateLoadAct(_lact, ddrAct, ItemName.Act, _ni.Nb.ActOffset, actSets, actClrs);
            if ((object)_dw != null)
            {
                List<CcrSet> dwSets = new List<CcrSet>();
                List<CcrClr> dwClrs = null;
                dwSets.Add(new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.DwWeight)), 1));
                actionUpdater.UpdateLoadDw(_dw, _weightGroup, ddrDw, _l1FuseNi.Nb.DwWeightOffset, dwSets, dwClrs);
                if (_dw[GNNEPdp0DW.Weights].CheckedDataType == DataTypes.UInt8)
                {
                    dwSets = new List<CcrSet>();
                    dwClrs = null;
                    dwSets.Add(new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.DwQarg)), 1));
                    actionUpdater.UpdateLoadDwQarg(_dw, _weightGroup, ddrDWQarg, _l1FuseNi.Nb.DwWeightQargOffset, dwSets,
                        dwClrs);
                }
            }

            if ((object)_act1 != null)
            {
                List<CcrSet> act1Sets = new List<CcrSet>();
                List<CcrClr> act1Clrs = null;
                act1Sets.Add(new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.MfuAct1)), 1));
                actionUpdater.UpdateLoadAct(_act1[GNNEActivation.Act] as Call, ddrAct1, ItemName.MfuAct1,
                    _l1FuseNi.Nb.Act1Offset, act1Sets, act1Clrs);
            }

            if ((object)_dw != null)
            {
                List<CcrSet> dwAct1Sets = new List<CcrSet>();
                List<CcrClr> dwAct1Clrs = null;
                dwAct1Sets.Add(new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.DwAct1)), 1));
                actionUpdater.UpdateLoadAct(_dw[GNNEPdp0DW.Act] as Call, ddrDWAct, ItemName.DwAct1,
                    _l1FuseNi.Nb.DwActOffset, dwAct1Sets, dwAct1Clrs);
            }
            else if ((object)_pool != null)
            {
                List<CcrSet> pdpActSets = new List<CcrSet>();
                List<CcrClr> pdpActClrs = null;
                pdpActSets.Add(new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.PdpAct1)), 1));
                actionUpdater.UpdateLoadAct(_pool[GNNEPdp0Reduce.Act] as Call, ddrPdpAct, ItemName.PdpAct1,
                    _l1FuseNi.Nb.Pdp0ActOffset, pdpActSets, pdpActClrs);
            }
        }

        _weightGroup.Current_aligned_offset_init();
        if (weightGroupOnly && _weightBufIdx != -1)
        {
            _nodesWeightRec[_conv][_weightBufIdx]
                .Add(new Tuple<SegmentND, TensorStat>(_weight,
                    new TensorStat(isFirstSlice: false, isLastSlice: false)));
        }

        if (!weightGroupOnly && (object)_act1 != null && (object)_lif2 != null && _ifmap[1].End == _inputShape[1])
        {
            TileUtilities.Assert(_if2BufIdx == -1, "_if2BufIdx == -1",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                595);
            TensorStat if2Stat = _nodesL2RIf2Rec[_conv][0][0].Item2;
            int ccrValue = ((if2Stat != null && if2Stat.IsFirstSlice && if2Stat.IsLastSlice) ? 1 : 2);
            List<CcrSet> ccrsToSet = new List<CcrSet>
            {
                new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap2)), ccrValue)
            };
            List<int> stridesD = new List<int>
            {
                glb.GlbMap[ItemName.Ifmap2].Dimensions[1],
                glb.GlbMap[ItemName.Ifmap2].Dimensions[2],
                glb.GlbMap[ItemName.Ifmap2].Dimensions[3]
            };
            actionUpdater.UpdateLoadIf(_ifmap2, _lif2, ofPp, ddrIf2, _ifmap2Offset, stridesD, _src2ItemName, ccrsToSet);
        }

        BuildL1Schedule(actionUpdater, glb, _ifmap, _weight, _ofmap, iPp, ofPp, wPp, _ifmap2, weightGroupOnly,
            _weightGroup, ddrW);
        if (weightGroupOnly)
        {
            _nodesOfmapRec[_conv][_ofBufIdx]
                .Add(new Tuple<SegmentND, TensorStat>(_ofmap, new TensorStat(isFirstSlice: false, isLastSlice: false)));
        }
        else
        {
            _nodesOfmapRec[_conv][_ofBufIdx].RemoveAt(0);
        }
    }

    private void BuildScheduleResize(GnneActionUpdater actionUpdater, TiledGlb glb, bool weightGroupOnly,
        NodeInfo currNode)
    {
    }

    private void BuildScheduleAct1(GnneActionUpdater actionUpdater, TiledGlb glb, Nncase.TIR.Buffer ddrAct,
        Nncase.TIR.Buffer ddrIf2, bool firstSlice = false)
    {
        int iPp = 0;
        List<CcrSet> ccrSets = new List<CcrSet>();
        List<CcrClr> ccrsToClr = null;
        if (firstSlice)
        {
            ccrSets.Add(new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.MfuAct1)), 1));
            actionUpdater.UpdateLoadAct(_lact, ddrAct, ItemName.MfuAct1, _ni.Nb.Act1Offset, ccrSets, ccrsToClr);
        }

        if ((object)_lif2 != null)
        {
            List<CcrSet> ccrsToSet = new List<CcrSet>
            {
                new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap2)), 1)
            };
            List<int> stridesD = new List<int>
            {
                glb.GlbMap[ItemName.Ifmap2].Dimensions[1],
                glb.GlbMap[ItemName.Ifmap2].Dimensions[2],
                glb.GlbMap[ItemName.Ifmap2].Dimensions[3]
            };
            actionUpdater.UpdateLoadIf(_ifmap2, _lif2, iPp, ddrIf2, _ifmap2Offset, stridesD, ItemName.Ifmap2,
                ccrsToSet);
        }

        bool isSingleInput = _act1[GNNEActivation.InputB] == None.Default;
        DataType inputAType = _act1[GNNEActivation.InputA].CheckedDataType;
        DataType inputBType = inputAType;
        if (!isSingleInput)
        {
            inputBType = _act1[GNNEActivation.InputB].CheckedDataType;
        }

        DeQuantizeParam deqAParams = new DeQuantizeParam(0, 1f);
        if (inputAType != DataTypes.Float16)
        {
            deqAParams = ((TensorConst)_act1[GNNEActivation.DeqAParams]).Value.ToScalar<DeQuantizeParam>();
        }

        DeQuantizeParam deqBParams = deqAParams;
        if (!isSingleInput && inputBType != DataTypes.Float16)
        {
            deqBParams = ((TensorConst)_act1[GNNEActivation.DeqBParams]).Value.ToScalar<DeQuantizeParam>();
        }

        int inAShiftBits = ((TensorConst)_act1[GNNEActivation.InAShiftBits]).Value.ToScalar<int>();
        int inBShiftBits = inAShiftBits;
        if (!isSingleInput)
        {
            inBShiftBits = ((TensorConst)_act1[GNNEActivation.InBShiftBits]).Value.ToScalar<int>();
        }

        int rshiftBitsD = ((TensorConst)_act1[GNNEActivation.OutShiftBits]).Value.ToScalar<int>();
        bool is16Segments = ((TensorConst)_act1[GNNEActivation.Is16Segments]).Value.ToScalar<bool>();
        if (_act1[GNNEActivation.InputA] is Call inputACall && inputACall.Target is GNNELoad && !isSingleInput && (object)_lif2 != null &&
            _swapAB)
        {
            inputAType = _act1[GNNEActivation.InputB].CheckedDataType;
            inputBType = _act1[GNNEActivation.InputA].CheckedDataType;
            deqAParams = ((TensorConst)_act1[GNNEActivation.DeqBParams]).Value.ToScalar<DeQuantizeParam>();
            deqBParams = ((TensorConst)_act1[GNNEActivation.DeqAParams]).Value.ToScalar<DeQuantizeParam>();
            inAShiftBits = ((TensorConst)_act1[GNNEActivation.InBShiftBits]).Value.ToScalar<int>();
            inBShiftBits = ((TensorConst)_act1[GNNEActivation.InAShiftBits]).Value.ToScalar<int>();
        }

        (ccrSets, ccrsToClr) = GetCcrSetAndClrVec(_ni);
        if ((object)_lif2 != null)
        {
            ccrsToClr.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap2))));
        }

        if (firstSlice)
        {
            ccrsToClr.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.MfuAct1))));
        }

        List<int> src2Stride = null;
        SegmentND src1Segment = _ifmap;
        SegmentND src2Segment = _ifmap2;
        if ((object)_lif2 == null && _ifmap2.Shape_size != 0)
        {
            src2Stride = new List<int>
            {
                glb.GlbMap[ItemName.Ifmap2].Dimensions[1],
                glb.GlbMap[ItemName.Ifmap2].Dimensions[2],
                glb.GlbMap[ItemName.Ifmap2].Dimensions[3]
            };
            src1Segment = _ifmapA;
            src2Segment = _ifmapB;
        }

        if (_ifmap2.Shape_size == 0 && src1Segment.Shape_size != 0 && (_ofmap[0].Length % src1Segment[0].Length != 0 ||
                                                                     _ofmap[1].Length % src1Segment[1].Length != 0 ||
                                                                     _ofmap[2].Length % src1Segment[2].Length != 0 ||
                                                                     _ofmap[3].Length % src1Segment[3].Length != 0))
        {
            src1Segment = _ofmap;
        }

        if (_ifmap2.Shape_size != 0 && _ifmap.Shape_size > _ifmap2.Shape_size &&
            _ifmap.Shape_size % _ifmap2.Shape_size != 0)
        {
            TileUtilities.Assert(
                _ifmap[0].Start <= _ifmap2[0].Start && _ifmap[1].Start <= _ifmap2[1].Start &&
                _ifmap[2].Start <= _ifmap2[2].Start && _ifmap[3].Start <= _ifmap2[3].Start,
                "_ifmap[0].Start <= _ifmap2[0].Start\r\n                && _ifmap[1].Start <= _ifmap2[1].Start\r\n                && _ifmap[2].Start <= _ifmap2[2].Start\r\n                && _ifmap[3].Start <= _ifmap2[3].Start",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                858);
            _ifmapOffset += TileUtilities.GetSliceOffsetInTensor(in _ifmap, in _ifmap2) *
                            TileUtilities.GetBytesPerElement(inputAType);
        }
        else if (_ifmap2.Shape_size != 0 && _ifmap2.Shape_size > _ifmap.Shape_size &&
                 _ifmap2.Shape_size % _ifmap.Shape_size != 0)
        {
            TileUtilities.Assert(
                _ifmap[0].Start >= _ifmap2[0].Start && _ifmap[1].Start >= _ifmap2[1].Start &&
                _ifmap[2].Start >= _ifmap2[2].Start && _ifmap[3].Start >= _ifmap2[3].Start,
                "_ifmap[0].Start >= _ifmap2[0].Start\r\n                && _ifmap[1].Start >= _ifmap2[1].Start\r\n                && _ifmap[2].Start >= _ifmap2[2].Start\r\n                && _ifmap[3].Start >= _ifmap2[3].Start",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                866);
            _ifmap2Offset += TileUtilities.GetSliceOffsetInTensor(in _ifmap2, in _ifmap) *
                             TileUtilities.GetBytesPerElement(inputBType);
        }

        actionUpdater.UpdateMfuAct1(src1Segment, src2Segment, _ofmap, ACT1_SOURCE_TYPE.l2, ACT1_SOURCE_TYPE.l2, inputAType,
            inputBType, _act1.CheckedDataType, deqAParams, deqBParams, inAShiftBits, inBShiftBits, rshiftBitsD, is16Segments,
            iPp, ccrSets, ccrsToClr, _ifmapOffset, _ifmap2Offset, _ni.Nb.OfmapOffset, _ni.Nb.Act1Offset, _src2ItemName,
            ItemName.Ifmap, ItemName.MfuAct1,
            (((GNNEActivation)_act1.Target).Type == GnneActivationType.Mul)
                ? MFU_ACT1_FUNCTION.mul
                : MFU_ACT1_FUNCTION.add, ItemName.Ofmap, null, src2Stride);
    }

    private void BuildSchedulePdp1(GnneActionUpdater action_updater, TiledGlb glb)
    {
        int iPp = 0;
        List<CcrSet> ccrSetsTmp;
        List<CcrClr> ccrClrsTmp;
        if (!_isGlobalPdp)
        {
            GetCcrSetAndClrVec(_ni).Deconstruct<List<CcrSet>, List<CcrClr>>(out ccrSetsTmp, out ccrClrsTmp);
            List<CcrSet> ccrsToSet = ccrSetsTmp;
            List<CcrClr> ccrsToClr = ccrClrsTmp;
            int[] paddingValues = ((TensorConst)_pdp1[GNNEPdp1.Padding]).Value.ToArray<int>();
            Padding paddingH = new Padding(paddingValues[0], paddingValues[1]);
            Padding paddingW = new Padding(paddingValues[2], paddingValues[3]);
            int[] filter = ((TensorConst)_pdp1[GNNEPdp1.Filter]).Value.ToArray<int>();
            int[] stride = ((TensorConst)_pdp1[GNNEPdp1.Stride]).Value.ToArray<int>();
            Segment1D inputRowSegment = TileUtilities.GetInputRowSegment(_ofmap[2].Start, _ofmap[2].Length,
                _inputShape[2], filter[0], stride[0], 1, in paddingH);
            Segment1D inputColSegment = TileUtilities.GetInputRowSegment(_ofmap[3].Start, _ofmap[3].Length,
                _inputShape[3], filter[1], stride[1], 1, in paddingW);
            SegmentND inputSegment = new SegmentND(_ifmap[0], _ifmap[1], inputRowSegment, inputColSegment);
            int bytesPerElement = TileUtilities.GetBytesPerElement(_pdp1[GNNEPdp1.Input].CheckedDataType);
            int glbDimC = glb.GlbMap[ItemName.Ifmap].Dimensions[1];
            int glbDimH = glb.GlbMap[ItemName.Ifmap].Dimensions[2];
            int glbDimW = glb.GlbMap[ItemName.Ifmap].Dimensions[3];
            int offsetS = (inputSegment[0].Start - _ifmap[0].Start) * glbDimC * glbDimH * glbDimW * bytesPerElement +
                          (inputSegment[1].Start - _ifmap[1].Start) * glbDimH * glbDimW * bytesPerElement +
                          (inputSegment[2].Start - _ifmap[2].Start) * glbDimW * bytesPerElement +
                          (inputSegment[3].Start - _ifmap[3].Start) * bytesPerElement + _ifmapOffset;
            action_updater.UpdateMfuPdp1(_pdp1, inputSegment, _ofmap, iPp, ccrsToSet, ccrsToClr, offsetS,
                _ni.Nb.OfmapOffset);
            return;
        }

        GetCcrSetAndClrVec(_ni).Deconstruct<List<CcrSet>, List<CcrClr>>(out ccrSetsTmp, out ccrClrsTmp);
        List<CcrSet> globalCcrsToSet = ccrSetsTmp;
        List<CcrClr> globalCcrsToClr = ccrClrsTmp;
        int filterH = ((TensorConst)_pdp1[GNNEPdp1.Filter]).Value.ToArray<int>()[0];
        (int R, int S) splitResult = SplitGlobalPdp(((TensorConst)_pdp1[GNNEPdp1.Filter]).Value.ToArray<int>()[1], filterH);
        int splitR = splitResult.R;
        int splitS = splitResult.S;
        SegmentND globalOfmap = new SegmentND(_ofmap);
        int rowSegmentCount = TileUtilities.GetSegmentStartEndLength(0, splitR, _inputShape[2]).Count;
        globalOfmap[2] = new Segment1D(..rowSegmentCount, Padding.Zero());
        TensorOnGlb originalGlbOfmap = glb.GlbMap[ItemName.Ofmap];
        glb.GlbMap[ItemName.Ofmap] =
            new TensorOnGlb(
                new int[4]
                {
                    globalOfmap[0].Length, globalOfmap[1].Length, globalOfmap[2].Length, originalGlbOfmap.Dimensions[3]
                }, DataTypes.Float16, 0, originalGlbOfmap.Mmu);
        List<SegmentND> ofmapSegments = new List<SegmentND> { globalOfmap };
        List<SegmentND> ifmapSegments = new List<SegmentND> { _ifmap };
        for (int i = 0; i < ofmapSegments.Count; i++)
        {
            SegmentND ofmapSegment = ofmapSegments[i];
            SegmentND ifmapSegment = ifmapSegments[i];
            List<Segment1D> rowSegments = TileUtilities.GetSegmentStartEndLength(0, splitR, _inputShape[2]);
            List<Segment1D> colSegments = TileUtilities.GetSegmentStartEndLength(0, splitS, _inputShape[3]);
            SegmentND ifmapSegmentCopy = new SegmentND(ifmapSegment);
            int alignedNum = TileUtilities.GetAlignedNum(ifmapSegment[3].Length,
                (TileUtilities.GetBytesPerElement(_inputType) == 1) ? 32 : 16);
            SegmentND alignedIfmap = new SegmentND(ifmapSegment[0], ifmapSegment[1], ifmapSegment[2],
                new Segment1D(..alignedNum, Padding.Zero()));
            alignedNum = TileUtilities.GetAlignedNum(colSegments.Count, 16);
            SegmentND partialOfmap = new SegmentND(ofmapSegment[0], ofmapSegment[1],
                new Segment1D(..rowSegments.Count, Padding.Zero()),
                new Segment1D(..colSegments.Count, Padding.Zero()));
            SegmentND alignedPartialOfmap = new SegmentND(ofmapSegment[0], ofmapSegment[1],
                new Segment1D(..rowSegments.Count, Padding.Zero()),
                new Segment1D(..alignedNum, Padding.Zero()));
            PDP_FUNCTION pdpOp = PdpFunc((((GNNEPdp1)_pdp1.Target).ReduceOp == MFU_PDP_OP.AVERAGE)
                ? MFU_PDP_OP.SUM
                : ((GNNEPdp1)_pdp1.Target).ReduceOp);
            Half sumScaleStage1 = (Half)(1f / (float)_inputShape[2]);
            Half sumScaleStage2 = (Half)(1f / (float)_inputShape[3]);
            for (int rowIdx = 0; rowIdx < rowSegments.Count; rowIdx++)
            {
                for (int colIdx = 0; colIdx < colSegments.Count; colIdx++)
                {
                    int isFirstTile = ((rowIdx == 0 && colIdx == 0) ? 1 : 0);
                    List<CcrClr> tileCcrsToClr = new List<CcrClr>();
                    if (isFirstTile > 0)
                    {
                        tileCcrsToClr.Add(globalCcrsToClr[0]);
                    }

                    SegmentND slice = new SegmentND(ifmapSegmentCopy[0], ifmapSegmentCopy[1], rowSegments[rowIdx],
                        colSegments[colIdx]);
                    SegmentND slice2 = new SegmentND(partialOfmap[0], partialOfmap[1],
                        new Segment1D(rowIdx..(rowIdx + 1), Padding.Zero()), new Segment1D(colIdx..(colIdx + 1), Padding.Zero()));
                    List<int> ofStride = new List<int> { alignedPartialOfmap[1].Length, alignedPartialOfmap[2].Length, alignedPartialOfmap[3].Length };
                    int offsetS2 =
                        TileUtilities.GetSliceOffsetInTensor(in alignedIfmap, in slice) *
                        TileUtilities.GetBytesPerElement(_inputType) + _ifmapOffset;
                    int offsetD =
                        TileUtilities.GetSliceOffsetInTensor(in alignedPartialOfmap, in slice2) *
                        TileUtilities.GetBytesPerElement(DataTypes.Float16) + _ofmapOffset;
                    action_updater.UpdateMfuGlobalPdp1(_pdp1, _inputType, DataTypes.Float16, pdpOp, slice, slice2, iPp,
                        sumScaleStage1, null, tileCcrsToClr, offsetS2, offsetD, ItemName.Ifmap, null, ofStride);
                }
            }

            SegmentND stage2Ifmap = partialOfmap;
            SegmentND singleCellOfmap = new SegmentND(partialOfmap[0], partialOfmap[1], new Segment1D(..1, Padding.Zero()),
                new Segment1D(..1, Padding.Zero()));
            List<int> ifStride = new List<int> { partialOfmap[1].Length, partialOfmap[2].Length, alignedPartialOfmap[3].Length };
            List<int> ofStride2 = new List<int>
            {
                partialOfmap[1].Length, singleCellOfmap[2].Length, originalGlbOfmap.Dimensions[3]
            };
            int ofmapOffset = _ofmapOffset;
            int ofmapOffset2 = _ofmapOffset;
            action_updater.UpdateMfuGlobalPdp1(_pdp1, DataTypes.Float16, _pdp1.CheckedDataType, pdpOp, stage2Ifmap,
                singleCellOfmap, iPp, sumScaleStage2, globalCcrsToSet, null, ofmapOffset, ofmapOffset2, ItemName.Ofmap, ifStride,
                ofStride2);
        }

        glb.GlbMap[ItemName.Ofmap] = originalGlbOfmap;

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

        static (int R, int S) SplitGlobalPdp(int w, int filterHeight)
        {
            int r = ((filterHeight > 16) ? 16 : filterHeight);
            int s = Math.Min(Math.Min(256 / r, w), 64);
            return (R: r, S: s);
        }
    }

    private void BuildScheduleTranspose(GnneActionUpdater actionUpdater, TiledGlb glb)
    {
        int iPp = 0;
        MFU_TRANS_PERMUTE perm = ((GNNETranspose)_transpose.Target).Perm;
        var (ccrsToSet, ccrsToClr) = GetCcrSetAndClrVec(_ni);
        actionUpdater.UpdateMfuTranspose(_ifmap, _ofmap, _inputType, perm, iPp, ccrsToSet, ccrsToClr, _ifmapOffset,
            _ofmapOffset);
    }

    private void BuildL1Schedule(GnneActionUpdater actionUpdater, TiledGlb glb, SegmentND ifmap1, SegmentND weight1,
        SegmentND psum, int iPp, int ofPp, int wPp, SegmentND ifmap2, bool weightGroupOnly,
        WeightGroupHandler weightGroup, Nncase.TIR.Buffer ddrW)
    {
        int psumPp = 0;
        List<int> l1Tiling = L1Search(glb, weight1, psum);
        List<Segment1D> psumRowSegs =
            TileUtilities.GetSegmentStartEndLength(psum[2].Start, l1Tiling[2], psum[2].End);
        List<Segment1D> psumColSegs =
            TileUtilities.GetSegmentStartEndLength(psum[3].Start, l1Tiling[3], psum[3].End);
        List<Segment1D> psumBatchSegs = TileUtilities.GetSegmentStartEndLength(psum[0].Start, 1, psum[0].End);
        List<Segment1D> weightRowSegs =
            TileUtilities.GetSegmentStartEndLength(weight1[2].Start, l1Tiling[4], weight1[2].End);
        List<Segment1D> weightColSegs =
            TileUtilities.GetSegmentStartEndLength(weight1[3].Start, l1Tiling[5], weight1[3].End);
        List<List<Segment1D>> l1McSeg = GetL1McSeg(ifmap1, psum, l1Tiling[1], l1Tiling[0]);
        List<Segment1D> outChannelSegs = l1McSeg[0];
        List<Segment1D> inChannelSegs = l1McSeg[1];
        int rowChunkSize = l1Tiling[4];
        int colChunkSize = Math.Min(GNNEEnv.PuKernelSpad / 2, l1Tiling[5]);
        if (_dilationH > 1)
        {
            rowChunkSize = 1;
        }

        if (_dilationW > 1)
        {
            colChunkSize = 1;
        }

        bool isFirstIfmap = true;
        bool isFirstWeight = true;
        bool isFirstOfmap = true;
        bool isFirstIf2 = true;
        foreach (Segment1D outChannelSeg in outChannelSegs)
        {
            foreach (Segment1D batchSeg in psumBatchSegs)
            {
                foreach (Segment1D psumRowSeg in psumRowSegs)
                {
                    Segment1D convInputRowSegment;
                    if (Conv1X1(_conv))
                    {
                        int outputRowStart = psumRowSeg.Start * _ofmapSt[2].Length;
                        int outputRowLength = psumRowSeg.Length * _ofmapSt[2].Length;
                        convInputRowSegment = TileUtilities.GetInputRowSegment(outputRowStart, outputRowLength,
                            _conv.CheckedShape.ToValueList()[2], _fusedKernelH, _fusedStrideH, _fusedDilationH,
                            in _fusedPaddingH);
                        convInputRowSegment /= _ofmapConv[2].Length;
                    }
                    else
                    {
                        convInputRowSegment = TileUtilities.GetInputRowSegment(psumRowSeg.Start, psumRowSeg.Length,
                            _convOutputShape[2], _fusedKernelH, _fusedStrideH, _fusedDilationH, in _fusedPaddingH);
                    }

                    Segment1D ifmapRowSegment = TileUtilities.GetInputRowSegment(convInputRowSegment.Start,
                        convInputRowSegment.Length, _inputShape[2], _weightsShape[2], _strideH, _dilationH, in _paddingH);
                    foreach (Segment1D psumColSeg in psumColSegs)
                    {
                        Segment1D convInputColSegment;
                        if (Conv1X1(_conv))
                        {
                            TileUtilities.Assert(psumColSeg.Length == _ofmap[3].Length, "f.Length == _ofmap[3].Length",
                                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                                1073);
                            int outputRowStart2 = psumColSeg.Start / _ofmapSt[2].Length;
                            int outputRowLength2 = psumColSeg.Length / _ofmapSt[2].Length;
                            convInputColSegment = TileUtilities.GetInputRowSegment(outputRowStart2, outputRowLength2,
                                _conv.CheckedShape.ToValueList()[3], _fusedKernelW, _fusedStrideW, _fusedDilationW,
                                in _fusedPaddingW);
                            convInputColSegment *= _ofmapConv[2].Length;
                        }
                        else
                        {
                            convInputColSegment = TileUtilities.GetInputColumnSegment(psumColSeg.Start, psumColSeg.Length,
                                _convOutputShape[3], _fusedKernelW, _fusedStrideW, _fusedDilationW, in _fusedPaddingW);
                        }

                        Segment1D ifmapColSegment = TileUtilities.GetInputColumnSegment(convInputColSegment.Start,
                            convInputColSegment.Length, _inputShape[3], _weightsShape[3], _strideW, _dilationW, in _paddingW);
                        SegmentND ofmapSlice = new SegmentND(batchSeg, outChannelSeg, psumRowSeg, psumColSeg);
                        RestoreTensorShape(_conv, ItemName.Ofmap, ofmapSlice);
                        SegmentND r2LPsum = new SegmentND(batchSeg, outChannelSeg, convInputRowSegment, convInputColSegment);
                        SegmentND psumSlice = new SegmentND(batchSeg, outChannelSeg, convInputRowSegment, convInputColSegment);
                        RestoreTensorShape(_conv, ItemName.Psum, psumSlice);
                        int icGroupStart = outChannelSeg.Start / _ocPerGroup * _icPerGroup;
                        int icGroupEnd = (outChannelSeg.End - 1) / _ocPerGroup * _icPerGroup + _icPerGroup;
                        Segment1D weightInChannelSeg = new Segment1D(..0, Padding.Zero());
                        bool isLoopStart = inChannelSegs[0].Start == 0;
                        foreach (Segment1D inChannelSeg in inChannelSegs)
                        {
                            if (inChannelSeg.Start < icGroupStart || inChannelSeg.End > icGroupEnd)
                            {
                                continue;
                            }

                            int icOffsetInGroup = inChannelSeg.Start % _icPerGroup;
                            weightInChannelSeg =
                                new Segment1D(
                                    new System.Range(end: Math.Min(icOffsetInGroup + inChannelSeg.Length, _icPerGroup), start: icOffsetInGroup),
                                    Padding.Zero());
                            foreach (Segment1D weightRowSeg in weightRowSegs)
                            {
                                foreach (Segment1D weightColSeg in weightColSegs)
                                {
                                    Segment1D ifmapChannelSeg = inChannelSeg;
                                    List<Segment1D> rowChunks =
                                        TileUtilities.GetSegmentStartEndLength(weightRowSeg.Start, rowChunkSize, weightRowSeg.End);
                                    List<Segment1D> colChunks =
                                        TileUtilities.GetSegmentStartEndLength(weightColSeg.Start, colChunkSize, weightColSeg.End);
                                    Segment1D inputBatchSeg = batchSeg;
                                    Segment1D outChannelSegForW = outChannelSeg;
                                    SegmentND x = new SegmentND(inputBatchSeg, ifmapChannelSeg, ifmapRowSegment,
                                        ifmapColSegment);
                                    SegmentND w = new SegmentND(outChannelSeg, weightInChannelSeg, weightRowSeg, weightColSeg);
                                    bool hasIfmapSlice = false;
                                    SegmentND shiftedInput = TileUtilities.ShiftInputTensor(in x, in w, _weightsShape[2],
                                        _weightsShape[3], _strideH, _strideW, _dilationH, _dilationW);
                                    if (shiftedInput[0].Length > 0 && shiftedInput[1].Length > 0 &&
                                        shiftedInput[2].Length > 0 && shiftedInput[3].Length > 0)
                                    {
                                        hasIfmapSlice = true;
                                        if (!weightGroupOnly)
                                        {
                                            int ifStatCount = _nodesG2LIfRec[_conv][_if1BufIdx][0].Item2.Stat_cnt();
                                            _nodesG2LIfRec[_conv][_if1BufIdx].RemoveAt(0);
                                            List<CcrClr> ifCcrsToClr = new List<CcrClr>();
                                            if (ifStatCount > 0)
                                            {
                                                ifCcrsToClr.Add(new CcrClr(
                                                    _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap,
                                                        _if1BufIdx))));
                                            }

                                            SegmentND slice = new SegmentND(shiftedInput);
                                            bool isReshaped = RestoreTensorShape(_conv);
                                            int strideNReshape = 0;
                                            int strideCReshape = 0;
                                            int strideHReshape = 0;
                                            if (isReshaped)
                                            {
                                                if (_preNi.Nb.AlignType == AlignedType.EAligned)
                                                {
                                                    strideNReshape = ifmap1[1].Length;
                                                    strideCReshape = ifmap1[2].Length;
                                                    strideHReshape = glb.GlbMap[ItemName.Ifmap].Dimensions[2] *
                                                                     glb.GlbMap[ItemName.Ifmap].Dimensions[3];
                                                }
                                                else if (_preNi.Nb.AlignType == AlignedType.FAligned)
                                                {
                                                    RestoreTensorShape(_conv, ItemName.Ifmap, slice);
                                                    strideNReshape = ifmap1[1].Length;
                                                    strideCReshape = ifmap1[2].Length * _ifmapLd[2].Length;
                                                    strideHReshape = glb.GlbMap[ItemName.Ifmap].Dimensions[3];
                                                }
                                                else
                                                {
                                                    strideNReshape = ifmap1[1].Length;
                                                    strideCReshape = ifmap1[2].Length;
                                                    strideHReshape = ifmap1[3].Length;
                                                }
                                            }

                                            actionUpdater.UpdateG2LIf(slice, ifmap1, _lif, iPp, null, ifCcrsToClr,
                                                _ifmapOffset, _inputType, ItemName.Ifmap, isReshaped, strideNReshape,
                                                strideCReshape, strideHReshape, _h2C, _weightsShape[2],
                                                ((TensorConst)_conv[GNNEConv2D.Stride]).Value.ToArray<int>()[0]);
                                        }
                                        else
                                        {
                                            if (_nodesG2LIfRec[_conv][_if1BufIdx].Count > 0)
                                            {
                                                List<Tuple<SegmentND, TensorStat>> ifRecList =
                                                    _nodesG2LIfRec[_conv][_if1BufIdx];
                                                ifRecList[ifRecList.Count - 1].Item2.IsLastSlice = isFirstIfmap;
                                            }

                                            _nodesG2LIfRec[_conv][_if1BufIdx]
                                                .Add(new Tuple<SegmentND, TensorStat>(ifmap1,
                                                    new TensorStat(isFirstIfmap, isLastSlice: false)));
                                            isFirstIfmap = false;
                                        }
                                    }

                                    foreach (Segment1D rowChunk in rowChunks)
                                    {
                                        foreach (Segment1D colChunk in colChunks)
                                        {
                                            SegmentND w2 = new SegmentND(outChannelSegForW, weightInChannelSeg, rowChunk, colChunk);
                                            SegmentND slice2 = TileUtilities.ShiftInputTensor(in x, in w2,
                                                _weightsShape[2], _weightsShape[3], _strideH, _strideW, _dilationH,
                                                _dilationW);
                                            int weightPasses = ((!(_weightType == DataTypes.Int16)) ? 1 : 2);
                                            int inputPasses = ((!(_inputType == DataTypes.Int16)) ? 1 : 2);
                                            for (int weightPassIdx = 0; weightPassIdx < weightPasses; weightPassIdx++)
                                            {
                                                for (int inputPassIdx = 0; inputPassIdx < inputPasses; inputPassIdx++)
                                                {
                                                    if (!weightGroupOnly)
                                                    {
                                                        int setWeightFake = 0;
                                                        int clearWeightBuf;
                                                        int weightSliceIdx;
                                                        ItemName wName;
                                                        int offsetWS;
                                                        if (_weightBufIdx != -1)
                                                        {
                                                            Tuple<SegmentND, TensorStat> weightRec =
                                                                _nodesWeightRec[_conv][_weightBufIdx][0];
                                                            Tuple<SegmentND, TensorStat> g2rwRec =
                                                                _nodesG2RWRec[_conv][_weightBufIdx][0];
                                                            _nodesG2RWRec[_conv][_weightBufIdx].RemoveAt(0);
                                                            if (g2rwRec.Item2.IsLastSlice)
                                                            {
                                                                _nodesWeightRec[_conv][_weightBufIdx].RemoveAt(0);
                                                            }

                                                            Tuple<SegmentND, TensorStat> g2rwSliceRec =
                                                                _nodesG2RWSliceRec[_conv][_weightBufIdx][0];
                                                            _nodesG2RWSliceRec[_conv][_weightBufIdx].RemoveAt(0);
                                                            TileUtilities.Assert(
                                                                w2 == g2rwSliceRec.Item1 && g2rwRec.Item1 == weightRec.Item1,
                                                                "l2RW == g2RWSliceRecStat.Item1 && g2RWRecStat.Item1 == weightRecStat.Item1",
                                                                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                                                                1210);
                                                            if (g2rwSliceRec.Item2.IsFirstSlice)
                                                            {
                                                                int clearWeightFake = ((_nodesQuenesAsW[_weightBufIdx][0].Item2 !=
                                                                    0 && g2rwRec.Item2.IsFirstSlice)
                                                                    ? 1
                                                                    : 0);
                                                                List<CcrClr> weightLoadCcrsToClr = new List<CcrClr>();
                                                                if (clearWeightFake > 0)
                                                                {
                                                                    weightLoadCcrsToClr.Add(new CcrClr(
                                                                        _ccrHandler.GetCcrItem(
                                                                            _ccrHandler.GetName(ItemName.WeightFake,
                                                                                _weightBufIdx))));
                                                                }

                                                                List<CcrSet> ccrsToSet = new List<CcrSet>
                                                                {
                                                                    new CcrSet(
                                                                        _ccrHandler.GetCcrItem(
                                                                            _ccrHandler.GetName(ItemName.Weight,
                                                                                (_weightBufIdx << 1) +
                                                                                (g2rwSliceRec.Item2.SliceIdx & 1))), 1)
                                                                };
                                                                actionUpdater.UpdateLoadW(w2, _lw, weightGroup, wPp,
                                                                    ddrW, ccrsToSet, weightLoadCcrsToClr, _ni.Nb.WeightOffset, _h2C,
                                                                    0, ItemName.Weight, weightPassIdx);
                                                            }

                                                            if (g2rwRec.Item2.IsLastSlice &&
                                                                _nodesQuenesAsW[_weightBufIdx].Count > 0)
                                                            {
                                                                setWeightFake = ((_nodesQuenesAsW[_weightBufIdx].Count > 1)
                                                                    ? 1
                                                                    : 0);
                                                                _nodesQuenesAsW[_weightBufIdx].RemoveAt(0);
                                                            }

                                                            clearWeightBuf = (g2rwSliceRec.Item2.IsFirstSlice ? 1 : 0);
                                                            weightSliceIdx = g2rwSliceRec.Item2.SliceIdx;
                                                            wName = ItemName.Weight;
                                                            offsetWS = _ni.Nb.WeightOffset;
                                                        }
                                                        else
                                                        {
                                                            setWeightFake = 0;
                                                            clearWeightBuf = 0;
                                                            weightSliceIdx = 0;
                                                            wName = ItemName.WeightPreload;
                                                            offsetWS = _ni.Nb.WeightPreloadOffset;
                                                        }

                                                        List<CcrSet> g2rwCcrsToSet = new List<CcrSet>();
                                                        if (setWeightFake > 0)
                                                        {
                                                            g2rwCcrsToSet.Add(new CcrSet(
                                                                _ccrHandler.GetCcrItem(
                                                                    _ccrHandler.GetName(ItemName.WeightFake,
                                                                        _weightBufIdx)), 1));
                                                        }

                                                        List<CcrClr> g2rwCcrsToClr = new List<CcrClr>();
                                                        if (clearWeightBuf > 0)
                                                        {
                                                            g2rwCcrsToClr.Add(new CcrClr(
                                                                _ccrHandler.GetCcrItem(
                                                                    _ccrHandler.GetName(ItemName.Weight,
                                                                        (_weightBufIdx << 1) + (weightSliceIdx & 1)))));
                                                        }

                                                        if (_ccrHandler.GetValue(
                                                                _ccrHandler.GetCcrItem(
                                                                    _ccrHandler.GetName(ItemName.WQarg))) > 0)
                                                        {
                                                            g2rwCcrsToClr.Add(new CcrClr(
                                                                _ccrHandler.GetCcrItem(
                                                                    _ccrHandler.GetName(ItemName.WQarg))));
                                                        }

                                                        actionUpdater.UpdateG2RW(w2, weightGroup, _ocPerGroup, _lw, wPp,
                                                            weightPassIdx, g2rwCcrsToSet, g2rwCcrsToClr, offsetWS, _ni.Nb.WeightQargOffset, wName,
                                                            ItemName.WQarg, _h2C);
                                                    }
                                                    else
                                                    {
                                                        weightGroup.UpdateWeightGroup(w2);
                                                        if (_weightBufIdx != -1)
                                                        {
                                                            if (_nodesG2RWRec[_conv][_weightBufIdx].Count > 0)
                                                            {
                                                                List<Tuple<SegmentND, TensorStat>> g2rwRecList =
                                                                    _nodesG2RWRec[_conv][_weightBufIdx];
                                                                g2rwRecList[g2rwRecList.Count - 1].Item2.IsLastSlice = isFirstWeight;
                                                            }

                                                            _nodesG2RWRec[_conv][_weightBufIdx]
                                                                .Add(new Tuple<SegmentND, TensorStat>(weight1,
                                                                    new TensorStat(isFirstWeight, isLastSlice: false)));
                                                            _nodesG2RWSliceRec[_conv][_weightBufIdx]
                                                                .Add(new Tuple<SegmentND, TensorStat>(w2,
                                                                    new TensorStat(isFirstSlice: false,
                                                                        isLastSlice: false, -1)));
                                                            isFirstWeight = false;
                                                        }
                                                    }

                                                    if (!weightGroupOnly)
                                                    {
                                                        actionUpdater.UpdateL2RIf(slice2, shiftedInput, _strideH,
                                                            _strideW, _icPerGroup, _lif, 0, inputPassIdx,
                                                            ((TensorConst)_conv[GNNEConv2D.DeqBias]).Value
                                                            .ToArray<int>()[0], _inputType, _h2C, _weightsShape[2]);
                                                    }

                                                    bool releaseIf = false;
                                                    bool isLastTile = colChunk == colChunks[colChunks.Count - 1] &&
                                                                      rowChunk == rowChunks[rowChunks.Count - 1] &&
                                                                      weightPassIdx == weightPasses - 1 && inputPassIdx == inputPasses - 1;
                                                    if (isLastTile && hasIfmapSlice)
                                                    {
                                                        releaseIf = true;
                                                        hasIfmapSlice = false;
                                                    }

                                                    bool loopStart = false;
                                                    if (isLoopStart && weightPassIdx == 0 && inputPassIdx == 0)
                                                    {
                                                        loopStart = true;
                                                        isLoopStart = false;
                                                    }

                                                    bool isLastWeightChunk = weightInChannelSeg.End == _weightsShape[1] &&
                                                                 colChunk.End == weight1[3].End &&
                                                                 rowChunk.End == weight1[2].End && weightPassIdx == weightPasses - 1 &&
                                                                 inputPassIdx == inputPasses - 1;
                                                    DataType ofType = _outputType;
                                                    ACT0_OUTPUT_DEST act0Dest = ACT0_OUTPUT_DEST.dm;
                                                    if ((object)_pool != null || (object)_dw != null ||
                                                        (object)_act1 != null)
                                                    {
                                                        act0Dest = ACT0_OUTPUT_DEST.psum;
                                                        ofType = _conv.CheckedDataType;
                                                    }

                                                    if (!weightGroupOnly)
                                                    {
                                                        int shift = ((TensorConst)_conv[GNNEConv2D.ShiftBits]).Value
                                                            .ToScalar<int>();
                                                        int setOfmapCcr = 0;
                                                        int clearOfmapFake = 0;
                                                        int clearAct = 0;
                                                        int ofmapCcrValue = 0;
                                                        if (isLastWeightChunk && act0Dest == ACT0_OUTPUT_DEST.dm)
                                                        {
                                                            Tuple<SegmentND, TensorStat> l2gOfRec =
                                                                _nodesL2GOfRec[_conv][_ofBufIdx][0];
                                                            _nodesL2GOfRec[_conv][_ofBufIdx].RemoveAt(0);
                                                            if (l2gOfRec.Item2.IsFirstSlice &&
                                                                _nodesQueNeedClearFake.Count > 0 &&
                                                                _nodesQueNeedClearFake[0].Item1 == _conv)
                                                            {
                                                                clearOfmapFake = ((_nodesQueNeedClearFake[0].Item2 != 0)
                                                                    ? 1
                                                                    : 0);
                                                                _nodesQueNeedClearFake.RemoveAt(0);
                                                            }

                                                            if (l2gOfRec.Item2.IsLastSlice)
                                                            {
                                                                setOfmapCcr = 1;
                                                                ofmapCcrValue = GetCcrSetAccordingPostNodes(_ni);
                                                            }
                                                        }

                                                        if (isLastWeightChunk && _ccrHandler.GetValue(
                                                                _ccrHandler.GetCcrItem(
                                                                    _ccrHandler.GetName(ItemName.Act))) > 0)
                                                        {
                                                            clearAct = 1;
                                                        }

                                                        List<CcrSet> psumCcrsToSet = new List<CcrSet>();
                                                        if (setOfmapCcr > 0)
                                                        {
                                                            psumCcrsToSet.Add(new CcrSet(
                                                                _ccrHandler.GetCcrItem(
                                                                    _ccrHandler.GetName(ItemName.Ofmap, _ofBufIdx)),
                                                                ofmapCcrValue));
                                                        }

                                                        List<CcrClr> psumCcrsToClr = new List<CcrClr>();
                                                        if (clearOfmapFake > 0)
                                                        {
                                                            psumCcrsToClr.Add(new CcrClr(
                                                                _ccrHandler.GetCcrItem(
                                                                    _ccrHandler.GetName(ItemName.OfmapFake,
                                                                        _ofBufIdx))));
                                                        }

                                                        List<CcrClr> actCcrsToClr = new List<CcrClr>();
                                                        if (clearAct > 0)
                                                        {
                                                            actCcrsToClr.Add(new CcrClr(
                                                                _ccrHandler.GetCcrItem(
                                                                    _ccrHandler.GetName(ItemName.Act))));
                                                        }

                                                        actionUpdater.UpdateR2LPsum(shift, r2LPsum, psumSlice,
                                                            _ofmapSt, ofPp, psumPp, act0Dest, releaseIf,
                                                            Math.Max(weightPassIdx, inputPassIdx), TcuComputeMode.NormalConv2d, loopStart,
                                                            isLastWeightChunk, _strideH, _strideW, _ocPerGroup, _inputType,
                                                            _weightType, ofType, _lact.CheckedDataType, psumCcrsToSet, psumCcrsToClr,
                                                            _ni.Nb.ActOffset, _ni.Nb.OfmapOffset, actCcrsToClr);
                                                    }
                                                    else if (isLastWeightChunk && act0Dest == ACT0_OUTPUT_DEST.dm)
                                                    {
                                                        if (_nodesL2GOfRec[_conv][_ofBufIdx].Count > 0)
                                                        {
                                                            List<Tuple<SegmentND, TensorStat>> ofRecList =
                                                                _nodesL2GOfRec[_conv][_ofBufIdx];
                                                            ofRecList[ofRecList.Count - 1].Item2.IsLastSlice = isFirstOfmap;
                                                        }

                                                        _nodesL2GOfRec[_conv][_ofBufIdx]
                                                            .Add(new Tuple<SegmentND, TensorStat>(psum,
                                                                new TensorStat(isFirstOfmap, isLastSlice: false)));
                                                        isFirstOfmap = false;
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
                                weightInChannelSeg.End != _weightsShape[1])
                            {
                                continue;
                            }

                            Segment1D emptySeg = new Segment1D(..0, new Padding(0, 0));
                            SegmentND l2RDw = new SegmentND(emptySeg, emptySeg, emptySeg, emptySeg);
                            SegmentND l2RIf = new SegmentND(emptySeg, emptySeg, emptySeg, emptySeg);
                            List<CcrClr> l2rCcrsToClr = new List<CcrClr>();
                            int offsetAct = _l1FuseNi.Nb.Pdp0ActOffset;
                            if ((object)_act1 != null && _act1[GNNEActivation.InputB] != None.Default)
                            {
                                l2RIf = psumSlice;
                                int index = ((_if2BufIdx != -1) ? _if2BufIdx : 0);
                                int if2StatCount = _nodesL2RIf2Rec[_conv][index][0].Item2.Stat_cnt();
                                _nodesL2RIf2Rec[_conv][index].RemoveAt(0);
                                if ((object)_lif2 != null)
                                {
                                    if (if2StatCount > 0)
                                    {
                                        l2rCcrsToClr.Add(new CcrClr(
                                            _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap2))));
                                    }
                                }
                                else if (if2StatCount > 0)
                                {
                                    l2rCcrsToClr.Add(new CcrClr(
                                        _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, _if2BufIdx))));
                                }
                            }

                            if ((object)_dw != null)
                            {
                                int[] dwWeightsShape = _dw[GNNEPdp0DW.Weights].CheckedShape.ToValueArray();
                                l2RDw = new SegmentND(new Segment1D(..1, Padding.Zero()), outChannelSeg,
                                    new Segment1D(..dwWeightsShape[2], Padding.Zero()),
                                    new Segment1D(..dwWeightsShape[3], Padding.Zero()));
                                offsetAct = _l1FuseNi.Nb.DwActOffset;
                            }

                            int setL2rOfmapCcr = 0;
                            int clearL2rOfmapFake = 0;
                            int l2rOfmapCcrValue = 0;
                            Tuple<SegmentND, TensorStat> l2gOfRecL2r = _nodesL2GOfRec[_conv][_ofBufIdx][0];
                            _nodesL2GOfRec[_conv][_ofBufIdx].RemoveAt(0);
                            if (l2gOfRecL2r.Item2.IsFirstSlice && _nodesQueNeedClearFake.Count > 0 &&
                                _nodesQueNeedClearFake[0].Item1 == _conv)
                            {
                                clearL2rOfmapFake = ((_nodesQueNeedClearFake[0].Item2 != 0) ? 1 : 0);
                                _nodesQueNeedClearFake.RemoveAt(0);
                            }

                            if (l2gOfRecL2r.Item2.IsLastSlice)
                            {
                                setL2rOfmapCcr = 1;
                                l2rOfmapCcrValue = GetCcrSetAccordingPostNodes(_ni.Children[0]);
                            }

                            List<CcrClr> dwCcrsToClr = new List<CcrClr>();
                            List<CcrClr> act1CcrsToClr = new List<CcrClr>();
                            if (_ccrHandler.GetValue(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.DwWeight))) >
                                0)
                            {
                                dwCcrsToClr.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.DwWeight))));
                            }

                            if (_ccrHandler.GetValue(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.DwQarg))) > 0)
                            {
                                dwCcrsToClr.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.DwQarg))));
                            }

                            if (_ccrHandler.GetValue(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.DwAct1))) > 0)
                            {
                                act1CcrsToClr.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.DwAct1))));
                            }

                            if (_ccrHandler.GetValue(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.MfuAct1))) > 0)
                            {
                                act1CcrsToClr.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.MfuAct1))));
                            }

                            if (_ccrHandler.GetValue(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.PdpAct1))) > 0)
                            {
                                act1CcrsToClr.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.PdpAct1))));
                            }

                            List<CcrSet> l2rCcrsToSet = new List<CcrSet>();
                            if (setL2rOfmapCcr > 0)
                            {
                                l2rCcrsToSet.Add(new CcrSet(
                                    _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, _ofBufIdx)), l2rOfmapCcrValue));
                            }

                            List<CcrClr> l2rFakeCcrsToClr = new List<CcrClr>();
                            if (clearL2rOfmapFake > 0)
                            {
                                l2rFakeCcrsToClr.Add(new CcrClr(
                                    _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.OfmapFake, _ofBufIdx))));
                            }

                            actionUpdater.UpdateL2RPsum(_dw, _pool, _act1, l2RIf, ifmap2, l2RDw, psumSlice, ofmapSlice,
                                _ofmapSt, ofPp, psumPp, ACT0_OUTPUT_DEST.dm, TcuComputeMode.NormalConv2d, _ocPerGroup,
                                weightGroup, l2rCcrsToClr, l2rCcrsToSet, l2rFakeCcrsToClr, _ifmap2Offset, offsetAct, _l1FuseNi.Nb.Act1Offset,
                                _ni.Nb.OfmapOffset, _l1FuseNi.Nb.DwWeightOffset, _l1FuseNi.Nb.DwWeightQargOffset,
                                _src2ItemName, dwCcrsToClr, act1CcrsToClr, _swapAB);
                            psumPp = (psumPp + 1) % 2;
                            continue;
                        }

                        if ((object)_act1 != null && weightInChannelSeg.End == _weightsShape[1] &&
                            _act1[GNNEActivation.InputB] != None.Default)
                        {
                            int index2 = ((_if2BufIdx != -1) ? _if2BufIdx : 0);
                            if (_nodesL2RIf2Rec[_conv][index2].Count > 0)
                            {
                                List<Tuple<SegmentND, TensorStat>> if2RecList = _nodesL2RIf2Rec[_conv][index2];
                                if2RecList[if2RecList.Count - 1].Item2.IsLastSlice = isFirstIf2;
                            }

                            _nodesL2RIf2Rec[_conv][index2]
                                .Add(new Tuple<SegmentND, TensorStat>(ifmap2,
                                    new TensorStat(isFirstIf2, isLastSlice: false)));
                            isFirstIf2 = false;
                        }

                        if (((object)_pool != null || (object)_dw != null || (object)_act1 != null) &&
                            weightInChannelSeg.End == _weightsShape[1])
                        {
                            if (_nodesL2GOfRec[_conv][_ofBufIdx].Count > 0)
                            {
                                List<Tuple<SegmentND, TensorStat>> ofRecListL2r = _nodesL2GOfRec[_conv][_ofBufIdx];
                                ofRecListL2r[ofRecListL2r.Count - 1].Item2.IsLastSlice = isFirstOfmap;
                            }

                            _nodesL2GOfRec[_conv][_ofBufIdx]
                                .Add(new Tuple<SegmentND, TensorStat>(psum, new TensorStat(isFirstOfmap, isLastSlice: false)));
                            isFirstOfmap = false;
                        }
                    }
                }
            }
        }
    }

    private List<List<Segment1D>> GetL1McSeg(SegmentND ifmap, SegmentND psum, int mInloop, int cInloop)
    {
        int groupCount = Math.Max(ifmap[1].Length / _icPerGroup, 1);
        List<Segment1D> outChannelSegs = new List<Segment1D>();
        List<Segment1D> inChannelSegs = new List<Segment1D>();
        if (groupCount >= 2 && _groupPerPass < 2)
        {
            for (int i = 0; i < groupCount; i++)
            {
                List<Segment1D> ocSegs =
                    TileUtilities.GetSegmentStartEndLength(psum[1].Start + i * _ocPerGroup, mInloop,
                        psum[1].Start + (i + 1) * _ocPerGroup);
                List<Segment1D> icSegs =
                    TileUtilities.GetSegmentStartEndLength(ifmap[1].Start + i * _icPerGroup, cInloop,
                        ifmap[1].Start + (i + 1) * _icPerGroup);
                outChannelSegs.AddRange(ocSegs);
                inChannelSegs.AddRange(icSegs);
            }
        }
        else
        {
            outChannelSegs = TileUtilities.GetSegmentStartEndLength(psum[1].Start, mInloop, psum[1].End);
            inChannelSegs = TileUtilities.GetSegmentStartEndLength(ifmap[1].Start, cInloop, ifmap[1].End);
        }

        return new List<List<Segment1D>> { outChannelSegs, inChannelSegs };
    }

    private int GetCcrSetAccordingPostNodes(NodeInfo currNode)
    {
        int outputsSize = (_l1Fused ? _l1FuseNi.Nb.OutputsSize : _ni.Nb.OutputsSize);
        int ccrSetCount = GetCcrSetCountForChild(currNode, currNode.Children[0].Op, outputsSize);
        if (outputsSize == 2)
        {
            ccrSetCount += GetCcrSetCountForChild(currNode, currNode.Children[1].Op, outputsSize);
        }

        return ccrSetCount;
    }

    private static int GetSliceCcrCount(TensorStat? stat)
    {
        return (stat != null && stat.IsFirstSlice && stat.IsLastSlice) ? 1 : 2;
    }

    // How many CCR tokens `currNode` has to set for one consumer (`child`) of its output.
    private int GetCcrSetCountForChild(NodeInfo currNode, Call child, int outputsSize)
    {
        bool inputFromL1 = false;
        if ((object)child != null)
        {
            Expr target = child.Target;
            if (target is GNNEConv2D)
            {
                inputFromL1 = true;
            }
            else if (target is GNNEActivation activation)
            {
                inputFromL1 = activation.InputFromL1.Count > 0 &&
                              ((activation.InputFromL1[0] && child[GNNEActivation.InputA] != currNode.Op) ||
                               (activation.InputFromL1[1] && child[GNNEActivation.InputB] != currNode.Op));
            }
            else if (target is Ai2dResize)
            {
                if (_nodesAi2dIfRec[child][_ofBufIdx].Count > 0)
                {
                    return GetSliceCcrCount(_nodesAi2dIfRec[child][_ofBufIdx][0].Item2);
                }

                return 0;
            }
        }

        if (!inputFromL1)
        {
            return (outputsSize != 2 || (object)child == null || !(child.Target is Concat)) ? 1 : 0;
        }

        Call source = child;
        bool sourceIsActivation = false;
        if (child.Target is GNNEActivation childActivation)
        {
            sourceIsActivation = true;
            source = childActivation.InputFromL1[0]
                ? ((Call)child[GNNEActivation.InputA])
                : ((Call)child[GNNEActivation.InputB]);
        }

        if (_nodesL2RIf2Rec.ContainsKey(source) && _nodesL2RIf2Rec[source][_ofBufIdx].Count > 0 &&
            sourceIsActivation)
        {
            return GetSliceCcrCount(_nodesL2RIf2Rec[source][_ofBufIdx][0].Item2);
        }

        if (_nodesG2LIfRec[source][_ofBufIdx].Count > 0)
        {
            return GetSliceCcrCount(_nodesG2LIfRec[source][_ofBufIdx][0].Item2);
        }

        return 0;
    }

    private bool Conv1X1(Call conv)
    {
        bool result = false;
        if ((object)conv == null)
        {
            return result;
        }

        int[] inputShape = conv[GNNEConv2D.Input].CheckedShape.ToValueArray();
        int[] outputShape = conv.CheckedShape.ToValueArray();
        int[] weightsShape = conv[GNNEConv2D.Weights].CheckedShape.ToValueArray();
        int[] paddingValues = ((TensorConst)conv[GNNEConv2D.Padding]).Value.ToArray<int>();
        Padding paddingH = new Padding(paddingValues[0], paddingValues[1]);
        Padding paddingW = new Padding(paddingValues[2], paddingValues[3]);
        int strideH = ((TensorConst)conv[GNNEConv2D.Stride]).Value.ToArray<int>()[0];
        int strideW = ((TensorConst)conv[GNNEConv2D.Stride]).Value.ToArray<int>()[1];
        int dilationH = ((TensorConst)conv[GNNEConv2D.Dilation]).Value.ToArray<int>()[0];
        int dilationW = ((TensorConst)conv[GNNEConv2D.Dilation]).Value.ToArray<int>()[1];
        int groups = ((TensorConst)conv[GNNEConv2D.Groups]).Value.ToScalar<int>();
        bool isDepthwise = inputShape[1] == outputShape[1] && outputShape[1] == groups && groups != 1;
        DataType inputType = conv[GNNEConv2D.Input].CheckedDataType;
        if ((object)conv != null && !isDepthwise && groups == 1 && strideH == 1 && strideW == 1 && weightsShape[2] == 1 && weightsShape[3] == 1 &&
            dilationH == 1 && dilationW == 1 && paddingH.Sum() == 0 && paddingW.Sum() == 0 &&
            (inputType == DataTypes.Int8 || inputType == DataTypes.UInt8) && outputShape[2] * outputShape[3] < 512 &&
            weightsShape[1] % 24 == 0 && _ifmapLd[2].Range.Equals(_ofmapSt[2].Range) &&
            _ifmapLd[3].Range.Equals(_ofmapSt[3].Range))
        {
            result = true;
        }

        return result;
    }

    private bool RestoreTensorShape(Call convCall, ItemName item_type = ItemName.None, SegmentND slice = null)
    {
        if (convCall == null)
        {
            throw new ArgumentNullException("convCall");
        }

        if (!Conv1X1(convCall))
        {
            return false;
        }

        switch (item_type)
        {
            case ItemName.None:
                return true;
            case ItemName.Ofmap:
            case ItemName.Psum:
                {
                    Segment1D ofmapRowSeg = _ofmapSt[2];
                    Segment1D ofmapColSeg = _ofmapSt[3];
                    if (item_type == ItemName.Psum)
                    {
                        ofmapRowSeg = _ofmapConv[2];
                        ofmapColSeg = _ofmapConv[3];
                    }

                    TileUtilities.Assert(
                        slice[3].Start % ofmapColSeg.Length == 0 && slice[3].End % ofmapColSeg.Length == 0,
                        "slice[3].Start % dim3.Length == 0 && slice[3].End % dim3.Length == 0",
                        "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                        1746);
                    slice[0] = slice[0];
                    slice[1] = new Segment1D((slice[1].Start * slice[2].Length)..(slice[1].End * slice[2].Length),
                        Padding.Zero());
                    slice[2] = new Segment1D(
                        (ofmapRowSeg.Start + slice[3].Start / ofmapColSeg.Length)..(ofmapRowSeg.Start +
                            slice[3].End / ofmapColSeg.Length),
                        ofmapRowSeg.Padding);
                    slice[3] = new Segment1D(ofmapColSeg.Start..ofmapColSeg.End, ofmapColSeg.Padding);
                    break;
                }
            default:
                {
                    Segment1D ifmapRowSeg = _ifmapLd[2];
                    Segment1D ifmapColSeg = _ifmapLd[3];
                    TileUtilities.Assert(
                        slice[3].Start % ifmapColSeg.Length == 0 && slice[3].End % ifmapColSeg.Length == 0,
                        "slice[3].Start % dim3.Length == 0 && slice[3].End % dim3.Length == 0",
                        "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                        1763);
                    slice[0] = slice[0];
                    slice[1] = slice[1];
                    slice[2] = new Segment1D(
                        (slice[2].Start * slice[3].Length / ifmapColSeg.Length)..(slice[2].End * slice[3].Length /
                            ifmapColSeg.Length), ifmapRowSeg.Padding);
                    slice[3] = new Segment1D((slice[3].Start / ifmapRowSeg.Length)..(slice[3].End / ifmapRowSeg.Length),
                        ifmapColSeg.Padding);
                    break;
                }
        }

        return true;
    }

    private List<int> L1Search(TiledGlb glb, SegmentND weight1, SegmentND psum)
    {
        bool isConv1X1 = Conv1X1(_conv);
        bool useFullWidth = (isConv1X1 || _l1FusedInfos[_conv] == L1FusedType.FusedAct1 ||
                     _l1FusedInfos[_conv] == L1FusedType.NoFused) && !_h2C;
        int icPerPass = Math.Min(GNNEEnv.PuHeight, _groupPerPass * _icPerGroup);
        int ocPerPass = Math.Min(GNNEEnv.PuWidth, _groupPerPass * _ocPerGroup);
        int h = 1;
        int w = 1;
        int e = 1;
        int f = ((!useFullWidth) ? 1 : _ofmap[3].Length);
        int fStep = ((!useFullWidth) ? 1 : _ofmap[3].Length);
        bool retriedWithMinWidth = false;
        int r;
        int s;
        int[] originalConvOutputShape;
        int hConvOut;
        int wConvOut;
        int psumPingPangSplit;
        int ifBytesPerElementGlb;
        bool fitsInL1;
        while (true)
        {
            r = ((_weightSplitPattern[_conv].Item1 == 0 || _dilationH > 3) ? 1 : _weightSplitPattern[_conv].Item1);
            s = ((_weightSplitPattern[_conv].Item2 == 0 || _dilationW > 3) ? 1 : _weightSplitPattern[_conv].Item2);
            originalConvOutputShape = _conv.CheckedShape.ToValueArray();
            if (isConv1X1)
            {
                int convRows = e * _ofmapSt[2].Length;
                hConvOut = SpaceSearcher.GetInputHeight(convRows, originalConvOutputShape[2], _fusedKernelH,
                    _ofmapSt[2].Length, _fusedStrideH, _fusedDilationH, in _fusedPaddingH);
                hConvOut /= _ofmapConv[2].Length;
            }
            else
            {
                hConvOut = SpaceSearcher.GetInputHeight(e, _convOutputShape[2], _fusedKernelH, _outputShape[2],
                    _fusedStrideH, _fusedDilationH, in _fusedPaddingH);
            }

            h = SpaceSearcher.GetInputHeight(hConvOut, _inputShape[2], r, _convOutputShape[2], _strideH, _dilationH,
                Padding.Zero());
            if (isConv1X1)
            {
                TileUtilities.Assert(f % _ofmapSt[2].Length == 0, "f % _ofmapSt[2].Length == 0",
                    "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                    1814);
                int convCols = f / _ofmapSt[2].Length;
                wConvOut = SpaceSearcher.GetInputHeight(convCols, originalConvOutputShape[3], _fusedKernelW,
                    _ofmapSt[3].Length, _fusedStrideW, _fusedDilationW, in _fusedPaddingW);
                wConvOut *= _ofmapConv[2].Length;
            }
            else
            {
                wConvOut = SpaceSearcher.GetInputHeight(f, _convOutputShape[3], _fusedKernelW, _outputShape[3],
                    _fusedStrideW, _fusedDilationW, in _fusedPaddingW);
            }

            w = SpaceSearcher.GetInputHeight(wConvOut, _inputShape[3], s, _convOutputShape[3], _strideW, _dilationW,
                Padding.Zero());
            psumPingPangSplit = 1;
            if ((object)_pool != null || (object)_dw != null || (object)_act1 != null)
            {
                psumPingPangSplit = 2;
            }

            ifBytesPerElementGlb = TileUtilities.GetBytesPerElement(_inputType);
            fitsInL1 = HandleL1Allocate(h, w, Math.Max(e, hConvOut), Math.Max(f, wConvOut), psumPingPangSplit,
                ifBytesPerElementGlb);
            if (fitsInL1 || retriedWithMinWidth)
            {
                break;
            }

            retriedWithMinWidth = true;
            f = 1;
            fStep = 1;
        }

        if (!fitsInL1)
        {
            throw new NotSupportedException("L1 is too small");
        }

        while (f < psum[3].Length && e < psum[2].Length && h * w * ifBytesPerElementGlb < GNNEEnv.IfL1SizePerChan / 2)
        {
            if (f <= e)
            {
                if (!IncreaseFByStep())
                {
                    break;
                }
            }
            else if (!IncreaseEByStep())
            {
                break;
            }
        }

        while (f < psum[3].Length && h * w * ifBytesPerElementGlb < GNNEEnv.IfL1SizePerChan / 2 && IncreaseFByStep())
        {
        }

        while (e < psum[2].Length && h * w * ifBytesPerElementGlb < GNNEEnv.IfL1SizePerChan / 2 && IncreaseEByStep())
        {
        }

        if (_weightSplitPattern[_conv].Equals(new Tuple<int, int>(0, 0)))
        {
            while (s < weight1[3].Length && r < weight1[2].Length && s < 31 && r < 31 && _dilationW <= 3 &&
                   _dilationH <= 3)
            {
                if (s <= r)
                {
                    if (!IncreaseSBy())
                    {
                        break;
                    }
                }
                else if (isConv1X1)
                {
                    if (!IncreaseRBy11X1Conv())
                    {
                        break;
                    }
                }
                else if (!IncreaseRBy())
                {
                    break;
                }
            }

            while (s < weight1[3].Length && s < 31 && _dilationW <= 3 && IncreaseSBy())
            {
            }

            while (r < weight1[2].Length && r < 31 && _dilationH <= 3)
            {
                if (isConv1X1)
                {
                    if (!IncreaseRBy11X1Conv())
                    {
                        break;
                    }
                }
                else if (!IncreaseRBy())
                {
                    break;
                }
            }
        }

        while (f < psum[3].Length && e < psum[2].Length)
        {
            if (f <= e)
            {
                if (!IncreaseFByStep())
                {
                    break;
                }
            }
            else if (!IncreaseEByStep())
            {
                break;
            }
        }

        while (f < psum[3].Length && IncreaseFByStep())
        {
        }

        while (e < psum[2].Length && IncreaseEByStep())
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

        _weightSplitPattern[_conv] = new Tuple<int, int>(r, s);
        return new List<int>
        {
            icPerPass,
            ocPerPass,
            e,
            f,
            r,
            s
        };

        bool IncreaseEByStep()
        {
            int newE = e + 1;
            int newHConvOut;
            if (isConv1X1)
            {
                newHConvOut = SpaceSearcher.GetInputHeight(newE * _ofmapSt[2].Length, originalConvOutputShape[2],
                    _fusedKernelH, _ofmapSt[2].Length, _fusedStrideH, _fusedDilationH, in _fusedPaddingH);
                newHConvOut /= _ofmapConv[2].Length;
            }
            else
            {
                newHConvOut = SpaceSearcher.GetInputHeight(newE, _convOutputShape[2], _fusedKernelH, _outputShape[2],
                    _fusedStrideH, _fusedDilationH, in _fusedPaddingH);
            }

            int newH = SpaceSearcher.GetInputHeight(newHConvOut, _inputShape[2], r, _convOutputShape[2],
                _strideH, _dilationH, new Padding(0, 0));
            bool fits = HandleL1Allocate(newH, w, Math.Max(newE, newHConvOut), Math.Max(f, wConvOut),
                psumPingPangSplit, ifBytesPerElementGlb);
            if (fits)
            {
                e = newE;
                h = newH;
                hConvOut = newHConvOut;
            }

            return fits;
        }

        bool IncreaseFByStep()
        {
            int newF = f + fStep;
            int newWConvOut;
            if (isConv1X1)
            {
                newWConvOut = SpaceSearcher.GetInputHeight(newF / _ofmapSt[2].Length, originalConvOutputShape[3],
                    _fusedKernelW, _ofmapSt[3].Length, _fusedStrideW, _fusedDilationW, in _fusedPaddingW);
                newWConvOut *= _ofmapConv[2].Length;
            }
            else
            {
                newWConvOut = SpaceSearcher.GetInputHeight(newF, _convOutputShape[3], _fusedKernelW, _outputShape[3],
                    _fusedStrideW, _fusedDilationW, in _fusedPaddingW);
            }

            int newW = SpaceSearcher.GetInputHeight(newWConvOut, _inputShape[3], s, _convOutputShape[3],
                _strideW, _dilationW, new Padding(0, 0));
            bool fits = HandleL1Allocate(h, newW, Math.Max(e, hConvOut), Math.Max(newF, newWConvOut),
                psumPingPangSplit, ifBytesPerElementGlb);
            if (fits)
            {
                f = newF;
                w = newW;
                wConvOut = newWConvOut;
            }

            return fits;
        }

        bool IncreaseRBy()
        {
            int newR = r + 1;
            int newHConvOut = SpaceSearcher.GetInputHeight(e, originalConvOutputShape[2], _fusedKernelH,
                _outputShape[2], _fusedStrideH, _fusedDilationH, in _fusedPaddingH);
            int newH = SpaceSearcher.GetInputHeight(newHConvOut, _inputShape[2], newR, _convOutputShape[2],
                _strideH, _dilationH, new Padding(0, 0));
            bool fits = HandleL1Allocate(newH, w, Math.Max(e, newHConvOut), Math.Max(f, wConvOut),
                psumPingPangSplit, ifBytesPerElementGlb);
            if (!fits)
            {
                return fits;
            }

            r = newR;
            h = newH;
            hConvOut = newHConvOut;
            return fits;
        }

        bool IncreaseRBy11X1Conv()
        {
            int newR = r + 1;
            int newHConvOut = SpaceSearcher.GetInputHeight(e, _convOutputShape[2], _fusedKernelH, _outputShape[2],
                _fusedStrideH, _fusedDilationH, in _fusedPaddingH);
            int newH = SpaceSearcher.GetInputHeight(newHConvOut, _inputShape[2], newR, _convOutputShape[2],
                _strideH, _dilationH, new Padding(0, 0));
            bool fits = HandleL1Allocate(newH, _ofmap[2].Length * _ofmap[3].Length, Math.Max(e, newHConvOut),
                Math.Max(f, wConvOut), psumPingPangSplit, ifBytesPerElementGlb);
            if (fits)
            {
                r = newR;
                h = newH;
                hConvOut = newHConvOut;
            }

            return fits;
        }

        bool IncreaseSBy()
        {
            int newS = s + 1;
            int newWConvOut = SpaceSearcher.GetInputHeight(f, _convOutputShape[3], _fusedKernelW, _outputShape[3],
                _fusedStrideW, _fusedDilationW, in _fusedPaddingW);
            int newW = SpaceSearcher.GetInputHeight(newWConvOut, _inputShape[3], newS, _convOutputShape[3],
                _strideW, _dilationW, new Padding(0, 0));
            bool fits = HandleL1Allocate(h, newW, Math.Max(e, hConvOut), Math.Max(f, newWConvOut),
                psumPingPangSplit, ifBytesPerElementGlb);
            if (fits)
            {
                s = newS;
                w = newW;
                wConvOut = newWConvOut;
            }

            return fits;
        }
    }

    private bool HandleL1Allocate(int h, int w, int e, int f, int psumPingPangSplit, int ifBytesPerElementGlb)
    {
        if (e * f > GNNEEnv.PsumL1ElePerChan / psumPingPangSplit)
        {
            return false;
        }

        return GNNEEnv.PuHeight * h * w * ifBytesPerElementGlb <= GNNEEnv.IfL1Size;
    }

    private void ArrangeWeights(DataType weightsType, int[] weightsShape, Span<byte> oldWeights,
        WeightGroupHandler weightGroup)
    {
        int bytesPerElement = TileUtilities.GetBytesPerElement(weightsType);
        List<SegmentND> weightSlices = weightGroup.WeightGroupSlice();
        byte[] arranged = new byte[oldWeights.Length];
        if (_h2C)
        {
            int offset = 0;
            int dstIdx = 0;
            foreach (SegmentND slice in weightSlices)
            {
                TileUtilities.Assert(offset == weightGroup.WeightGroupOffset(slice) * bytesPerElement,
                    "offset == weightGroup.WeightGroupOffset(slice) * bytesPerElement",
                    "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                    2129);
                for (int byteIdx = 0; byteIdx < bytesPerElement; byteIdx++)
                {
                    for (int n = 0; n < slice[0].Length; n++)
                    {
                        for (int col = 0; col < slice[3].Length; col++)
                        {
                            for (int row = 0; row < slice[2].Length; row++)
                            {
                                for (int ch = 0; ch < slice[1].Length; ch++)
                                {
                                    int srcIdx = ((((n + slice[0].Start) * weightsShape[1] + ch + slice[1].Start) *
                                                    weightsShape[2] + row + slice[2].Start) * weightsShape[3] + col +
                                                slice[3].Start) *
                                               bytesPerElement;
                                    arranged[dstIdx++] = oldWeights[srcIdx + byteIdx];
                                    offset++;
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            int[] inputShape = _conv[GNNEConv2D.Input].CheckedShape.ToValueArray();
            int[] outputShape = _conv.CheckedShape.ToValueArray();
            int[] convOutputShape = _conv.CheckedShape.ToValueArray();
            int[] weightsShape2 = _conv[GNNEConv2D.Weights].CheckedShape.ToValueArray();
            ReshapeConv(_conv, ref inputShape, ref outputShape, ref convOutputShape, ref weightsShape2);
            int offset = 0;
            int dstIdx = 0;
            foreach (SegmentND slice in weightSlices)
            {
                TileUtilities.Assert(offset == weightGroup.WeightGroupOffset(slice) * bytesPerElement,
                    "offset == weightGroup.WeightGroupOffset(slice) * bytesPerElement",
                    "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                    2162);
                for (int byteIdx = 0; byteIdx < bytesPerElement; byteIdx++)
                {
                    for (int n = 0; n < slice[0].Length; n++)
                    {
                        for (int row = 0; row < slice[2].Length; row++)
                        {
                            for (int col = 0; col < slice[3].Length; col++)
                            {
                                for (int ch = 0; ch < slice[1].Length; ch++)
                                {
                                    int srcIdx =
                                        ((((n + slice[0].Start) * weightsShape2[1] + ch + slice[1].Start) *
                                             weightsShape2[2] + row + slice[2].Start) * weightsShape2[3] + col +
                                         slice[3].Start) * bytesPerElement;
                                    arranged[dstIdx++] = oldWeights[srcIdx + byteIdx];
                                    offset++;
                                }
                            }
                        }
                    }
                }
            }
        }

        arranged.CopyTo(oldWeights);
    }

    private void ArrangeDwWeights(DataType weightsType, int[] weightsShape, Span<byte> oldWeights,
        WeightGroupHandler weightGroup)
    {
        int bytesPerElement = TileUtilities.GetBytesPerElement(weightsType);
        int copiedCount = 0;
        int dstRow = 0;
        byte[] arranged = new byte[oldWeights.Length];
        ref int dim0Ref = ref weightsShape[0];
        ref int dim1Ref = ref weightsShape[1];
        int oldDim1 = weightsShape[1];
        int oldDim0 = weightsShape[0];
        dim0Ref = oldDim1;
        dim1Ref = oldDim0;
        foreach (SegmentND slice in from wg in weightGroup.DwGroupSlice()
                 let cs = wg[1].Start
                 let ce = wg[1].End
                 let dwShape = _dw[GNNEPdp0DW.Weights].CheckedShape.ToValueArray()
                 select new SegmentND(..1, cs..ce, ..dwShape[2], ..dwShape[3]))
        {
            for (int n = 0; n < slice[0].Length; n++)
            {
                for (int row = 0; row < slice[2].Length; row++)
                {
                    for (int col = 0; col < slice[3].Length; col++)
                    {
                        for (int ch = 0; ch < slice[1].Length; ch++)
                        {
                            int srcIdx =
                                ((((n + slice[0].Start) * weightsShape[1] + ch + slice[1].Start) * weightsShape[2] +
                                  row + slice[2].Start) * weightsShape[3] + col + slice[3].Start) * bytesPerElement;
                            for (int byteIdx = 0; byteIdx < bytesPerElement; byteIdx++)
                            {
                                arranged[dstRow * GNNEEnv.PuWidth + ch] = oldWeights[srcIdx + byteIdx];
                            }

                            copiedCount++;
                        }

                        dstRow++;
                    }
                }
            }
        }

        arranged.CopyTo(oldWeights);
    }

    private void InitParameters(FusionInfo fusionInfo)
    {
        if (_weightGroups.Count > 0)
        {
            _weightGroups.Clear();
        }

        foreach (NodeInfo fusedNode in fusionInfo.FusedNodes)
        {
            Call op = fusedNode.Op;
            if ((object)op != null && op.Target is GNNEConv2D)
            {
                DataType checkedDataType = fusedNode.Op[GNNEConv2D.Weights].CheckedDataType;
                _weightGroups.Add(fusedNode.Op, new WeightGroupHandler(checkedDataType, checkedDataType));
                _weightSplitPattern.Add(fusedNode.Op, new Tuple<int, int>(0, 0));
            }
        }

        _l1FusedInfos.Clear();
    }

    private void GetSliceInfo(FusionInfo fusionInfo, TiledGlb glb, out List<List<NodeInfo>> currSliceInfo,
        out List<Dictionary<Call, NodeInfo>> preSliceInfo)
    {
        currSliceInfo = new List<List<NodeInfo>>();
        preSliceInfo = new List<Dictionary<Call, NodeInfo>>();
        List<NodeInfo> fusedNodes = fusionInfo.FusedNodes;
        int[] outputShape = fusedNodes[fusedNodes.Count - 1].Op.CheckedShape.ToValueArray();
        int[] lastOutShape = fusionInfo.LastOutShape;
        List<SegmentND> outputSlices = new List<SegmentND>();
        foreach (Segment1D glbOutputBatch in TileUtilities.GetSegmentStartEndLength(0, lastOutShape[0], outputShape[0]))
        {
            List<Segment1D> channelSegments =
                TileUtilities.GetSegmentStartEndLength(0, lastOutShape[1], outputShape[1]);
            outputSlices.AddRange(from glbOutputChannel in channelSegments
                let outputRowSeg = TileUtilities.GetSegmentStartEndLength(0, lastOutShape[2], outputShape[2])
                from glbOutputRow in outputRowSeg
                let outputColSeg = TileUtilities.GetSegmentStartEndLength(0, lastOutShape[3], outputShape[3])
                from glbOutputColumn in outputColSeg
                select new SegmentND(glbOutputBatch, glbOutputChannel, glbOutputRow, glbOutputColumn));
        }

        Dictionary<int, int> ofBufMap = new Dictionary<int, int> { { 0, 0 }, { 1, 1 }, { 2, 2 } };
        Dictionary<int, int> ofBufOffset = new Dictionary<int, int> { { 0, 0 }, { 1, 0 }, { 2, 0 } };
        Dictionary<int, int> weightBufOffset = new Dictionary<int, int> { { 0, 0 }, { 1, 0 } };
        List<NodeInfo> nodes = fusionInfo.FusedNodes;
        List<NodeInfo> currSlice;
        Call prevNode;
        for (int sliceIndex = 0; sliceIndex < outputSlices.Count; sliceIndex++)
        {
            currSlice = new List<NodeInfo>();
            Dictionary<Call, NodeInfo> sliceNodeInfos = new Dictionary<Call, NodeInfo>();
            for (int nodeIdx = nodes.Count - 1; nodeIdx >= 0; nodeIdx--)
            {
                Call op = nodes[nodeIdx].Op;
                if ((object)op != null && op.Target is GNNEStore)
                {
                    prevNode = (Call)op[GNNEStore.Input];
                    NodeBuffer nodeBuffer = GetPreNodeBuffer(prevNode);
                    nodeBuffer.OfBufferIndex = OfBufferIdx(prevNode);
                    nodeBuffer.OfmapOffset = GetOfBufOffset(nodeBuffer.OfBufferIndex);
                    nodeBuffer.WeightOffset = GetWeightBufOffset(prevNode, nodeBuffer, sliceIndex);
                    currSlice.Add(new NodeInfo(op, outputSlices[sliceIndex], nodes[nodeIdx].Nb, nodes[nodeIdx].Children));
                    sliceNodeInfos.Add(prevNode,
                        new NodeInfo(prevNode, outputSlices[sliceIndex], nodeBuffer,
                            new List<NodeInfo> { currSlice.Find((NodeInfo n) => n.Op == op) }));
                }

                if ((object)op != null)
                {
                    Expr target = op.Target;
                    if (!(target is GNNEConv2D))
                    {
                        if (!(target is GNNEPdp0DW))
                        {
                            if (!(target is GNNEPdp0Reduce))
                            {
                                if (!(target is GNNEPdp1))
                                {
                                    if (!(target is GNNETranspose))
                                    {
                                        if (!(target is Concat))
                                        {
                                            if (!(target is Ai2dResize))
                                            {
                                                if (!(target is GNNEActivation))
                                                {
                                                    if (target is GNNELoad)
                                                    {
                                                        currSlice.Add(sliceNodeInfos[op]);
                                                    }
                                                }
                                                else
                                                {
                                                    SegmentND opOfmap = sliceNodeInfos[op].Ofmap;
                                                    bool isSingleInput = op[GNNEActivation.InputB] == None.Default;
                                                    var (inputASeg, inputBSeg) = GetAct1InputsShape(
                                                        op[GNNEActivation.InputA].CheckedShape.ToValueArray(),
                                                        isSingleInput
                                                            ? op[GNNEActivation.InputA].CheckedShape.ToValueArray()
                                                            : op[GNNEActivation.InputB].CheckedShape.ToValueArray(),
                                                        op.CheckedShape.ToValueArray(), opOfmap);
                                                    if (nodes.Find((NodeInfo n) =>
                                                            n.Op == op[GNNEActivation.InputA]) != null)
                                                    {
                                                        SegmentND prevSegmentA = new SegmentND(inputASeg[0],
                                                            inputASeg[1], inputASeg[2], inputASeg[3]);
                                                        prevNode = (Call)op[GNNEActivation.InputA];
                                                        NodeBuffer prevNodeBuffer = GetPreNodeBuffer(prevNode);
                                                        prevNodeBuffer.OfBufferIndex = OfBufferIdx(prevNode);
                                                        prevNodeBuffer.OfmapOffset =
                                                            GetOfBufOffset(prevNodeBuffer.OfBufferIndex);
                                                        prevNodeBuffer.WeightOffset =
                                                            GetWeightBufOffset(prevNode, prevNodeBuffer, sliceIndex);
                                                        if (sliceNodeInfos.ContainsKey(prevNode))
                                                        {
                                                            SegmentND prevOfmap = sliceNodeInfos[prevNode].Ofmap;
                                                            sliceNodeInfos[prevNode].Nb = prevNodeBuffer;
                                                            sliceNodeInfos[prevNode].Children.Add(sliceNodeInfos[op]);
                                                            sliceNodeInfos[prevNode].Ofmap = prevOfmap + prevSegmentA;
                                                        }
                                                        else
                                                        {
                                                            sliceNodeInfos.Add(prevNode,
                                                                new NodeInfo(prevNode, prevSegmentA, prevNodeBuffer,
                                                                    new List<NodeInfo> { sliceNodeInfos[op] }));
                                                        }
                                                    }

                                                    if (!isSingleInput && nodes.Find((NodeInfo n) =>
                                                            n.Op == op[GNNEActivation.InputB]) != null)
                                                    {
                                                        SegmentND prevSegmentB = new SegmentND(inputBSeg[0],
                                                            inputBSeg[1], inputBSeg[2], inputBSeg[3]);
                                                        prevNode = (Call)op[GNNEActivation.InputB];
                                                        NodeBuffer prevNodeBuffer = GetPreNodeBuffer(prevNode);
                                                        prevNodeBuffer.OfBufferIndex = OfBufferIdx(prevNode);
                                                        prevNodeBuffer.OfmapOffset =
                                                            GetOfBufOffset(prevNodeBuffer.OfBufferIndex);
                                                        prevNodeBuffer.WeightOffset =
                                                            GetWeightBufOffset(prevNode, prevNodeBuffer, sliceIndex);
                                                        if (sliceNodeInfos.ContainsKey(prevNode))
                                                        {
                                                            SegmentND prevOfmap = sliceNodeInfos[prevNode].Ofmap;
                                                            sliceNodeInfos[prevNode].Nb = prevNodeBuffer;
                                                            sliceNodeInfos[prevNode].Children.Add(sliceNodeInfos[op]);
                                                            sliceNodeInfos[prevNode].Ofmap = prevOfmap + prevSegmentB;
                                                        }
                                                        else
                                                        {
                                                            sliceNodeInfos.Add(prevNode,
                                                                new NodeInfo(prevNode, prevSegmentB, prevNodeBuffer,
                                                                    new List<NodeInfo> { sliceNodeInfos[op] }));
                                                        }
                                                    }

                                                    currSlice.Add(sliceNodeInfos[op]);
                                                }
                                            }
                                            else
                                            {
                                                int[] resizeInputShape = op[Ai2dResize.Input].CheckedShape.ToValueArray();
                                                SegmentND opOfmap = sliceNodeInfos[_resize].Ofmap;
                                                Segment1D resizeInputH = new Segment1D(..resizeInputShape[2], Padding.Zero());
                                                Segment1D resizeInputW = new Segment1D(..resizeInputShape[3], Padding.Zero());
                                                SegmentND prevSegment = new SegmentND(opOfmap[0],
                                                    new Segment1D(..resizeInputShape[1], Padding.Zero()), resizeInputH, resizeInputW);
                                                prevNode = (Call)op[Ai2dResize.Input];
                                                NodeBuffer prevNodeBuffer = GetPreNodeBuffer(prevNode);
                                                prevNodeBuffer.OfBufferIndex = OfBufferIdx(prevNode);
                                                if (sliceNodeInfos.ContainsKey(prevNode))
                                                {
                                                    SegmentND prevOfmap = sliceNodeInfos[prevNode].Ofmap;
                                                    sliceNodeInfos[prevNode].Nb = prevNodeBuffer;
                                                    sliceNodeInfos[prevNode].Children.Add(sliceNodeInfos[op]);
                                                    sliceNodeInfos[prevNode].Ofmap = prevOfmap + prevSegment;
                                                }
                                                else
                                                {
                                                    sliceNodeInfos.Add(prevNode,
                                                        new NodeInfo(prevNode, prevSegment, prevNodeBuffer,
                                                            new List<NodeInfo> { sliceNodeInfos[op] }));
                                                }

                                                currSlice.Add(sliceNodeInfos[op]);
                                            }
                                        }
                                        else
                                        {
                                            int[] firstInputShape = op[Concat.Input][0].CheckedShape.ToValueArray();
                                            Segment1D firstN = new Segment1D(..firstInputShape[0], Padding.Zero());
                                            Segment1D firstC = new Segment1D(..firstInputShape[1], Padding.Zero());
                                            Segment1D firstH = new Segment1D(..firstInputShape[2], Padding.Zero());
                                            Segment1D firstW = new Segment1D(..firstInputShape[3], Padding.Zero());
                                            SegmentND firstInputSegment = new SegmentND(firstN, firstC, firstH,
                                                firstW);
                                            prevNode = (Call)op[Concat.Input][0];
                                            NodeBuffer firstNodeBuffer = GetPreNodeBuffer(prevNode);
                                            _ = nodes.Find((NodeInfo n) => n.Op == prevNode).Children;
                                            firstNodeBuffer.OfBufferIndex = OfBufferIdx(prevNode);
                                            firstNodeBuffer.OfmapOffset = GetOfBufOffset(firstNodeBuffer.OfBufferIndex);
                                            firstNodeBuffer.WeightOffset = GetWeightBufOffset(prevNode, firstNodeBuffer, sliceIndex);
                                            if (sliceNodeInfos.ContainsKey(prevNode))
                                            {
                                                SegmentND prevOfmap = sliceNodeInfos[prevNode].Ofmap;
                                                sliceNodeInfos[prevNode].Nb = firstNodeBuffer;
                                                sliceNodeInfos[prevNode].Children.Add(sliceNodeInfos[op]);
                                                sliceNodeInfos[prevNode].Ofmap = prevOfmap + firstInputSegment;
                                            }
                                            else
                                            {
                                                sliceNodeInfos.Add(prevNode,
                                                    new NodeInfo(prevNode, firstInputSegment, firstNodeBuffer,
                                                        new List<NodeInfo> { sliceNodeInfos[op] }));
                                            }

                                            int[] secondInputShape = op[Concat.Input][1].CheckedShape.ToValueArray();
                                            Segment1D secondN = new Segment1D(..secondInputShape[0], Padding.Zero());
                                            Segment1D secondC = new Segment1D(..secondInputShape[1], Padding.Zero());
                                            Segment1D secondH = new Segment1D(..secondInputShape[2], Padding.Zero());
                                            Segment1D secondW = new Segment1D(..secondInputShape[3], Padding.Zero());
                                            SegmentND secondInputSegment = new SegmentND(secondN, secondC, secondH,
                                                secondW);
                                            prevNode = (Call)op[Concat.Input][1];
                                            NodeBuffer secondNodeBuffer = GetPreNodeBuffer(prevNode);
                                            List<NodeInfo> children = nodes.Find((NodeInfo n) => n.Op == prevNode)
                                                .Children;
                                            secondNodeBuffer.OfBufferIndex = OfBufferIdx(prevNode);
                                            secondNodeBuffer.OfmapOffset = GetOfBufOffset(secondNodeBuffer.OfBufferIndex);
                                            secondNodeBuffer.WeightOffset = GetWeightBufOffset(prevNode, secondNodeBuffer, sliceIndex);
                                            if (sliceNodeInfos.ContainsKey(prevNode))
                                            {
                                                SegmentND prevOfmap = sliceNodeInfos[prevNode].Ofmap;
                                                sliceNodeInfos[prevNode] = new NodeInfo(prevNode, prevOfmap + secondInputSegment,
                                                    secondNodeBuffer, children);
                                            }
                                            else
                                            {
                                                sliceNodeInfos.Add(prevNode,
                                                    new NodeInfo(prevNode, secondInputSegment, secondNodeBuffer, children));
                                            }

                                            currSlice.Add(sliceNodeInfos[op]);
                                        }
                                    }
                                    else
                                    {
                                        op[GNNETranspose.Input].CheckedShape.ToValueArray();
                                        SegmentND opOfmap = sliceNodeInfos[op].Ofmap;
                                        MFU_TRANS_PERMUTE perm = ((GNNETranspose)op.Target).Perm;
                                        Segment1D[] inputSegs = GetInputSeg(opOfmap[0], opOfmap[1], opOfmap[2], opOfmap[3],
                                            perm);
                                        Segment1D inN = inputSegs[0];
                                        Segment1D inC = inputSegs[1];
                                        Segment1D inH = inputSegs[2];
                                        Segment1D inW = inputSegs[3];
                                        SegmentND prevSegment = new SegmentND(inN, inC, inH,
                                            inW);
                                        prevNode = (Call)op[GNNETranspose.Input];
                                        NodeBuffer prevNodeBuffer = GetPreNodeBuffer(prevNode);
                                        prevNodeBuffer.OfBufferIndex = OfBufferIdx(prevNode);
                                        prevNodeBuffer.OfmapOffset = GetOfBufOffset(prevNodeBuffer.OfBufferIndex);
                                        prevNodeBuffer.WeightOffset = GetWeightBufOffset(prevNode, prevNodeBuffer, sliceIndex);
                                        if (sliceNodeInfos.ContainsKey(prevNode))
                                        {
                                            SegmentND prevOfmap = sliceNodeInfos[prevNode].Ofmap;
                                            sliceNodeInfos[prevNode].Nb = prevNodeBuffer;
                                            sliceNodeInfos[prevNode].Children.Add(sliceNodeInfos[op]);
                                            sliceNodeInfos[prevNode].Ofmap = prevOfmap + prevSegment;
                                        }
                                        else
                                        {
                                            sliceNodeInfos.Add(prevNode,
                                                new NodeInfo(prevNode, prevSegment, prevNodeBuffer,
                                                    new List<NodeInfo> { sliceNodeInfos[op] }));
                                        }

                                        currSlice.Add(sliceNodeInfos[op]);
                                    }
                                }
                                else
                                {
                                    int[] inputShape = op[GNNEPdp1.Input].CheckedShape.ToValueArray();
                                    SegmentND opOfmap = sliceNodeInfos[op].Ofmap;
                                    int[] paddingValues = ((TensorConst)op[GNNEPdp1.Padding]).Value.ToArray<int>();
                                    Padding paddingH = new Padding(paddingValues[0], paddingValues[1]);
                                    Padding paddingW = new Padding(paddingValues[2], paddingValues[3]);
                                    int strideH = ((TensorConst)op[GNNEPdp1.Stride]).Value.ToArray<int>()[0];
                                    int strideW = ((TensorConst)op[GNNEPdp1.Stride]).Value.ToArray<int>()[1];
                                    int dilationH = 1;
                                    int dilationW = 1;
                                    int kernelH = ((TensorConst)op[GNNEPdp1.Filter]).Value.ToArray<int>()[0];
                                    int kernelW = ((TensorConst)op[GNNEPdp1.Filter]).Value.ToArray<int>()[1];
                                    Segment1D inputRowSegment = TileUtilities.GetInputRowSegment(opOfmap[2].Start,
                                        opOfmap[2].Length, inputShape[2], kernelH, strideH, dilationH, in paddingH);
                                    Segment1D inputColumnSegment = TileUtilities.GetInputColumnSegment(opOfmap[3].Start,
                                        opOfmap[3].Length, inputShape[3], kernelW, strideW, dilationW, in paddingW);
                                    SegmentND prevSegment = new SegmentND(opOfmap[0],
                                        new Segment1D(..inputShape[1], Padding.Zero()), inputRowSegment,
                                        inputColumnSegment);
                                    prevNode = (Call)op[GNNEPdp1.Input];
                                    NodeBuffer prevNodeBuffer = GetPreNodeBuffer(prevNode);
                                    prevNodeBuffer.OfBufferIndex = OfBufferIdx(prevNode);
                                    prevNodeBuffer.OfmapOffset = GetOfBufOffset(prevNodeBuffer.OfBufferIndex);
                                    prevNodeBuffer.WeightOffset = GetWeightBufOffset(prevNode, prevNodeBuffer, sliceIndex);
                                    if (sliceNodeInfos.ContainsKey(prevNode))
                                    {
                                        SegmentND prevOfmap = sliceNodeInfos[prevNode].Ofmap;
                                        sliceNodeInfos[prevNode].Nb = prevNodeBuffer;
                                        sliceNodeInfos[prevNode].Children.Add(sliceNodeInfos[op]);
                                        sliceNodeInfos[prevNode].Ofmap = prevOfmap + prevSegment;
                                    }
                                    else
                                    {
                                        sliceNodeInfos.Add(prevNode,
                                            new NodeInfo(prevNode, prevSegment, prevNodeBuffer,
                                                new List<NodeInfo> { sliceNodeInfos[op] }));
                                    }

                                    currSlice.Add(sliceNodeInfos[op]);
                                }
                            }
                            else
                            {
                                int[] inputShape = op[GNNEPdp0Reduce.Input].CheckedShape.ToValueArray();
                                SegmentND opOfmap = sliceNodeInfos[op].Ofmap;
                                int[] paddingValues = ((TensorConst)op[GNNEPdp0Reduce.Padding]).Value.ToArray<int>();
                                Padding paddingH = new Padding(paddingValues[0], paddingValues[1]);
                                Padding paddingW = new Padding(paddingValues[2], paddingValues[3]);
                                int strideH = ((TensorConst)op[GNNEPdp0Reduce.Stride]).Value.ToArray<int>()[0];
                                int strideW = ((TensorConst)op[GNNEPdp0Reduce.Stride]).Value.ToArray<int>()[1];
                                int dilationH = 1;
                                int dilationW = 1;
                                int kernelH = ((TensorConst)op[GNNEPdp0Reduce.Filter]).Value.ToArray<int>()[0];
                                int kernelW = ((TensorConst)op[GNNEPdp0Reduce.Filter]).Value.ToArray<int>()[1];
                                Segment1D inputRowSegment = TileUtilities.GetInputRowSegment(opOfmap[2].Start,
                                    opOfmap[2].Length, inputShape[2], kernelH, strideH, dilationH, in paddingH);
                                Segment1D inputColumnSegment = TileUtilities.GetInputColumnSegment(opOfmap[3].Start,
                                    opOfmap[3].Length, inputShape[3], kernelW, strideW, dilationW, in paddingW);
                                SegmentND prevSegment = new SegmentND(opOfmap[0],
                                    new Segment1D(..inputShape[1], Padding.Zero()), inputRowSegment, inputColumnSegment);
                                prevNode = (Call)op[GNNEPdp0Reduce.Input];
                                NodeBuffer prevNodeBuffer = GetPreNodeBuffer(prevNode);
                                prevNodeBuffer.OfBufferIndex = OfBufferIdx(prevNode);
                                prevNodeBuffer.OfmapOffset = GetOfBufOffset(prevNodeBuffer.OfBufferIndex);
                                prevNodeBuffer.WeightOffset = GetWeightBufOffset(prevNode, prevNodeBuffer, sliceIndex);
                                if (sliceNodeInfos.ContainsKey(prevNode))
                                {
                                    SegmentND prevOfmap = sliceNodeInfos[prevNode].Ofmap;
                                    sliceNodeInfos[prevNode].Nb = prevNodeBuffer;
                                    sliceNodeInfos[prevNode].Children.Add(sliceNodeInfos[op]);
                                    sliceNodeInfos[prevNode].Ofmap = prevOfmap + prevSegment;
                                }
                                else
                                {
                                    sliceNodeInfos.Add(prevNode,
                                        new NodeInfo(prevNode, prevSegment, prevNodeBuffer,
                                            new List<NodeInfo> { sliceNodeInfos[op] }));
                                }

                                currSlice.Add(sliceNodeInfos[op]);
                            }
                        }
                        else
                        {
                            int[] inputShape = op[GNNEPdp0DW.Input].CheckedShape.ToValueArray();
                            SegmentND opOfmap = sliceNodeInfos[op].Ofmap;
                            int[] weightsShape = op[GNNEPdp0DW.Weights].CheckedShape.ToValueArray();
                            int[] paddingValues = ((TensorConst)op[GNNEPdp0DW.Padding]).Value.ToArray<int>();
                            Padding paddingH = new Padding(paddingValues[0], paddingValues[1]);
                            Padding paddingW = new Padding(paddingValues[2], paddingValues[3]);
                            int strideH = ((TensorConst)op[GNNEPdp0DW.Stride]).Value.ToArray<int>()[0];
                            int strideW = ((TensorConst)op[GNNEPdp0DW.Stride]).Value.ToArray<int>()[1];
                            int dilationH = ((TensorConst)op[GNNEPdp0DW.Dilation]).Value.ToArray<int>()[0];
                            int dilationW = ((TensorConst)op[GNNEPdp0DW.Dilation]).Value.ToArray<int>()[1];
                            Segment1D inputRowSegment = TileUtilities.GetInputRowSegment(opOfmap[2].Start,
                                opOfmap[2].Length, inputShape[2], weightsShape[2], strideH, dilationH, in paddingH);
                            Segment1D inputColumnSegment = TileUtilities.GetInputColumnSegment(opOfmap[3].Start,
                                opOfmap[3].Length, inputShape[3], weightsShape[3], strideW, dilationW, in paddingW);
                            SegmentND prevSegment = new SegmentND(opOfmap[0],
                                new Segment1D(..inputShape[1], Padding.Zero()), inputRowSegment, inputColumnSegment);
                            prevNode = (Call)op[GNNEPdp0DW.Input];
                            NodeBuffer prevNodeBuffer = GetPreNodeBuffer(prevNode);
                            prevNodeBuffer.OfBufferIndex = OfBufferIdx(prevNode);
                            prevNodeBuffer.OfmapOffset = GetOfBufOffset(prevNodeBuffer.OfBufferIndex);
                            prevNodeBuffer.WeightOffset = GetWeightBufOffset(prevNode, prevNodeBuffer, sliceIndex);
                            if (sliceNodeInfos.ContainsKey(prevNode))
                            {
                                SegmentND prevOfmap = sliceNodeInfos[prevNode].Ofmap;
                                sliceNodeInfos[prevNode].Nb = prevNodeBuffer;
                                sliceNodeInfos[prevNode].Children.Add(sliceNodeInfos[op]);
                                sliceNodeInfos[prevNode].Ofmap = prevOfmap + prevSegment;
                            }
                            else
                            {
                                sliceNodeInfos.Add(prevNode,
                                    new NodeInfo(prevNode, prevSegment, prevNodeBuffer,
                                        new List<NodeInfo> { sliceNodeInfos[op] }));
                            }

                            currSlice.Add(sliceNodeInfos[op]);
                        }
                    }
                    else
                    {
                        int[] inputShape = op[GNNEConv2D.Input].CheckedShape.ToValueArray();
                        SegmentND opOfmap = sliceNodeInfos[op].Ofmap;
                        int[] weightsShape = op[GNNEConv2D.Weights].CheckedShape.ToValueArray();
                        int[] paddingValues = ((TensorConst)op[GNNEConv2D.Padding]).Value.ToArray<int>();
                        Padding paddingH = new Padding(paddingValues[0], paddingValues[1]);
                        Padding paddingW = new Padding(paddingValues[2], paddingValues[3]);
                        int strideH = ((TensorConst)op[GNNEConv2D.Stride]).Value.ToArray<int>()[0];
                        int strideW = ((TensorConst)op[GNNEConv2D.Stride]).Value.ToArray<int>()[1];
                        int dilationH = ((TensorConst)op[GNNEConv2D.Dilation]).Value.ToArray<int>()[0];
                        int dilationW = ((TensorConst)op[GNNEConv2D.Dilation]).Value.ToArray<int>()[1];
                        Segment1D inputRowSegment = TileUtilities.GetInputRowSegment(opOfmap[2].Start,
                            opOfmap[2].Length, inputShape[2], weightsShape[2], strideH, dilationH, in paddingH);
                        Segment1D inputColumnSegment = TileUtilities.GetInputColumnSegment(opOfmap[3].Start,
                            opOfmap[3].Length, inputShape[3], weightsShape[3], strideW, dilationW, in paddingW);
                        prevNode = (Call)op[GNNEConv2D.Input];
                        SegmentND prevSegment = new SegmentND(opOfmap[0], new Segment1D(..inputShape[1], Padding.Zero()),
                            inputRowSegment, inputColumnSegment);
                        NodeBuffer prevNodeBuffer = GetPreNodeBuffer(prevNode);
                        prevNodeBuffer.OfBufferIndex = OfBufferIdx(prevNode);
                        prevNodeBuffer.OfmapOffset = GetOfBufOffset(prevNodeBuffer.OfBufferIndex);
                        prevNodeBuffer.WeightOffset = GetWeightBufOffset(prevNode, prevNodeBuffer, sliceIndex);
                        if (sliceNodeInfos.ContainsKey(prevNode))
                        {
                            SegmentND prevOfmap = sliceNodeInfos[prevNode].Ofmap;
                            sliceNodeInfos[prevNode].Nb = prevNodeBuffer;
                            sliceNodeInfos[prevNode].Children.Add(sliceNodeInfos[op]);
                            sliceNodeInfos[prevNode].Ofmap = prevOfmap + prevSegment;
                        }
                        else
                        {
                            sliceNodeInfos.Add(prevNode,
                                new NodeInfo(prevNode, prevSegment, prevNodeBuffer,
                                    new List<NodeInfo> { sliceNodeInfos[op] }));
                        }

                        currSlice.Add(sliceNodeInfos[op]);
                    }
                }
            }

            currSlice.Reverse();
            foreach (NodeInfo sliceNode in currSlice)
            {
                sliceNode.Nb.OutputsSize = sliceNode.Children.Count;
            }

            foreach (NodeInfo info in sliceNodeInfos.Values)
            {
                info.Nb.OutputsSize = info.Children.Count;
            }

            currSliceInfo.Add(currSlice);
            preSliceInfo.Add(sliceNodeInfos);
            int bufCount = GetOfBufferNum();
            UpdateOfBufMap(bufCount);

            int GetOfBufferNum()
            {
                int ofBufferIndex = currSlice[0].Nb.OfBufferIndex;
                ofBufferIndex = currSlice.Select((NodeInfo node) => node.Nb.OfBufferIndex).Prepend(ofBufferIndex).Max();
                return ofBufferIndex + 1;
            }
        }

        glb.Items = fusionInfo.Mmu;
        glb.LastOutShape = lastOutShape;
        foreach (KeyValuePair<ItemName, MmuItem> mmuEntry in fusionInfo.Mmu)
        {
            glb.GlbMap.Add(mmuEntry.Key, new TensorOnGlb(new int[4] { 0, 0, 0, 0 }, DataTypes.Float16, 0));
            glb.GlbMap[mmuEntry.Key].Mmu = mmuEntry.Value;
        }

        glb.GlbMap.Add(ItemName.Ifmap, new TensorOnGlb(new int[4] { 0, 0, 0, 0 }, DataTypes.Float16, 0));
        glb.GlbMap[ItemName.Ifmap].Mmu = fusionInfo.Mmu[ItemName.Ofmap];
        if (!fusionInfo.Mmu.ContainsKey(ItemName.Ifmap2))
        {
            glb.GlbMap.Add(ItemName.Ifmap2, new TensorOnGlb(new int[4] { 0, 0, 0, 0 }, DataTypes.Float16, 0));
            glb.GlbMap[ItemName.Ifmap2].Mmu = fusionInfo.Mmu[ItemName.Ofmap];
        }

        static Segment1D[] GetInputSeg(Segment1D dimN, Segment1D dimC, Segment1D dimH,
            Segment1D dimW, MFU_TRANS_PERMUTE transposePerm)
        {
            return transposePerm switch
            {
                MFU_TRANS_PERMUTE.NCHW => new Segment1D[4] { dimN, dimC, dimH, dimW },
                MFU_TRANS_PERMUTE.NCWH => new Segment1D[4] { dimN, dimC, dimW, dimH },
                MFU_TRANS_PERMUTE.NHCW => new Segment1D[4] { dimN, dimH, dimC, dimW },
                MFU_TRANS_PERMUTE.NHWC => new Segment1D[4] { dimN, dimW, dimC, dimH },
                MFU_TRANS_PERMUTE.NWCH => new Segment1D[4] { dimN, dimH, dimW, dimC },
                MFU_TRANS_PERMUTE.NWHC => new Segment1D[4] { dimN, dimW, dimH, dimC },
                MFU_TRANS_PERMUTE.CNHW => new Segment1D[4] { dimC, dimN, dimH, dimW },
                MFU_TRANS_PERMUTE.CNWH => new Segment1D[4] { dimC, dimN, dimW, dimH },
                MFU_TRANS_PERMUTE.CHNW => new Segment1D[4] { dimH, dimN, dimC, dimW },
                MFU_TRANS_PERMUTE.CHWN => new Segment1D[4] { dimW, dimN, dimC, dimH },
                MFU_TRANS_PERMUTE.CWNH => new Segment1D[4] { dimH, dimN, dimW, dimC },
                MFU_TRANS_PERMUTE.CWHN => new Segment1D[4] { dimW, dimN, dimH, dimC },
                MFU_TRANS_PERMUTE.HNCW => new Segment1D[4] { dimC, dimH, dimN, dimW },
                MFU_TRANS_PERMUTE.HNWC => new Segment1D[4] { dimC, dimW, dimN, dimH },
                MFU_TRANS_PERMUTE.HCNW => new Segment1D[4] { dimH, dimC, dimN, dimW },
                MFU_TRANS_PERMUTE.HCWN => new Segment1D[4] { dimW, dimC, dimN, dimH },
                MFU_TRANS_PERMUTE.HWNC => new Segment1D[4] { dimH, dimW, dimN, dimC },
                MFU_TRANS_PERMUTE.HWCN => new Segment1D[4] { dimW, dimH, dimN, dimC },
                MFU_TRANS_PERMUTE.WNCH => new Segment1D[4] { dimC, dimH, dimW, dimN },
                MFU_TRANS_PERMUTE.WNHC => new Segment1D[4] { dimC, dimW, dimH, dimN },
                MFU_TRANS_PERMUTE.WCNH => new Segment1D[4] { dimH, dimC, dimW, dimN },
                MFU_TRANS_PERMUTE.WCHN => new Segment1D[4] { dimW, dimC, dimH, dimN },
                MFU_TRANS_PERMUTE.WHNC => new Segment1D[4] { dimH, dimW, dimC, dimN },
                MFU_TRANS_PERMUTE.WHCN => new Segment1D[4] { dimW, dimH, dimC, dimN },
                _ => new Segment1D[4] { dimN, dimC, dimH, dimW },
            };
        }

        int GetOfBufOffset(int bufIdx)
        {
            foreach (NodeInfo fusedNode in fusionInfo.FusedNodes)
            {
                TileUtilities.Assert(
                    ofBufOffset[fusedNode.Nb.OfBufferIndex] == 0 ||
                    ofBufOffset[fusedNode.Nb.OfBufferIndex] == fusedNode.Nb.OfmapOffset,
                    "ofBufOffset[node.Nb.OfBufferIndex] == 0 || ofBufOffset[node.Nb.OfBufferIndex] == node.Nb.OfmapOffset",
                    "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                    2294);
                ofBufOffset[fusedNode.Nb.OfBufferIndex] = fusedNode.Nb.OfmapOffset;
            }

            if (bufIdx != -1)
            {
                return ofBufOffset[bufIdx];
            }

            return -1;
        }

        NodeBuffer GetPreNodeBuffer(Call node)
        {
            foreach (NodeInfo fusedNode in fusionInfo.FusedNodes)
            {
                if (node == fusedNode.Op)
                {
                    return new NodeBuffer(fusedNode.Nb);
                }
            }

            return new NodeBuffer();
        }

        int GetWeightBufOffset(Call curNode, NodeBuffer nb, int sliceIdx)
        {
            int convCount = 0;
            int weightOffset = 0;
            foreach (NodeInfo fusedNode in fusionInfo.FusedNodes)
            {
                Call nodeOp = fusedNode.Op;
                if ((object)nodeOp != null && nodeOp.Target is GNNEConv2D)
                {
                    TileUtilities.Assert(
                        fusedNode.Nb.WeightOffset == weightBufOffset[0] ||
                        fusedNode.Nb.WeightOffset == weightBufOffset[1] ||
                        (weightBufOffset[0] == 0 && weightBufOffset[1] == 0),
                        "node.Nb.WeightOffset == weightBufOffset[0] || node.Nb.WeightOffset == weightBufOffset[1] || (weightBufOffset[0] == 0 && weightBufOffset[1] == 0)",
                        "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                        2312);
                    weightBufOffset[(fusedNode.Nb.WeightOffset != 0) ? 1 : 0] = fusedNode.Nb.WeightOffset;
                    convCount++;
                }

                if (fusedNode.Op == curNode)
                {
                    weightOffset = nb.WeightOffset;
                }
            }

            if ((object)curNode != null && curNode.Target is GNNEConv2D)
            {
                int key = ((weightOffset != 0) ? 1 : (sliceIdx * convCount)) & 1;
                weightOffset = weightBufOffset[key];
            }

            return weightOffset;
        }

        int OfBufferIdx(Call currNode)
        {
            foreach (NodeInfo fusedNode in fusionInfo.FusedNodes)
            {
                if (currNode == fusedNode.Op)
                {
                    return ofBufMap[fusedNode.Nb.OfBufferIndex];
                }
            }

            return -1;
        }

        void UpdateOfBufMap(int bufCount)
        {
            ofBufMap[0] = (currSlice[currSlice.Count - 2].Nb.OfBufferIndex + 1) % bufCount;
            ofBufMap[1] = (currSlice[currSlice.Count - 2].Nb.OfBufferIndex + 2) % bufCount;
            ofBufMap[2] = (currSlice[currSlice.Count - 2].Nb.OfBufferIndex + 3) % bufCount;
        }
    }

    private Tuple<SegmentND, SegmentND> GetAct1InputsShape(int[] inputAShape, int[] inputBShape, int[] outputShape,
        SegmentND slice)
    {
        int[] minShape = new int[4];
        for (int i = 0; i < 4; i++)
        {
            minShape[i] = Math.Min(inputAShape[i], inputBShape[i]);
        }

        int[] aScale = new int[4];
        int[] bScale = new int[4];
        int[] outScale = new int[4];
        for (int j = 0; j < 4; j++)
        {
            aScale[j] = ((minShape[j] == 1) ? 1 : (inputAShape[j] / minShape[j]));
            bScale[j] = ((minShape[j] == 1) ? 1 : (inputBShape[j] / minShape[j]));
            outScale[j] = ((minShape[j] == 1) ? 1 : (outputShape[j] / minShape[j]));
        }

        Segment1D sliceN = slice[0];
        Segment1D inputASegN = sliceN / (outScale[0] / aScale[0]);
        Segment1D inputBSegN = sliceN / (outScale[0] / bScale[0]);
        if (inputAShape[0] == 1)
        {
            inputASegN = new Segment1D(..1, Padding.Zero());
        }

        if (inputBShape[0] == 1)
        {
            inputBSegN = new Segment1D(..1, Padding.Zero());
        }

        Segment1D sliceC = slice[1];
        Segment1D inputASegC = sliceC / (outScale[1] / aScale[1]);
        Segment1D inputBSegC = sliceC / (outScale[1] / bScale[1]);
        if (inputAShape[1] == 1)
        {
            inputASegC = new Segment1D(..1, Padding.Zero());
        }

        if (inputBShape[1] == 1)
        {
            inputBSegC = new Segment1D(..1, Padding.Zero());
        }

        Segment1D sliceH = slice[2];
        Segment1D inputASegH = sliceH / (outScale[2] / aScale[2]);
        Segment1D inputBSegH = sliceH / (outScale[2] / bScale[2]);
        if (inputAShape[2] == 1)
        {
            inputASegH = new Segment1D(..1, Padding.Zero());
        }

        if (inputBShape[2] == 1)
        {
            inputBSegH = new Segment1D(..1, Padding.Zero());
        }

        Segment1D sliceW = slice[3];
        Segment1D inputASegW = sliceW / (outScale[3] / aScale[3]);
        Segment1D inputBSegW = sliceW / (outScale[3] / bScale[3]);
        if (inputAShape[3] == 1)
        {
            inputASegW = new Segment1D(..1, Padding.Zero());
        }

        if (inputBShape[3] == 1)
        {
            inputBSegW = new Segment1D(..1, Padding.Zero());
        }

        SegmentND inputASegment = new SegmentND(inputASegN, inputASegC, inputASegH, inputASegW);
        SegmentND inputBSegment = new SegmentND(inputBSegN, inputBSegC, inputBSegH, inputBSegW);
        return new Tuple<SegmentND, SegmentND>(inputASegment, inputBSegment);
    }

    private void ItemRecStatusInit(List<List<NodeInfo>> currSliceInfo)
    {
        _nodesWeightRec.Clear();
        _nodesOfmapRec.Clear();
        _nodesG2LIfRec.Clear();
        _nodesG2RWRec.Clear();
        _nodesL2GOfRec.Clear();
        _nodesL2RIf2Rec.Clear();
        _nodesG2RWSliceRec.Clear();
        _nodesAi2dIfRec.Clear();
        _nodesAi2dOfRec.Clear();
        _nodesQuenesAsW.Clear();
        _nodesQuenesAsW = new List<List<Tuple<Call, int>>>(2)
        {
            new List<Tuple<Call, int>>(), new List<Tuple<Call, int>>()
        };
        _nodesQueNeedClearFake.Clear();
        for (int i = 0; i < currSliceInfo.Count - 1; i++)
        {
            List<NodeInfo> prevSlice = currSliceInfo[i];
            int ofBufferIndex = prevSlice[prevSlice.Count - 2].Nb.OfBufferIndex;
            for (int j = 0; j < currSliceInfo[i + 1].Count - 1; j++)
            {
                NodeInfo nodeInfo = currSliceInfo[i + 1][j];
                if (nodeInfo.Nb.OfBufferIndex == ofBufferIndex)
                {
                    Call op = nodeInfo.Op;
                    if ((object)op != null && op.Target is GNNEConv2D)
                    {
                        _nodesQueNeedClearFake.Add(new Tuple<Call, int>(nodeInfo.Op, i + 1));
                    }

                    break;
                }
            }
        }

        if (_nodesQueNeedClearFake.Count > 0)
        {
            _nodesQueNeedClearFake.Insert(0, new Tuple<Call, int>(_nodesQueNeedClearFake[0].Item1, 0));
        }
    }

    private void UpdateL2FusePara(FusionInfo fusionInfo, NodeInfo curNode, Dictionary<Call, NodeInfo> sliceInfo,
        TiledGlb glb, bool weightGroupOnly = false, bool firstLayer = false)
    {
        _conv = null;
        _pool = null;
        _dw = null;
        _act1 = null;
        _lif = null;
        _lw = null;
        _lact = null;
        _lwQarg = null;
        _sof = null;
        _lif2 = null;
        _pdp1 = null;
        _transpose = null;
        _cat = null;
        _resize = null;
        _preNi = null;
        _ni = null;
        _l1FuseNi = null;
        _l1Fused = false;
        if (firstLayer)
        {
            _l1FusedInfos.Clear();
        }

        _l1FusedInfos.Add(curNode.Op, L1FusedType.NoFused);
        _swapAB = false;
        _h2C = false;
        _src2ItemName = ItemName.Ifmap2;
        Segment1D emptySegment = new Segment1D(..0, Padding.Zero());
        _ifmap = new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment);
        _ifmap2 = _ifmap;
        _ofmap = _ifmap;
        _ofmapSt = _ifmap;
        _ifmapLd = _ifmap;
        _ofmapConv = _ifmap;
        _ifmapA = _ifmap;
        _ifmapB = _ifmap;
        _if2Type = DataTypes.Float16;
        _ni = curNode;
        _if1BufIdx = -1;
        _if2BufIdx = -1;
        _ofBufIdx = -1;
        _weightBufIdx = -1;
        Call op = curNode.Op;
        _lif = (((object)op != null && op.Target is GNNELoad && !(curNode.Op[GNNELoad.Input] is TensorConst))
            ? curNode.Op
            : null);
        if ((object)_lif != null)
        {
            _ifmap = _ni.Ofmap;
            _ifmapOffset = 0;
            _ofmap = _ni.Ofmap;
            _if1BufIdx = -1;
            _if2BufIdx = -1;
            _ofBufIdx = _ni.Nb.OfBufferIndex;
            _inputType = _lif[GNNELoad.Input].CheckedDataType;
            _outputType = _lif.CheckedDataType;
            Call op2 = curNode.Children[0].Op;
            if ((object)op2 != null && op2.Target is GNNEConv2D)
            {
                int[] nextWeightsShape = op2[GNNEConv2D.Weights].CheckedShape.ToValueArray();
                int[] nextInputShape = op2[GNNEConv2D.Input].CheckedShape.ToValueArray();
                int[] nextOutputShape = op2.CheckedShape.ToValueArray();
                int nextGroups = ((TensorConst)op2[GNNEConv2D.Groups]).Value.ToScalar<int>();
                bool nextIsDepthwise = nextInputShape[1] == nextOutputShape[1] && nextOutputShape[1] == nextGroups && nextGroups != 1;
                int nextDeqBias = ((TensorConst)op2[GNNEConv2D.DeqBias]).Value.ToScalar<int>();
                _h2C = (object)op2 != null && op2.Target is GNNEConv2D && nextWeightsShape[1] * nextWeightsShape[2] <= GNNEEnv.PuHeight &&
                       nextWeightsShape[2] != 1 && nextInputShape[2] > 200 && nextInputShape[3] > 200 && !nextIsDepthwise && _outputType != DataTypes.Int16;
                _memsetValue = (_h2C ? nextDeqBias : 0);
            }
        }

        op = curNode.Op;
        if ((object)op != null && op.Target is GNNEStore)
        {
            _sof = curNode.Op;
            Call storeInput = (Call)_sof[GNNEStore.Input];
            _ifmap = sliceInfo[storeInput].Ofmap;
            _ifmapOffset = sliceInfo[storeInput].Nb.OfmapOffset;
            _ofmap = _ni.Ofmap;
            _if1BufIdx = sliceInfo[storeInput].Nb.OfBufferIndex;
            _if2BufIdx = -1;
            _ofBufIdx = sliceInfo[storeInput].Nb.OfBufferIndex;
            _inputType = storeInput.CheckedDataType;
            _outputType = _sof.CheckedDataType;
        }

        op = curNode.Op;
        if ((object)op != null && op.Target is GNNEConv2D)
        {
            _conv = curNode.Op;
            _preNi = sliceInfo[(Call)_conv[GNNEConv2D.Input]];
            _inputShape = _conv[GNNEConv2D.Input].CheckedShape.ToValueArray();
            _outputShape = _conv.CheckedShape.ToValueArray();
            _convOutputShape = _outputShape;
            _weightsShape = _conv[GNNEConv2D.Weights].CheckedShape.ToValueArray();
            int[] paddingValues = ((TensorConst)_conv[GNNEConv2D.Padding]).Value.ToArray<int>();
            _paddingH = new Padding(paddingValues[0], paddingValues[1]);
            _paddingW = new Padding(paddingValues[2], paddingValues[3]);
            _strideH = ((TensorConst)_conv[GNNEConv2D.Stride]).Value.ToArray<int>()[0];
            _strideW = ((TensorConst)_conv[GNNEConv2D.Stride]).Value.ToArray<int>()[1];
            _dilationH = ((TensorConst)_conv[GNNEConv2D.Dilation]).Value.ToArray<int>()[0];
            _dilationW = ((TensorConst)_conv[GNNEConv2D.Dilation]).Value.ToArray<int>()[1];
            _groups = ((TensorConst)_conv[GNNEConv2D.Groups]).Value.ToScalar<int>();
            int deqBias = ((TensorConst)_conv[GNNEConv2D.DeqBias]).Value.ToScalar<int>();
            bool isDepthwise = _inputShape[1] == _outputShape[1] && _outputShape[1] == _groups && _groups != 1;
            _icPerGroup = ((_groups == 1) ? _weightsShape[1] : (_inputShape[1] / _groups));
            _ocPerGroup = _outputShape[1] / _groups;
            _groupPerPass =
                Math.Min(Math.Max(Math.Min(GNNEEnv.PuHeight / _icPerGroup, GNNEEnv.PuWidth / _ocPerGroup), 1), _groups);
            op = curNode.Children[0].Op;
            if ((object)op != null && op.Target is GNNEPdp0Reduce)
            {
                _pool = curNode.Children[0].Op;
            }

            op = curNode.Children[0].Op;
            if ((object)op != null && op.Target is GNNEPdp0DW)
            {
                _dw = curNode.Children[0].Op;
            }

            op = curNode.Children[0].Op;
            if ((object)op != null && op.Target is GNNEActivation && curNode.Op.CheckedDataType == DataTypes.Float16 &&
                curNode.Children.Count == 1 && ((GNNEActivation)curNode.Children[0].Op.Target).InputFromL1.Count > 0 &&
                ((((GNNEActivation)curNode.Children[0].Op.Target).InputFromL1[0] &&
                  curNode.Children[0].Op[GNNEActivation.InputA] == curNode.Op) ||
                 (((GNNEActivation)curNode.Children[0].Op.Target).InputFromL1[1] &&
                  curNode.Children[0].Op[GNNEActivation.InputB] == curNode.Op)))
            {
                _act1 = curNode.Children[0].Op;
            }

            _inputType = _conv[GNNEConv2D.Input].CheckedDataType;
            _weightType = _conv[GNNEConv2D.Weights].CheckedDataType;
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

            _lw = (Call)_conv[GNNEConv2D.Weights];
            _lact = (Call)_conv[GNNEConv2D.Act];
            _lwQarg = (Call)_conv[GNNEConv2D.WeightsBias];
            Call convInput = (Call)_conv[GNNEConv2D.Input];
            _ifmap = sliceInfo[convInput].Ofmap;
            if (_groups == 1)
            {
                _ifmap = new SegmentND(_ifmap[0], new Segment1D(.._weightsShape[1], Padding.Zero()), _ifmap[2],
                    _ifmap[3]);
            }

            _ifmapOffset = sliceInfo[convInput].Nb.OfmapOffset;
            _ofmap = _ni.Ofmap;
            _ofmapOffset = _ni.Nb.OfmapOffset;
            _ofmapConv = _ni.Ofmap;
            _h2C = _weightsShape[1] * _weightsShape[2] <= GNNEEnv.PuHeight && _weightsShape[2] != 1 &&
                   _conv[GNNEConv2D.Input] is Call inputLoadCall && inputLoadCall.Target is GNNELoad && _inputShape[2] > 200 &&
                   _inputShape[3] > 200 && !isDepthwise && _inputType != DataTypes.Int16;
            _memsetValue = (_h2C ? deqBias : 0);
            _fusedKernelH = 1;
            _fusedKernelW = 1;
            _fusedPaddingH = Padding.Zero();
            _fusedPaddingW = Padding.Zero();
            _fusedStrideH = 1;
            _fusedStrideW = 1;
            _fusedDilationH = 1;
            _fusedDilationW = 1;
            if ((object)_pool != null)
            {
                _outputShape = _pool.CheckedShape.ToValueArray();
                _fusedKernelH = ((TensorConst)_pool[GNNEPdp0Reduce.Filter]).Value.ToArray<int>()[0];
                _fusedKernelW = ((TensorConst)_pool[GNNEPdp0Reduce.Filter]).Value.ToArray<int>()[1];
                int[] poolPadding = ((TensorConst)_pool[GNNEPdp0Reduce.Padding]).Value.ToArray<int>();
                _fusedPaddingH = new Padding(poolPadding[0], poolPadding[1]);
                _fusedPaddingW = new Padding(poolPadding[2], poolPadding[3]);
                _fusedStrideH = ((TensorConst)_pool[GNNEPdp0Reduce.Stride]).Value.ToArray<int>()[0];
                _fusedStrideW = ((TensorConst)_pool[GNNEPdp0Reduce.Stride]).Value.ToArray<int>()[1];
                _ofmap = sliceInfo[_pool].Ofmap;
                _ofmapOffset = sliceInfo[_pool].Nb.OfmapOffset;
                _l1FuseNi = sliceInfo[_pool];
                _l1Fused = true;
                _l1FusedInfos[curNode.Op] = L1FusedType.FusedPool;
            }
            else if ((object)_dw != null)
            {
                _outputShape = _dw.CheckedShape.ToValueArray();
                int[] dwWeightsShape = _dw[GNNEPdp0DW.Weights].CheckedShape.ToValueArray();
                _fusedKernelH = dwWeightsShape[2];
                _fusedKernelW = dwWeightsShape[3];
                int[] dwPadding = ((TensorConst)_dw[GNNEPdp0DW.Padding]).Value.ToArray<int>();
                _fusedPaddingH = new Padding(dwPadding[0], dwPadding[1]);
                _fusedPaddingW = new Padding(dwPadding[2], dwPadding[3]);
                _fusedStrideH = ((TensorConst)_dw[GNNEPdp0DW.Stride]).Value.ToArray<int>()[0];
                _fusedStrideW = ((TensorConst)_dw[GNNEPdp0DW.Stride]).Value.ToArray<int>()[1];
                _fusedDilationH = ((TensorConst)_dw[GNNEPdp0DW.Dilation]).Value.ToArray<int>()[0];
                _fusedDilationW = ((TensorConst)_dw[GNNEPdp0DW.Dilation]).Value.ToArray<int>()[1];
                _ofmap = sliceInfo[_dw].Ofmap;
                _ofmapOffset = sliceInfo[_dw].Nb.OfmapOffset;
                _l1FuseNi = sliceInfo[_dw];
                _l1Fused = true;
                _l1FusedInfos[curNode.Op] = L1FusedType.FusedDw;
            }
            else if ((object)_act1 != null)
            {
                bool isSingleInput = _act1[GNNEActivation.InputB] == None.Default;
                _outputShape = _act1.CheckedShape.ToValueArray();
                TileUtilities.Assert(_outputShape.SequenceEqual(_convOutputShape),
                    "_outputShape.SequenceEqual(_convOutputShape)",
                    "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                    3133);
                _ofmap = sliceInfo[_act1].Ofmap;
                _ofmapOffset = sliceInfo[_act1].Nb.OfmapOffset;
                if (!isSingleInput)
                {
                    NodeInfo act1NodeInfo = sliceInfo[_act1];
                    if (_act1[GNNEActivation.InputA] is Call aIsLoad && aIsLoad.Target is GNNELoad)
                    {
                        Call inputALoad = (Call)_act1[GNNEActivation.InputA];
                        if (inputALoad[GNNELoad.Input] is Var && sliceInfo.ContainsKey(inputALoad))
                        {
                            _if2BufIdx = sliceInfo[inputALoad].Nb.OfBufferIndex;
                            _src2ItemName = ItemName.Ifmap2;
                            _ifmap2 = sliceInfo[inputALoad].Ofmap;
                            _ifmap2Offset = sliceInfo[inputALoad].Nb.OfmapOffset;
                            _if2BufIdx = sliceInfo[inputALoad].Nb.OfBufferIndex;
                            _if2Type = inputALoad.CheckedDataType;
                        }
                        else
                        {
                            _lif2 = inputALoad;
                            _ifmap2 = _ofmap;
                            _ifmap2Offset = act1NodeInfo.Nb.Ifmap2Offset;
                            _src2ItemName = ItemName.Ifmap2;
                            _if2Type = inputALoad.CheckedDataType;
                        }
                    }
                    else if (_act1[GNNEActivation.InputB] is Call bIsLoad && bIsLoad.Target is GNNELoad)
                    {
                        Call inputBLoad = (Call)_act1[GNNEActivation.InputB];
                        if (inputBLoad[GNNELoad.Input] is Var && sliceInfo.ContainsKey(inputBLoad))
                        {
                            _if2BufIdx = sliceInfo[inputBLoad].Nb.OfBufferIndex;
                            _src2ItemName = ItemName.Ifmap2;
                            _ifmap2 = sliceInfo[inputBLoad].Ofmap;
                            _ifmap2Offset = sliceInfo[inputBLoad].Nb.OfmapOffset;
                            _if2BufIdx = sliceInfo[inputBLoad].Nb.OfBufferIndex;
                            _if2Type = inputBLoad.CheckedDataType;
                        }
                        else
                        {
                            _lif2 = inputBLoad;
                            _ifmap2 = _ofmap;
                            _ifmap2Offset = act1NodeInfo.Nb.Ifmap2Offset;
                            _src2ItemName = ItemName.Ifmap2;
                            _if2Type = _act1[GNNEActivation.InputB].CheckedDataType;
                        }
                    }
                    else
                    {
                        _src2ItemName = ItemName.Ifmap2;
                        if (((GNNEActivation)_act1.Target).InputFromL1[0])
                        {
                            NodeInfo inputBNodeInfo = sliceInfo[(Call)_act1[GNNEActivation.InputB]];
                            _ifmap2 = inputBNodeInfo.Ofmap;
                            _ifmap2Offset = inputBNodeInfo.Nb.OfmapOffset;
                            _if2BufIdx = inputBNodeInfo.Nb.OfBufferIndex;
                            _if2Type = _act1[GNNEActivation.InputB].CheckedDataType;
                        }
                        else
                        {
                            if (!((GNNEActivation)_act1.Target).InputFromL1[1])
                            {
                                throw new NotSupportedException("not support conv_act1 fuse type!");
                            }

                            NodeInfo inputANodeInfo = sliceInfo[(Call)_act1[GNNEActivation.InputA]];
                            _ifmap2 = inputANodeInfo.Ofmap;
                            _ifmap2Offset = inputANodeInfo.Nb.OfmapOffset;
                            _if2BufIdx = inputANodeInfo.Nb.OfBufferIndex;
                            _if2Type = _act1[GNNEActivation.InputA].CheckedDataType;
                            _swapAB = true;
                        }
                    }
                }

                _l1FuseNi = sliceInfo[_act1];
                _l1Fused = true;
                _l1FusedInfos[curNode.Op] = L1FusedType.FusedAct1;
            }

            _convOutputShape[2] = TileUtilities.GetInputRowSegment(0, _outputShape[2], _convOutputShape[2],
                _fusedKernelH, _fusedStrideH, _fusedDilationH, in _fusedPaddingH).Length;
            _convOutputShape[3] = TileUtilities.GetInputColumnSegment(0, _outputShape[3], _convOutputShape[3],
                _fusedKernelW, _fusedStrideW, _fusedDilationW, in _fusedPaddingW).Length;
            _weightGroup = _weightGroups[_conv];
            _weight = new SegmentND(new Segment1D(.._weightsShape[0], Padding.Zero()),
                new Segment1D(.._weightsShape[1], Padding.Zero()), new Segment1D(.._weightsShape[2], Padding.Zero()),
                new Segment1D(.._weightsShape[3], Padding.Zero()));
            _if1BufIdx = sliceInfo[convInput].Nb.OfBufferIndex;
            _ofBufIdx = (_l1Fused ? _l1FuseNi.Nb.OfBufferIndex : _ni.Nb.OfBufferIndex);
            _weightBufIdx = ((_ni.Nb.WeightPreloadOffset == -1) ? ((_ni.Nb.WeightOffset != 0) ? 1 : 0) : (-1));
        }

        op = curNode.Op;
        if ((object)op != null && op.Target is GNNEActivation)
        {
            _act1 = curNode.Op;
            _lact = (Call)_act1[GNNEActivation.Act];
            _outputShape = _act1.CheckedShape.ToValueArray();
            _outputType = _act1.CheckedDataType;
            _ofmap = _ni.Ofmap;
            _ofmapOffset = _ni.Nb.OfmapOffset;
            if (!(_act1[GNNEActivation.InputB] == None.Default))
            {
                (_ifmapA, _ifmapB) = GetAct1InputsShape(_act1[GNNEActivation.InputA].CheckedShape.ToValueArray(),
                    _act1[GNNEActivation.InputB].CheckedShape.ToValueArray(), _act1.CheckedShape.ToValueArray(),
                    _ofmap);
                if (_act1[GNNEActivation.InputA] is Call loadedA && loadedA.Target is GNNELoad &&
                    _act1[GNNEActivation.InputB] is Call loadedB && loadedB.Target is GNNELoad)
                {
                    Call inputA = _act1[GNNEActivation.InputA] as Call;
                    Call inputB = _act1[GNNEActivation.InputB] as Call;
                    _src2ItemName = ItemName.Ifmap2;
                    _ifmap2Offset = _ni.Nb.Ifmap2Offset;
                    if (sliceInfo.ContainsKey(inputA))
                    {
                        _ifmap = sliceInfo[inputA].Ofmap;
                        _ifmapOffset = sliceInfo[inputA].Nb.OfmapOffset;
                        _lif2 = inputB;
                        _ifmap2 = _ifmapB;
                        _if1BufIdx = sliceInfo[inputA].Nb.OfBufferIndex;
                        _if2BufIdx = -1;
                        _ofBufIdx = _ni.Nb.OfBufferIndex;
                        _if2Type = inputB.CheckedDataType;
                        _inputType = inputB.CheckedDataType;
                        TileUtilities.Assert(_ifmap == _ifmapA, "_ifmap == _ifmapA",
                            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                            3278);
                    }
                    else
                    {
                        TileUtilities.Assert(sliceInfo.ContainsKey(inputB), "sliceInfo.ContainsKey(act1LifB!)",
                            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                            3282);
                        _ifmap = sliceInfo[inputB].Ofmap;
                        _ifmapOffset = sliceInfo[inputB].Nb.OfmapOffset;
                        _lif2 = inputA;
                        _ifmap2 = _ifmapA;
                        _if1BufIdx = sliceInfo[inputB].Nb.OfBufferIndex;
                        _if2BufIdx = -1;
                        _ofBufIdx = _ni.Nb.OfBufferIndex;
                        _if2Type = inputA.CheckedDataType;
                        _inputType = inputB.CheckedDataType;
                        TileUtilities.Assert(_ifmap == _ifmapB, "_ifmap == _ifmapB",
                            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                            3292);
                    }
                }
                else if (_act1[GNNEActivation.InputA] is Call loadedA2 && loadedA2.Target is GNNELoad &&
                         !sliceInfo.ContainsKey((Call)_act1[GNNEActivation.InputA]))
                {
                    Call inputA = _act1[GNNEActivation.InputA] as Call;
                    Call inputB = _act1[GNNEActivation.InputB] as Call;
                    _ifmap = sliceInfo[inputB].Ofmap;
                    _ifmapOffset = sliceInfo[inputB].Nb.OfmapOffset;
                    _ifmap2 = _ifmapA;
                    _src2ItemName = ItemName.Ifmap2;
                    _ifmap2Offset = _ni.Nb.Ifmap2Offset;
                    _lif2 = inputA;
                    _if1BufIdx = sliceInfo[inputB].Nb.OfBufferIndex;
                    _if2BufIdx = -1;
                    _ofBufIdx = _ni.Nb.OfBufferIndex;
                    _if2Type = inputA.CheckedDataType;
                    _inputType = inputB.CheckedDataType;
                    _swapAB = true;
                    TileUtilities.Assert(_ifmap == _ifmapB, "_ifmap == _ifmapB",
                        "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                        3311);
                }
                else if (_act1[GNNEActivation.InputB] is Call loadedB2 && loadedB2.Target is GNNELoad &&
                         !sliceInfo.ContainsKey((Call)_act1[GNNEActivation.InputB]))
                {
                    Call inputB = _act1[GNNEActivation.InputB] as Call;
                    Call inputA = _act1[GNNEActivation.InputA] as Call;
                    _ifmapOffset = sliceInfo[inputA].Nb.OfmapOffset;
                    _ifmap = sliceInfo[inputA].Ofmap;
                    _ifmap2 = _ifmapB;
                    _src2ItemName = ItemName.Ifmap2;
                    _ifmap2Offset = _ni.Nb.Ifmap2Offset;
                    _lif2 = inputB;
                    _if1BufIdx = sliceInfo[inputA].Nb.OfBufferIndex;
                    _if2BufIdx = -1;
                    _ofBufIdx = _ni.Nb.OfBufferIndex;
                    _if2Type = inputB.CheckedDataType;
                    _inputType = inputA.CheckedDataType;
                    TileUtilities.Assert(_ifmap == _ifmapA, "_ifmap == _ifmapA",
                        "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                        3328);
                }
                else
                {
                    Call inputA = _act1[GNNEActivation.InputA] as Call;
                    Call inputB = _act1[GNNEActivation.InputB] as Call;
                    _ifmap = sliceInfo[inputA].Ofmap;
                    _ifmapOffset = sliceInfo[inputA].Nb.OfmapOffset;
                    _ifmap2 = sliceInfo[inputB].Ofmap;
                    _ifmap2Offset = sliceInfo[inputB].Nb.OfmapOffset;
                    _src2ItemName = ItemName.Ifmap2;
                    _if1BufIdx = sliceInfo[inputA].Nb.OfBufferIndex;
                    _if2BufIdx = sliceInfo[inputB].Nb.OfBufferIndex;
                    _ofBufIdx = _ni.Nb.OfBufferIndex;
                    _if2Type = inputB.CheckedDataType;
                    _inputType = inputA.CheckedDataType;
                    TileUtilities.Assert(_ifmap2.Shape_size > 0, "_ifmap2.Shape_size > 0",
                        "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                        3347);
                }
            }
            else
            {
                Call inputA = _act1[GNNEActivation.InputA] as Call;
                _ifmap = sliceInfo[inputA].Ofmap;
                _ifmapOffset = sliceInfo[inputA].Nb.OfmapOffset;
                _if1BufIdx = sliceInfo[inputA].Nb.OfBufferIndex;
                _if2BufIdx = -1;
                _ofBufIdx = _ni.Nb.OfBufferIndex;
                _inputType = inputA.CheckedDataType;
                _if2Type = _inputType;
            }
        }

        op = curNode.Op;
        if ((object)op != null && op.Target is GNNEPdp1)
        {
            _pdp1 = curNode.Op;
            Call input = _pdp1[GNNEPdp1.Input] as Call;
            _ifmap = sliceInfo[input].Ofmap;
            _ifmapOffset = sliceInfo[input].Nb.OfmapOffset;
            _inputType = input.CheckedDataType;
            _outputType = _pdp1.CheckedDataType;
            _ofmap = _ni.Ofmap;
            _ofmapOffset = _ni.Nb.OfmapOffset;
            _inputShape = _pdp1[GNNEPdp1.Input].CheckedShape.ToValueArray();
            _outputShape = _pdp1.CheckedShape.ToValueArray();
            _if1BufIdx = sliceInfo[input].Nb.OfBufferIndex;
            _if2BufIdx = -1;
            _ofBufIdx = _ni.Nb.OfBufferIndex;
            int filterH = ((TensorConst)_pdp1[GNNEPdp1.Filter]).Value.ToArray<int>()[0];
            int filterW = ((TensorConst)_pdp1[GNNEPdp1.Filter]).Value.ToArray<int>()[1];
            if (_inputShape[2] == filterH && _inputShape[3] == filterW && (_inputShape[2] > 16 || _inputShape[3] > 64 ||
                                                                     _inputShape[2] * _inputShape[3] > 256))
            {
                _isGlobalPdp = true;
            }
            else
            {
                _isGlobalPdp = false;
            }
        }

        op = curNode.Op;
        if ((object)op != null && op.Target is GNNETranspose)
        {
            _transpose = curNode.Op;
            Call input = _transpose[GNNETranspose.Input] as Call;
            _ifmap = sliceInfo[input].Ofmap;
            _ifmapOffset = sliceInfo[input].Nb.OfmapOffset;
            _inputType = input.CheckedDataType;
            _outputType = _transpose.CheckedDataType;
            _ofmap = _ni.Ofmap;
            _ofmapOffset = _ni.Nb.OfmapOffset;
            _inputShape = input.CheckedShape.ToValueArray();
            _outputShape = _transpose.CheckedShape.ToValueArray();
            _if1BufIdx = sliceInfo[input].Nb.OfBufferIndex;
            _if2BufIdx = -1;
            _ofBufIdx = _ni.Nb.OfBufferIndex;
        }

        op = curNode.Op;
        if ((object)op != null && op.Target is Concat)
        {
            _cat = curNode.Op;
            _outputType = _cat.CheckedDataType;
            Call catInput = _cat[Concat.Input][1] as Call;
            _ifmap = sliceInfo[catInput].Ofmap;
            _ifmapOffset = sliceInfo[catInput].Nb.OfmapOffset;
            _if1BufIdx = sliceInfo[catInput].Nb.OfBufferIndex;
            catInput = _cat[Concat.Input][0] as Call;
            _ifmap2 = sliceInfo[catInput].Ofmap;
            _ifmap2Offset = sliceInfo[catInput].Nb.OfmapOffset;
            _if2BufIdx = sliceInfo[catInput].Nb.OfBufferIndex;
            _ofmap = _ni.Ofmap;
            _ofmapOffset = _ni.Nb.OfmapOffset;
            _ofBufIdx = _ni.Nb.OfBufferIndex;
            TileUtilities.Assert(_if1BufIdx != _ofBufIdx, "_if1BufIdx != _ofBufIdx",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                3441);
        }

        op = curNode.Op;
        if ((object)op != null && op.Target is Ai2dResize)
        {
            _resize = curNode.Op;
            Call input = _resize[Ai2dResize.Input] as Call;
            _ifmap = sliceInfo[input].Ofmap;
            _ifmapOffset = sliceInfo[input].Nb.OfmapOffset;
            _inputType = input.CheckedDataType;
            _outputType = input.CheckedDataType;
            _ofmap = _ni.Ofmap;
            _ofmapOffset = _ni.Nb.OfmapOffset;
            _inputShape = input.CheckedShape.ToValueArray();
            _outputShape = _resize.CheckedShape.ToValueArray();
            _if1BufIdx = sliceInfo[input].Nb.OfBufferIndex;
            _if2BufIdx = -1;
            _ofBufIdx = _ni.Nb.OfBufferIndex;
        }

        _ofmapSt = _ofmap;
        Call convInputCall = (((object)_conv != null) ? ((Call)_conv[GNNEConv2D.Input]) : null);
        _ifmapLd = (((object)convInputCall != null) ? sliceInfo[convInputCall].Ofmap : _ifmap);
        if (ReshapeConv(_conv, ref _inputShape, ref _outputShape, ref _convOutputShape, ref _weightsShape))
        {
            int channelBlock = ((_conv[GNNEConv2D.Weights].CheckedShape.ToValueArray()[1] % 24 == 0)
                ? 24
                : ((_conv[GNNEConv2D.Weights].CheckedShape.ToValueArray()[1] % 20 == 0) ? 20 : 16));
            _ifmap = new SegmentND(new Segment1D(.._ifmap[0].Length, Padding.Zero()),
                new Segment1D(..channelBlock, Padding.Zero()), new Segment1D(..(_ifmap[1].Length / channelBlock), Padding.Zero()),
                new Segment1D(..(_ifmap[2].Length * _ifmap[3].Length), Padding.Zero()));
            _ofmap = new SegmentND(new Segment1D(.._ofmap[0].Length, Padding.Zero()),
                new Segment1D(.._ofmap[1].Length, Padding.Zero()), new Segment1D(..1, Padding.Zero()),
                new Segment1D(..(_ofmap[2].Length * _ofmap[3].Length), Padding.Zero()));
            _convOutputShape[2] = TileUtilities.GetInputRowSegment(0, _outputShape[2], _convOutputShape[2],
                _fusedKernelH, _fusedStrideH, _fusedDilationH, in _fusedPaddingH).Length;
            _convOutputShape[3] = TileUtilities.GetInputColumnSegment(0, _outputShape[3], _convOutputShape[3],
                _fusedKernelW, _fusedStrideW, _fusedDilationW, in _fusedPaddingW).Length;
            _weight = new SegmentND(new Segment1D(.._weightsShape[0], Padding.Zero()),
                new Segment1D(.._weightsShape[1], Padding.Zero()), new Segment1D(.._weightsShape[2], Padding.Zero()),
                new Segment1D(.._weightsShape[3], Padding.Zero()));
        }

        if (!weightGroupOnly)
        {
            GetGlbLayouts(fusionInfo, curNode, glb, sliceInfo);
        }
    }

    private bool ReshapeConv(Call conv, ref int[] inputShape, ref int[] outputShape, ref int[] convOutputShape,
        ref int[] weightsShape)
    {
        if (Conv1X1(conv))
        {
            TileUtilities.Assert(conv[GNNEConv2D.Weights].CheckedShape.ToValueList()[1] % 24 == 0,
                "conv[GNNEConv2D.Weights].CheckedShape.ToValueList()[1] % 24 == 0",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
                3504);
            int channelBlock = 24;
            while (conv[GNNEConv2D.Weights].CheckedShape.ToValueList()[1] % channelBlock != 0)
            {
                channelBlock--;
            }

            inputShape = new int[4]
            {
                conv[GNNEConv2D.Input].CheckedShape.ToValueList()[0], channelBlock,
                conv[GNNEConv2D.Weights].CheckedShape.ToValueList()[1] / channelBlock,
                conv[GNNEConv2D.Input].CheckedShape.ToValueList()[2] *
                conv[GNNEConv2D.Input].CheckedShape.ToValueList()[3]
            };
            weightsShape = new int[4]
            {
                conv[GNNEConv2D.Weights].CheckedShape.ToValueList()[0], channelBlock,
                conv[GNNEConv2D.Weights].CheckedShape.ToValueList()[1] / channelBlock, 1
            };
            outputShape = new int[4]
            {
                conv.CheckedShape.ToValueList()[0], conv.CheckedShape.ToValueList()[1], 1,
                conv.CheckedShape.ToValueList()[2] * conv.CheckedShape.ToValueList()[3]
            };
            convOutputShape = new int[4]
            {
                conv.CheckedShape.ToValueList()[0], conv.CheckedShape.ToValueList()[1], 1,
                conv.CheckedShape.ToValueList()[2] * conv.CheckedShape.ToValueList()[3]
            };
            if ((object)_dw != null)
            {
                outputShape = new int[4]
                {
                    _dw.CheckedShape.ToValueList()[0], _dw.CheckedShape.ToValueList()[1], 1,
                    _dw.CheckedShape.ToValueList()[2] * _dw.CheckedShape.ToValueList()[3]
                };
            }

            if ((object)_pool != null)
            {
                outputShape = new int[4]
                {
                    _pool.CheckedShape.ToValueList()[0], _pool.CheckedShape.ToValueList()[1], 1,
                    _pool.CheckedShape.ToValueList()[2] * _pool.CheckedShape.ToValueList()[3]
                };
            }

            if ((object)_act1 != null)
            {
                outputShape = new int[4]
                {
                    _act1.CheckedShape.ToValueList()[0], _act1.CheckedShape.ToValueList()[1], 1,
                    _act1.CheckedShape.ToValueList()[2] * _act1.CheckedShape.ToValueList()[3]
                };
            }

            return true;
        }

        return false;
    }

    private void GetGlbLayouts(FusionInfo fusionInfo, NodeInfo curNode, TiledGlb glb,
        Dictionary<Call, NodeInfo> sliceInfo)
    {
        MmuItem mmu = glb.GlbMap[ItemName.Ofmap].Mmu;
        NodeInfo layerNode = curNode;
        Call op = curNode.Op;
        if ((object)op != null && op.Target is GNNEConv2D && (object)_act1 != null)
        {
            layerNode = curNode.Children[0];
        }

        switch (layerNode.Nb.AlignType)
        {
            case AlignedType.FAligned:
                {
                    int ofmapWidth = _ofmapSt[3].Length;
                    int widthAlignment = ((TileUtilities.GetBytesPerElement(_outputType) == 1) ? 32 : 16);
                    int alignedWidth = TileUtilities.GetAlignedNum(ofmapWidth, widthAlignment);
                    glb.GlbMap[ItemName.Ofmap] =
                        new TensorOnGlb(
                            new int[4] { _ofmapSt[0].Length, _ofmapSt[1].Length, _ofmapSt[2].Length, alignedWidth },
                            _outputType, 0, mmu);
                    break;
                }
            case AlignedType.EAligned:
                {
                    int ofmapHeight = _ofmapSt[2].Length;
                    int heightAlignment = ((TileUtilities.GetBytesPerElement(_outputType) == 1) ? 32 : 16);
                    int alignedHeight = TileUtilities.GetAlignedNum(ofmapHeight, heightAlignment);
                    glb.GlbMap[ItemName.Ofmap] =
                        new TensorOnGlb(
                            new int[4] { _ofmapSt[0].Length, _ofmapSt[1].Length, alignedHeight, _ofmapSt[3].Length },
                            _outputType, 0, mmu);
                    break;
                }
            default:
                glb.GlbMap[ItemName.Ofmap] =
                    new TensorOnGlb(
                        new int[4] { _ofmapSt[0].Length, _ofmapSt[1].Length, _ofmapSt[2].Length, _ofmapSt[3].Length },
                        _outputType, 0, mmu);
                break;
        }

        int index = 0;
        op = curNode.Op;
        if ((object)op != null && op.Target is GNNEActivation && curNode.Op[GNNEActivation.InputB] != None.Default &&
            sliceInfo.ContainsKey((Call)curNode.Op[GNNEActivation.InputB]) &&
            !sliceInfo.ContainsKey((Call)curNode.Op[GNNEActivation.InputA]))
        {
            index = 1;
        }

        mmu = glb.GlbMap[ItemName.Ofmap].Mmu;
        op = curNode.Op;
        if ((object)op == null || !(op.Target is GNNELoad))
        {
            if (sliceInfo[(Call)curNode.Op.Arguments[index]].Nb.AlignType == AlignedType.FAligned)
            {
                int ifmapHeight = _ifmapLd[2].Length;
                int ifmapWidth = _ifmapLd[3].Length;
                int ifmapHeightOut = ifmapHeight;
                int ifmapWidthAlignment = ((TileUtilities.GetBytesPerElement(_inputType) == 1) ? 32 : 16);
                int ifmapAlignedWidth = TileUtilities.GetAlignedNum(ifmapWidth, ifmapWidthAlignment);
                glb.GlbMap[ItemName.Ifmap] =
                    new TensorOnGlb(new int[4] { _ifmapLd[0].Length, _ifmapLd[1].Length, ifmapHeightOut, ifmapAlignedWidth }, _inputType,
                        0, mmu);
            }
            else if (sliceInfo[(Call)curNode.Op.Arguments[index]].Nb.AlignType == AlignedType.EAligned)
            {
                int ifmapHeight = _ifmapLd[2].Length;
                int ifmapWidth = _ifmapLd[3].Length;
                int ifmapHeightAlignment = ((TileUtilities.GetBytesPerElement(_inputType) == 1) ? 32 : 16);
                int ifmapAlignedHeight = TileUtilities.GetAlignedNum(ifmapHeight, ifmapHeightAlignment);
                int ifmapWidthOut = ifmapWidth;
                glb.GlbMap[ItemName.Ifmap] =
                    new TensorOnGlb(new int[4] { _ifmapLd[0].Length, _ifmapLd[1].Length, ifmapAlignedHeight, ifmapWidthOut },
                        _inputType, 0, mmu);
            }
            else
            {
                DataType inputType = _inputType;
                glb.GlbMap[ItemName.Ifmap] =
                    new TensorOnGlb(
                        new int[4] { _ifmapLd[0].Length, _ifmapLd[1].Length, _ifmapLd[2].Length, _ifmapLd[3].Length },
                        inputType, 0, mmu);
            }
        }

        MmuItem mmu2 = (fusionInfo.Mmu.ContainsKey(ItemName.Ifmap2)
            ? fusionInfo.Mmu[ItemName.Ifmap2]
            : glb.GlbMap[ItemName.Ifmap2].Mmu);
        op = curNode.Op;
        if ((object)op != null && op.Target is GNNEActivation && curNode.Op[GNNEActivation.InputB] != None.Default)
        {
            index = 1;
            if (sliceInfo.ContainsKey((Call)curNode.Op[GNNEActivation.InputB]) &&
                !sliceInfo.ContainsKey((Call)curNode.Op[GNNEActivation.InputA]))
            {
                index = 0;
            }

            int if2Height = _ifmap2[2].Length;
            int if2Width = _ifmap2[3].Length;
            int if2Alignment = 32 / TileUtilities.GetBytesPerElement(_if2Type);
            int if2OutH;
            int if2OutW;
            if (!sliceInfo.ContainsKey((Call)curNode.Op.Arguments[index]))
            {
                if2OutH = if2Height;
                if2OutW = if2Width;
            }
            else
            {
                mmu2 = glb.GlbMap[ItemName.Ofmap].Mmu;
                switch (sliceInfo[(Call)curNode.Op.Arguments[index]].Nb.AlignType)
                {
                    case AlignedType.FAligned:
                        if2OutH = if2Height;
                        if2OutW = TileUtilities.GetAlignedNum(if2Width, if2Alignment);
                        break;
                    case AlignedType.EAligned:
                        if2OutH = TileUtilities.GetAlignedNum(if2Height, if2Alignment);
                        if2OutW = if2Width;
                        break;
                    default:
                        if2OutH = if2Height;
                        if2OutW = if2Width;
                        break;
                }
            }

            glb.GlbMap[ItemName.Ifmap2] =
                new TensorOnGlb(new int[4] { _ifmap2[0].Length, _ifmap2[1].Length, if2OutH, if2OutW }, _if2Type, 0, mmu2);
        }
        else
        {
            glb.GlbMap[ItemName.Ifmap2] =
                new TensorOnGlb(
                    new int[4] { _ifmap2[0].Length, _ifmap2[1].Length, _ifmap2[2].Length, _ifmap2[3].Length }, _if2Type,
                    0, mmu2);
        }

        op = curNode.Op;
        if ((object)op != null && op.Target is GNNEConv2D && (object)_act1 != null &&
            _act1[GNNEActivation.InputB] != None.Default)
        {
            index = (((GNNEActivation)_act1.Target).InputFromL1[0] ? 1 : 0);
            int if2Height = _ifmap2[2].Length;
            int if2Width = _ifmap2[3].Length;
            int if2Alignment = 32 / TileUtilities.GetBytesPerElement(_if2Type);
            int if2OutH;
            int if2OutW;
            if (!sliceInfo.ContainsKey((Call)_act1.Arguments[index]))
            {
                if2OutH = TileUtilities.GetAlignedNum(if2Height, if2Alignment);
                if2OutW = if2Width;
            }
            else
            {
                mmu2 = glb.GlbMap[ItemName.Ofmap].Mmu;
                switch (sliceInfo[(Call)_act1.Arguments[index]].Nb.AlignType)
                {
                    case AlignedType.FAligned:
                        if2OutH = if2Height;
                        if2OutW = TileUtilities.GetAlignedNum(if2Width, if2Alignment);
                        break;
                    case AlignedType.EAligned:
                        if2OutH = TileUtilities.GetAlignedNum(if2Height, if2Alignment);
                        if2OutW = if2Width;
                        break;
                    default:
                        if2OutH = if2Height;
                        if2OutW = if2Width;
                        break;
                }
            }

            glb.GlbMap[ItemName.Ifmap2] =
                new TensorOnGlb(new int[4] { _ifmap2[0].Length, _ifmap2[1].Length, if2OutH, if2OutW }, _if2Type, 0, mmu2);
        }

        if (_h2C)
        {
            if ((object)_lif != null)
            {
                int[] convPadding = ((TensorConst)curNode.Children[0].Op[GNNEConv2D.Padding]).Value.ToArray<int>();
                MmuItem mmu3 = glb.GlbMap[ItemName.Ofmap].Mmu;
                glb.GlbMap[ItemName.Ofmap] =
                    new TensorOnGlb(
                        new int[4]
                        {
                            _ofmapSt[0].Length, _ofmapSt[1].Length, _ofmapSt[2].Length + convPadding[0] + convPadding[1],
                            _ofmapSt[3].Length
                        }, _outputType, 0, mmu3);
                _ofmapSt[2].Padding = _ofmap[2].Padding;
                _ofmapSt[3].Padding = _ofmap[3].Padding;
                glb.GlbMap[ItemName.Ifmap] = glb.GlbMap[ItemName.Ofmap];
            }
            else
            {
                int[] convPadding = ((TensorConst)_conv[GNNEConv2D.Padding]).Value.ToArray<int>();
                MmuItem mmu4 = glb.GlbMap[ItemName.Ifmap].Mmu;
                glb.GlbMap[ItemName.Ifmap] =
                    new TensorOnGlb(
                        new int[4]
                        {
                            _ifmapLd[0].Length, _ifmapLd[1].Length, _ifmapLd[2].Length + convPadding[0] + convPadding[1],
                            _ifmapLd[3].Length
                        }, _inputType, 0, mmu4);
                _ifmapLd[2].Padding = _ifmap[2].Padding;
                _ifmapLd[3].Padding = _ifmap[3].Padding;
            }
        }
    }

    private void ItemRecStatusUpdate()
    {
        if ((object)_conv != null)
        {
            if (!_nodesWeightRec.ContainsKey(_conv))
            {
                _nodesWeightRec.Add(_conv,
                    new List<List<Tuple<SegmentND, TensorStat>>>
                    {
                        new List<Tuple<SegmentND, TensorStat>>(), new List<Tuple<SegmentND, TensorStat>>()
                    });
            }

            if (!_nodesOfmapRec.ContainsKey(_conv))
            {
                _nodesOfmapRec.Add(_conv,
                    new List<List<Tuple<SegmentND, TensorStat>>>
                    {
                        new List<Tuple<SegmentND, TensorStat>>(),
                        new List<Tuple<SegmentND, TensorStat>>(),
                        new List<Tuple<SegmentND, TensorStat>>()
                    });
            }

            if (!_nodesG2LIfRec.ContainsKey(_conv))
            {
                _nodesG2LIfRec.Add(_conv,
                    new List<List<Tuple<SegmentND, TensorStat>>>
                    {
                        new List<Tuple<SegmentND, TensorStat>>(),
                        new List<Tuple<SegmentND, TensorStat>>(),
                        new List<Tuple<SegmentND, TensorStat>>()
                    });
            }

            if (!_nodesG2RWRec.ContainsKey(_conv))
            {
                _nodesG2RWRec.Add(_conv,
                    new List<List<Tuple<SegmentND, TensorStat>>>
                    {
                        new List<Tuple<SegmentND, TensorStat>>(), new List<Tuple<SegmentND, TensorStat>>()
                    });
            }

            if (!_nodesL2GOfRec.ContainsKey(_conv))
            {
                _nodesL2GOfRec.Add(_conv,
                    new List<List<Tuple<SegmentND, TensorStat>>>
                    {
                        new List<Tuple<SegmentND, TensorStat>>(),
                        new List<Tuple<SegmentND, TensorStat>>(),
                        new List<Tuple<SegmentND, TensorStat>>()
                    });
            }

            if (!_nodesL2RIf2Rec.ContainsKey(_conv))
            {
                _nodesL2RIf2Rec.Add(_conv,
                    new List<List<Tuple<SegmentND, TensorStat>>>
                    {
                        new List<Tuple<SegmentND, TensorStat>>(),
                        new List<Tuple<SegmentND, TensorStat>>(),
                        new List<Tuple<SegmentND, TensorStat>>()
                    });
            }

            if (!_nodesG2RWSliceRec.ContainsKey(_conv))
            {
                _nodesG2RWSliceRec.Add(_conv,
                    new List<List<Tuple<SegmentND, TensorStat>>>
                    {
                        new List<Tuple<SegmentND, TensorStat>>(), new List<Tuple<SegmentND, TensorStat>>()
                    });
            }

            if (_weightBufIdx != -1)
            {
                _nodesQuenesAsW[_weightBufIdx].Add(new Tuple<Call, int>(_conv, 0));
            }
        }

        if ((object)_resize != null)
        {
            if (!_nodesOfmapRec.ContainsKey(_resize))
            {
                _nodesOfmapRec.Add(_resize, new List<List<Tuple<SegmentND, TensorStat>>>());
            }

            if (_nodesAi2dIfRec.ContainsKey(_resize))
            {
                _nodesAi2dIfRec.Add(_resize, new List<List<Tuple<SegmentND, TensorStat>>>());
            }

            if (_nodesAi2dOfRec.ContainsKey(_resize))
            {
                _nodesAi2dOfRec.Add(_resize, new List<List<Tuple<SegmentND, TensorStat>>>());
            }
        }
    }

    private void UpdateCcrRecStat()
    {
        foreach (List<Tuple<SegmentND, TensorStat>> recs in _nodesG2LIfRec.Values.SelectMany((
                     List<List<Tuple<SegmentND, TensorStat>>> iter) => iter))
        {
            for (int idx = 0; idx < recs.Count; idx++)
            {
                if (idx == 0)
                {
                    recs[idx].Item2.IsFirstSlice = true;
                }

                if (idx == recs.Count - 1)
                {
                    recs[idx].Item2.IsLastSlice = true;
                }

                if (idx < recs.Count - 1 && recs[idx].Item1 != recs[idx + 1].Item1)
                {
                    recs[idx].Item2.IsLastSlice = true;
                    recs[idx + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        foreach (List<Tuple<SegmentND, TensorStat>> recs in _nodesG2RWRec.Values.SelectMany((
                     List<List<Tuple<SegmentND, TensorStat>>> iter) => iter))
        {
            for (int idx = 0; idx < recs.Count; idx++)
            {
                if (idx == 0)
                {
                    recs[idx].Item2.IsFirstSlice = true;
                }

                if (idx == recs.Count - 1)
                {
                    recs[idx].Item2.IsLastSlice = true;
                }

                if (idx < recs.Count - 1 && recs[idx].Item1 != recs[idx + 1].Item1)
                {
                    recs[idx].Item2.IsLastSlice = true;
                    recs[idx + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        foreach (List<Tuple<SegmentND, TensorStat>> recs in _nodesL2GOfRec.Values.SelectMany((
                     List<List<Tuple<SegmentND, TensorStat>>> iter) => iter))
        {
            for (int idx = 0; idx < recs.Count; idx++)
            {
                if (idx == 0)
                {
                    recs[idx].Item2.IsFirstSlice = true;
                }

                if (idx == recs.Count - 1)
                {
                    recs[idx].Item2.IsLastSlice = true;
                }

                if (idx < recs.Count - 1 && recs[idx].Item1 != recs[idx + 1].Item1)
                {
                    recs[idx].Item2.IsLastSlice = true;
                    recs[idx + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        foreach (List<Tuple<SegmentND, TensorStat>> recs in _nodesL2RIf2Rec.Values.SelectMany((
                     List<List<Tuple<SegmentND, TensorStat>>> iter) => iter))
        {
            for (int idx = 0; idx < recs.Count; idx++)
            {
                if (idx == 0)
                {
                    recs[idx].Item2.IsFirstSlice = true;
                }

                if (idx == recs.Count - 1)
                {
                    recs[idx].Item2.IsLastSlice = true;
                }

                if (idx < recs.Count - 1 && recs[idx].Item1 != recs[idx + 1].Item1)
                {
                    recs[idx].Item2.IsLastSlice = true;
                    recs[idx + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        foreach (List<Tuple<SegmentND, TensorStat>> recs in
                 _nodesWeightRec.Values.SelectMany((List<List<Tuple<SegmentND, TensorStat>>> iter) =>
                     iter.Where((List<Tuple<SegmentND, TensorStat>> t) => t.Count > 0)))
        {
            recs[0].Item2.IsFirstSlice = true;
            recs[recs.Count - 1].Item2.IsLastSlice = true;
        }

        foreach (List<Tuple<SegmentND, TensorStat>> recs in from iter in _nodesOfmapRec.Values
                 from t in iter
                 where t.Count > 0
                 select t)
        {
            recs[0].Item2.IsFirstSlice = true;
            recs[recs.Count - 1].Item2.IsLastSlice = true;
        }

        foreach (KeyValuePair<Call, List<List<Tuple<SegmentND, TensorStat>>>> sliceRec in _nodesG2RWSliceRec)
        {
            List<List<Tuple<SegmentND, TensorStat>>> g2rwRec = _nodesG2RWRec[sliceRec.Key];
            for (int dimIdx = 0; dimIdx < sliceRec.Value.Count; dimIdx++)
            {
                List<SegmentND> uniqueSlices = new List<SegmentND>();
                List<SegmentND> allSlices = new List<SegmentND>();
                List<Tuple<SegmentND, TensorStat>> sliceRecs = new List<Tuple<SegmentND, TensorStat>>();
                for (int recIdx = 0; recIdx < g2rwRec[dimIdx].Count; recIdx++)
                {
                    if (g2rwRec[dimIdx][recIdx].Item2.IsFirstSlice)
                    {
                        uniqueSlices.Clear();
                        allSlices.Clear();
                        sliceRecs.Clear();
                    }

                    if (!uniqueSlices.Contains(sliceRec.Value[dimIdx][recIdx].Item1))
                    {
                        uniqueSlices.Add(sliceRec.Value[dimIdx][recIdx].Item1);
                    }

                    allSlices.Add(sliceRec.Value[dimIdx][recIdx].Item1);
                    sliceRecs.Add(sliceRec.Value[dimIdx][recIdx]);
                    if (!g2rwRec[dimIdx][recIdx].Item2.IsLastSlice)
                    {
                        continue;
                    }

                    foreach (SegmentND uniqueSlice in uniqueSlices)
                    {
                        sliceRecs[allSlices.IndexOf(uniqueSlice)].Item2.IsFirstSlice = true;
                        sliceRecs[sliceRecs.Count - 1 - allSlices.IndexOf(uniqueSlice)].Item2.IsLastSlice = true;
                    }

                    for (int entryIdx = 0; entryIdx < allSlices.Count; entryIdx++)
                    {
                        sliceRecs[entryIdx].Item2.SliceIdx = uniqueSlices.IndexOf(allSlices[entryIdx]);
                        sliceRec.Value[dimIdx][recIdx - allSlices.Count + 1 + entryIdx] = sliceRecs[entryIdx];
                    }
                }

                Call conv = sliceRec.Key;
                if ((object)conv == null || !(conv.Target is GNNEConv2D) ||
                    conv[GNNEConv2D.Weights].CheckedDataType != DataTypes.Int16)
                {
                    continue;
                }

                for (int revIdx = 1; revIdx < sliceRec.Value[dimIdx].Count; revIdx++)
                {
                    if (sliceRec.Value[dimIdx][sliceRec.Value[dimIdx].Count - revIdx - 1].Item2.IsFirstSlice)
                    {
                        sliceRec.Value[dimIdx][sliceRec.Value[dimIdx].Count - revIdx].Item2.IsFirstSlice = true;
                    }

                    if (sliceRec.Value[dimIdx][revIdx].Item2.IsLastSlice)
                    {
                        sliceRec.Value[dimIdx][revIdx - 1].Item2.IsLastSlice = true;
                    }
                }

                for (int sliceIdx = 0; sliceIdx < sliceRec.Value[dimIdx].Count; sliceIdx++)
                {
                    sliceRec.Value[dimIdx][sliceIdx].Item2.SliceIdx = sliceRec.Value[dimIdx][sliceIdx].Item2.SliceIdx * 2 + (sliceIdx & 1);
                }
            }
        }

        foreach (List<Tuple<Call, int>> queue in _nodesQuenesAsW)
        {
            for (int queueIdx = 0; queueIdx < queue.Count; queueIdx++)
            {
                queue[queueIdx] = new Tuple<Call, int>(queue[queueIdx].Item1, queueIdx);
            }
        }
    }

    private void UpdateAi2dCcrRecStat()
    {
        foreach (List<Tuple<SegmentND, TensorStat>> recs in _nodesAi2dIfRec.Values.SelectMany((
                     List<List<Tuple<SegmentND, TensorStat>>> iter) => iter))
        {
            for (int idx = 0; idx < recs.Count; idx++)
            {
                if (idx == 0)
                {
                    recs[idx].Item2.IsFirstSlice = true;
                }

                if (idx == recs.Count - 1)
                {
                    recs[idx].Item2.IsLastSlice = true;
                }

                if (idx < recs.Count - 1 && !(recs[idx].Item1 == recs[idx + 1].Item1))
                {
                    recs[idx].Item2.IsLastSlice = true;
                    recs[idx + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        foreach (List<Tuple<SegmentND, TensorStat>> recs in _nodesAi2dOfRec.Values.SelectMany((
                     List<List<Tuple<SegmentND, TensorStat>>> iter) => iter))
        {
            for (int idx = 0; idx < recs.Count; idx++)
            {
                if (idx == 0)
                {
                    recs[idx].Item2.IsFirstSlice = true;
                }

                if (idx == recs.Count - 1)
                {
                    recs[idx].Item2.IsLastSlice = true;
                }

                if (idx < recs.Count - 1 && !(recs[idx].Item1 == recs[idx + 1].Item1))
                {
                    recs[idx].Item2.IsLastSlice = true;
                    recs[idx + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        foreach (List<Tuple<SegmentND, TensorStat>> recs in from iter in _nodesOfmapRec.Values
                 from t in iter
                 where t.Count > 0
                 select t)
        {
            recs[0].Item2.IsFirstSlice = true;
            recs[recs.Count - 1].Item2.IsLastSlice = true;
        }
    }

    private Tuple<List<CcrSet>, List<CcrClr>> GetCcrSetAndClrVec(NodeInfo currNode)
    {
        List<CcrSet> ccrsToSet = new List<CcrSet>();
        List<CcrClr> ccrsToClr = new List<CcrClr>();
        if (!GNNEEnv.UseCcr)
        {
            return new Tuple<List<CcrSet>, List<CcrClr>>(ccrsToSet, ccrsToClr);
        }

        Call op = currNode.Op;
        TileUtilities.Assert((object)op == null || !(op.Target is GNNEConv2D),
            "currNode.Op is not { Target: GNNEConv2D }",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLayerGroup.cs",
            4100);
        bool isStore = (object)op != null && op.Target is GNNEStore;
        bool isLoad = (object)op != null && op.Target is GNNELoad;
        if (isStore)
        {
            if (_nodesQueNeedClearFake.Count > 0)
            {
                ccrsToSet.Add(new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.OfmapFake, _ofBufIdx)), 1));
            }
        }
        else
        {
            int ccrSetAccordingPostNodes = GetCcrSetAccordingPostNodes(currNode);
            if (ccrSetAccordingPostNodes != 0)
            {
                ccrsToSet.Add(new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, _ofBufIdx)),
                    ccrSetAccordingPostNodes));
            }
        }

        if (!isLoad && !isStore)
        {
            if (_if1BufIdx != -1)
            {
                ccrsToClr.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, _if1BufIdx))));
            }

            if (_if2BufIdx != -1 && ((object)op == null || !(op.Target is Concat)))
            {
                ccrsToClr.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, _if2BufIdx))));
            }
        }
        else if (isStore)
        {
            ccrsToClr.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, _if1BufIdx))));
        }

        return new Tuple<List<CcrSet>, List<CcrClr>>(ccrsToSet, ccrsToClr);
    }
}
