// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.K230;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.K230;
using Nncase.Runtime.K230;
using Nncase.TIR;
using Nncase.TIR.Instructions;

namespace Nncase.Passes.Rules.Tile;

internal sealed class FusionConvertVisitor
{
    public record GlbItemSize
    {
        public int BasementSize { get; set; }

        public int MaxOfmapSize { get; set; }

        public int MaxWeightSize { get; set; }

        public int MaxActSize { get; set; }

        public int MaxWeightQargSize { get; set; }

        public int MaxDwWeightSize { get; set; }

        public int MaxDwActSize { get; set; }

        public int MaxDwWeightQargSize { get; set; }

        public int MaxIfmap2Size { get; set; }

        public int MaxAct1Size { get; set; }

        public int MaxPdp0ActSize { get; set; }

        public int MaxWeightPreloadSize { get; set; }

        public int SumSize => BasementSize + MaxOfmapSize + MaxWeightSize + MaxActSize + MaxWeightQargSize +
                              MaxDwWeightSize + MaxDwActSize + MaxDwWeightQargSize + MaxIfmap2Size + MaxAct1Size +
                              MaxPdp0ActSize + MaxWeightPreloadSize;
    }

    private readonly TileOptions _tileOptions;

    private Fusion? _currentFusion;

    private int _ofBufNum = 2;

    private bool _containNotSupportedOp;

    private RunPassContext _passOptions;

    public RunPassContext PassOptions
    {
        get { return _passOptions; }
        set { _passOptions = value; }
    }

    public Fusion CurrentFusion => _currentFusion;

    public FusionConvertVisitor(RunPassContext passOptions, TileOptions tileOptions)
    {
        PassOptions = passOptions;
        _tileOptions = tileOptions;
    }

    public bool TryL1Fuse(Fusion fusion)
    {
        if ((object)_currentFusion != null)
        {
            throw new InvalidOperationException("Can't Visit More Than One Fusion!");
        }

        _currentFusion = fusion;
        List<NodeInfo> list = new List<NodeInfo>();
        Visit(fusion.Body, list);
        if (_containNotSupportedOp)
        {
            return false;
        }

        if (list.Count == 4)
        {
            Call op = list[1].Op;
            if ((object)op != null && op.Target is GNNEPdp0DW or GNNEPdp0Reduce)
            {
                op = list[2].Op;
                if ((object)op != null && op.Target is GNNEConv2D)
                {
                    return true;
                }
            }
        }

        if (list.Count == 4)
        {
            Call op = list[1].Op;
            if ((object)op != null && op.Target is GNNEActivation)
            {
                op = list[2].Op;
                if ((object)op != null && op.Target is GNNEConv2D)
                {
                    if (!((GNNEActivation)list[1].Op.Target).InputFromL1[0] ||
                        (object)list[1].Op[GNNEActivation.InputA] != list[2].Op)
                    {
                        if (((GNNEActivation)list[1].Op.Target).InputFromL1[1])
                        {
                            return (object)list[1].Op[GNNEActivation.InputB] == list[2].Op;
                        }

                        return false;
                    }

                    return true;
                }
            }
        }

        return false;
    }

    public AllocateResult TryL2Fuse(Fusion fusion, FusionType fusionType)
    {
        if ((object)_currentFusion == null)
        {
            _currentFusion = fusion;
            List<NodeInfo> list = new List<NodeInfo>();
            _containNotSupportedOp = false;
            Visit(fusion.Body, list);
            if (_containNotSupportedOp)
            {
                return new AllocateResult { IsOk = false };
            }

            List<NodeInfo> filteredNodes = new List<NodeInfo>();
            for (int i = 0; i < list.Count; i++)
            {
                Call op = list[i].Op;
                if ((object)op != null && op.Target is Reshape &&
                    (!(op[Reshape.Input] is Call call) || !(call.Target is GNNEStore) || list[i].Children.Count > 0) &&
                    !(op[Reshape.Input] is Var))
                {
                    return new AllocateResult { IsOk = false };
                }

                if (((object)op == null || !(op.Target is Reshape)) && ((object)op == null || !(op.Target is Concat)))
                {
                    filteredNodes.Add(list[i]);
                }
            }

            list = filteredNodes;
            foreach (NodeInfo item in list.Where(delegate(NodeInfo ni)
                     {
                         Call op2 = ni.Op;
                         return (object)op2 != null && op2.Target is GNNEActivation;
                     }))
            {
                Expr expr = item.Op[GNNEActivation.InputA];
                if (expr == list[^1].Op)
                {
                    if (!list[^1].Children.Contains(item))
                    {
                        list[^1].Children.Add(item);
                    }
                }

                if (!(item.Op[GNNEActivation.InputB] != None.Default))
                {
                    continue;
                }

                Expr expr2 = item.Op[GNNEActivation.InputB];
                if (expr2 == list[^1].Op)
                {
                    if (!list[^1].Children.Contains(item))
                    {
                        list[^1].Children.Add(item);
                    }
                }
            }

            UpdateAlignType(list);
            _ofBufNum = list.Exists((NodeInfo ni) => ni.Children.Count > 1)
                ? 3
                : ((list.Count(delegate(NodeInfo ni)
                {
                    Call op2 = ni.Op;
                    return (object)op2 != null && op2.Target is GNNELoad;
                }) > 1)
                    ? 3
                    : 2);
            List<int[]> tileShapes = new List<int[]>();
            int[] bodyShape = fusion.Body.CheckedShape.ToValueArray();
            if (fusion.Body is Call call2 && call2.Target is Reshape)
            {
                bodyShape = call2[Reshape.Input].CheckedShape.ToValueArray();
            }

            int tileH = Math.Min(GNNEEnv.MultiLayerTilingH, bodyShape[2]);
            int tileW = Math.Min(GNNEEnv.MultiLayerTilingW, bodyShape[3]);
            if (fusionType == FusionType.L2IfFullWPp)
            {
                tileH = bodyShape[2];
                tileW = bodyShape[3];
            }

            AllocateResult allocateResult = new AllocateResult { IsOk = false };
            if ((tileH == 0 || tileW == 0) &&
                (fusionType == FusionType.L2IfSplitWPp || fusionType == FusionType.L2IfSplitWFull))
            {
                double usableGlbBytes = Math.Floor(0.9 * (double)GNNEEnv.GlbSize);
                if ((double)(bodyShape[0] * bodyShape[1] * bodyShape[2] * bodyShape[3] * _ofBufNum *
                             TileUtilities.GetBytesPerElement(list[0].Op[GNNEStore.Input].CheckedDataType)) < usableGlbBytes)
                {
                    return allocateResult;
                }

                int splitsH = 1;
                int splitsW = 1;
                while (splitsH < bodyShape[2] || splitsW < bodyShape[3])
                {
                    if ((splitsH <= splitsW && splitsH < bodyShape[2]) || splitsW == bodyShape[3])
                    {
                        splitsH++;
                    }
                    else
                    {
                        splitsW++;
                    }

                    tileH = (int)Math.Ceiling(1f * (float)bodyShape[2] / (float)splitsH);
                    tileW = (int)Math.Ceiling(1f * (float)bodyShape[3] / (float)splitsW);
                    tileShapes.Add(new int[4] { bodyShape[0], bodyShape[1], tileH, tileW });
                }
            }
            else
            {
                tileShapes.Add(new int[4] { bodyShape[0], bodyShape[1], tileH, tileW });
            }

            for (int tileIndex = 0; tileIndex < tileShapes.Count && tileIndex < 3; tileIndex++)
            {
                int[] tileOutShape = tileShapes[tileIndex];
                List<BoxOnGlb> boxes = new List<BoxOnGlb>();
                GlbItemSize glbItemSize = new GlbItemSize();
                glbItemSize.BasementSize = TileUtilities.GetAlignedNum(
                    SpaceSearcher.GetBasementSize(),
                    GNNEEnv.GlbWidth * GNNEEnv.GlbBankWidth);
                List<NodeInfo> nodeInfo = list;
                Visit(fusionType, tileOutShape, glbItemSize, list);
                boxes.Add(new BoxOnGlb(
                    new int[2] { GNNEEnv.GlbWidth, glbItemSize.BasementSize / GNNEEnv.GlbWidth / GNNEEnv.GlbBankWidth },
                    ItemName.Basement));
                boxes.Add(new BoxOnGlb(
                    new int[2]
                    {
                        GNNEEnv.OfmapBankWidth,
                        glbItemSize.MaxOfmapSize / GNNEEnv.OfmapBankWidth / GNNEEnv.GlbBankWidth,
                    }, ItemName.Ofmap));
                boxes.Add(new BoxOnGlb(
                    new int[2]
                    {
                        GNNEEnv.WBankWidth, glbItemSize.MaxWeightSize / GNNEEnv.WBankWidth / GNNEEnv.GlbBankWidth,
                    }, ItemName.Weight));
                boxes.Add(new BoxOnGlb(
                    new int[2]
                    {
                        GNNEEnv.ActBankWidth, glbItemSize.MaxActSize / GNNEEnv.ActBankWidth / GNNEEnv.GlbBankWidth,
                    }, ItemName.Act));
                if (glbItemSize.MaxDwWeightSize > 0)
                {
                    boxes.Add(new BoxOnGlb(
                        new int[2]
                        {
                            GNNEEnv.WBankWidth,
                            glbItemSize.MaxDwWeightSize / GNNEEnv.WBankWidth / GNNEEnv.GlbBankWidth,
                        }, ItemName.DwWeight));
                }

                if (glbItemSize.MaxDwActSize > 0)
                {
                    boxes.Add(new BoxOnGlb(
                        new int[2]
                        {
                            GNNEEnv.ActBankWidth,
                            glbItemSize.MaxDwActSize / GNNEEnv.ActBankWidth / GNNEEnv.GlbBankWidth,
                        }, ItemName.DwAct1));
                }

                if (glbItemSize.MaxWeightQargSize > 0)
                {
                    boxes.Add(new BoxOnGlb(
                        new int[2]
                        {
                            GNNEEnv.WQargBankWidth,
                            glbItemSize.MaxWeightQargSize / GNNEEnv.WQargBankWidth / GNNEEnv.GlbBankWidth,
                        }, ItemName.WQarg));
                }

                if (glbItemSize.MaxDwWeightQargSize > 0)
                {
                    boxes.Add(new BoxOnGlb(
                        new int[2]
                        {
                            GNNEEnv.WQargBankWidth,
                            glbItemSize.MaxDwWeightQargSize / GNNEEnv.WQargBankWidth / GNNEEnv.GlbBankWidth,
                        }, ItemName.DwQarg));
                }

                if (glbItemSize.MaxPdp0ActSize > 0)
                {
                    boxes.Add(new BoxOnGlb(
                        new int[2]
                        {
                            GNNEEnv.ActBankWidth,
                            glbItemSize.MaxPdp0ActSize / GNNEEnv.ActBankWidth / GNNEEnv.GlbBankWidth,
                        }, ItemName.PdpAct1));
                }

                if (glbItemSize.MaxIfmap2Size > 0)
                {
                    boxes.Add(new BoxOnGlb(
                        new int[2]
                        {
                            GNNEEnv.IfmapBankWidth,
                            glbItemSize.MaxIfmap2Size / GNNEEnv.IfmapBankWidth / GNNEEnv.GlbBankWidth,
                        }, ItemName.Ifmap2));
                }

                if (glbItemSize.MaxAct1Size > 0)
                {
                    boxes.Add(new BoxOnGlb(
                        new int[2]
                        {
                            GNNEEnv.ActBankWidth,
                            glbItemSize.MaxAct1Size / GNNEEnv.ActBankWidth / GNNEEnv.GlbBankWidth,
                        }, ItemName.MfuAct1));
                }

                if (glbItemSize.MaxWeightPreloadSize > 0)
                {
                    boxes.Add(new BoxOnGlb(
                        new int[2]
                        {
                            GNNEEnv.WBankWidth,
                            glbItemSize.MaxWeightPreloadSize / GNNEEnv.WBankWidth / GNNEEnv.GlbBankWidth,
                        }, ItemName.WeightPreload));
                }

                allocateResult = SpaceSearcher.TryAllocate(new BoxPacker(16) { Boxes = boxes });
                allocateResult.LastOutShape = tileShapes[tileIndex];
                int value = (fusionType == FusionType.L2IfFullWPp || fusionType == FusionType.L2IfSplitWPp)
                    ? glbItemSize.MaxWeightSize
                    : 0;
                allocateResult.BufferSize = new Dictionary<ItemName, int>
                {
                    { ItemName.Ofmap, glbItemSize.MaxOfmapSize / _ofBufNum }, { ItemName.Weight, value },
                };
                allocateResult.NodeInfo = nodeInfo;
                if (fusionType == FusionType.L2IfSplitWPp || fusionType == FusionType.L2IfSplitWFull)
                {
                    if (glbItemSize.MaxWeightSize > GNNEEnv.GlbSize)
                    {
                        allocateResult.IsOk = false;
                        break;
                    }

                    if (allocateResult.IsOk)
                    {
                        break;
                    }
                }
            }

            if (allocateResult.IsOk)
            {
                GetFusionInfo(fusion, allocateResult);
            }

            return allocateResult;
        }

        throw new InvalidOperationException("Can't Visit More Than One Fusion!");
    }

    public void Visit(Expr expr, List<NodeInfo> nodeInfos)
    {
        if (expr is Call)
        {
            Call call = expr as Call;
            Expr target = call.Target;
            if (!(target is GNNELoad))
            {
                if (!(target is GNNEStore))
                {
                    if (!(target is GNNEConv2D))
                    {
                        if (!(target is GNNEPdp0DW))
                        {
                            if (!(target is GNNEPdp0Reduce))
                            {
                                if (!(target is Uninitialized))
                                {
                                    if (!(target is GNNEPdp1))
                                    {
                                        if (!(target is GNNETranspose))
                                        {
                                            if (!(target is GNNEActivation))
                                            {
                                                if (!(target is Reshape))
                                                {
                                                    if (target is Concat)
                                                    {
                                                        if (((Nncase.IR.Tuple)call[Concat.Input]).Fields.ToArray()
                                                            .Any((Expr i) => !(i is Var)))
                                                        {
                                                            _containNotSupportedOp = true;
                                                        }
                                                    }
                                                    else
                                                    {
                                                        _containNotSupportedOp = true;
                                                    }
                                                }
                                                else
                                                {
                                                    LowerReshape(call, nodeInfos);
                                                }
                                            }
                                            else if (call[GNNEActivation.InputB] is Call call2 &&
                                                     call2.Target is GNNELoad && call2[GNNELoad.Input] is Call call3 &&
                                                     call3.Target is Reshape)
                                            {
                                                _containNotSupportedOp = true;
                                            }
                                            else if (call[GNNEActivation.InputB] != None.Default &&
                                                     (!(call[GNNEActivation.InputB] is Call call4) ||
                                                      !(call4.Target is GNNELoad)) &&
                                                     call[GNNEActivation.InputA] is Call call5 &&
                                                     call5.Target is GNNELoad && call5[GNNELoad.Input] is Call call6 &&
                                                     call6.Target is Reshape)
                                            {
                                                _containNotSupportedOp = true;
                                            }
                                            else
                                            {
                                                LowerGnneActivation(call, nodeInfos);
                                            }
                                        }
                                        else
                                        {
                                            LowerGnneTranspose(call, nodeInfos);
                                        }
                                    }
                                    else
                                    {
                                        LowerGnnePdp1(call, nodeInfos);
                                    }
                                }
                                else
                                {
                                    LowerUninitialized(call, nodeInfos);
                                }
                            }
                            else
                            {
                                LowerGnnePdp0Reduce(call, nodeInfos);
                            }
                        }
                        else
                        {
                            LowerGNNEPdp0Dw(call, nodeInfos);
                        }
                    }
                    else
                    {
                        LowerGNNEConv2D(call, nodeInfos);
                    }
                }
                else
                {
                    LowerGnneStore(call, nodeInfos);
                }
            }
            else
            {
                LowerGNNELoad(call, nodeInfos);
            }
        }
        else
        {
            _containNotSupportedOp = true;
        }
    }

    public void Visit(FusionType fusionType, int[] tileOutShape, GlbItemSize maxSizes, List<NodeInfo> nodesInfo)
    {
        nodesInfo[0].OutShape = tileOutShape;
        foreach (NodeInfo item in nodesInfo)
        {
            Call op = item.Op;
            Expr target = op.Target;
            if (!(target is GNNELoad op2))
            {
                if (!(target is GNNEStore op3))
                {
                    if (!(target is GNNEConv2D op4))
                    {
                        if (!(target is GNNEPdp0DW op5))
                        {
                            if (!(target is GNNEPdp0Reduce op6))
                            {
                                if (!(target is Uninitialized op7))
                                {
                                    if (!(target is GNNEPdp1 op8))
                                    {
                                        if (!(target is GNNETranspose op9))
                                        {
                                            if (!(target is GNNEActivation op10))
                                            {
                                                throw new NotImplementedException("Not implemented Op in L2 fuse");
                                            }

                                            LowerGnneActivation(op, op10, fusionType, nodesInfo, maxSizes, item);
                                        }
                                        else
                                        {
                                            LowerGnneTranspose(op, op9, fusionType, nodesInfo, maxSizes, item);
                                        }
                                    }
                                    else
                                    {
                                        LowerGnnePdp1(op, op8, fusionType, nodesInfo, maxSizes, item);
                                    }
                                }
                                else
                                {
                                    LowerUninitialized(op, op7, fusionType, nodesInfo, maxSizes, item);
                                }
                            }
                            else
                            {
                                LowerGnnePdp0Reduce(op, op6, fusionType, nodesInfo, maxSizes, item);
                            }
                        }
                        else
                        {
                            LowerGnnePdp0Dw(op, op5, fusionType, nodesInfo, maxSizes, item);
                        }
                    }
                    else
                    {
                        LowerGnneConv2D(op, op4, fusionType, nodesInfo, maxSizes, item);
                    }
                }
                else
                {
                    LowerGnneStore(op, op3, fusionType, nodesInfo, maxSizes, item);
                }
            }
            else
            {
                LowerGnneLoad(op, op2, fusionType, nodesInfo, maxSizes, item);
            }
        }
    }

    public List<NodeInfo> GetFusionInfo(Fusion fusion, AllocateResult allocation)
    {
        List<NodeInfo> list = new List<NodeInfo>();
        List<int> freeBufferIndices = (_ofBufNum == 2) ? new int[2] { 0, 1 }.ToList() : new int[3] { 0, 1, 2 }.ToList();
        Dictionary<Call, int> dictionary = new Dictionary<Call, int>();
        List<NodeInfo> reversedNodes = new List<NodeInfo>();
        reversedNodes.AddRange(allocation.NodeInfo);
        reversedNodes.Reverse();
        using List<NodeInfo>.Enumerator enumerator = reversedNodes.GetEnumerator();
        NodeInfo ni;
        for (; enumerator.MoveNext(); list.Add(ni))
        {
            ni = enumerator.Current;
            Call op = ni.Op;
            if ((object)op == null || !(op.Target is GNNEPdp0DW))
            {
                op = ni.Op;
                if ((object)op == null || !(op.Target is GNNEPdp0Reduce))
                {
                    op = ni.Op;
                    if ((object)op == null || !(op.Target is GNNEActivation) ||
                        !(ni.Op[GNNEActivation.InputB] == None.Default) ||
                        !((GNNEActivation)ni.Op.Target).InputFromL1[0])
                    {
                        op = ni.Op;
                        if ((object)op != null && op.Target is GNNEActivation &&
                            ni.Op[GNNEActivation.InputB] != None.Default &&
                            ((GNNEActivation)ni.Op.Target).InputFromL1.Any((bool x) => x))
                        {
                            op = ni.Op;
                            int l1InputIndex = ((object)op != null && op.Target is GNNEActivation &&
                                        ((GNNEActivation)ni.Op.Target).InputFromL1[1])
                                ? 1
                                : 0;
                            int index = l1InputIndex;
                            NodeInfo preNi = reversedNodes.Find((NodeInfo nI) => nI.Op == ni.Op.Arguments[index]);
                            ni.Nb.OfBufferIndex = preNi.Nb.OfBufferIndex;
                            ni.Nb.OutputsSize = ni.Children.Count;
                            ni.Nb.OfmapOffset = preNi.Nb.OfmapOffset;
                            list.Find((NodeInfo nI) => nI.Op == preNi.Op).Nb.AlignType = ni.Nb.AlignType;
                            dictionary.Add(ni.Op, ni.Nb.OutputsSize);
                            if (!dictionary.ContainsKey((Call)ni.Op.Arguments[1 - index]))
                            {
                                continue;
                            }

                            dictionary[(Call)ni.Op.Arguments[1 - index]]--;
                            if (dictionary[(Call)ni.Op.Arguments[1 - index]] == 0)
                            {
                                freeBufferIndices.Add(list.Find((NodeInfo nI) => nI.Op == (Call)ni.Op.Arguments[1 - index]).Nb
                                    .OfBufferIndex);
                            }

                            continue;
                        }

                        if (freeBufferIndices.Count == 0)
                        {
                            allocation.IsOk = false;
                            return new List<NodeInfo>();
                        }

                        int bufferIndex = freeBufferIndices[0];
                        freeBufferIndices.RemoveAt(0);
                        ni.Nb.OfBufferIndex = bufferIndex;
                        ni.Nb.OutputsSize = ni.Children.Count;
                        ni.Nb.OfmapOffset = bufferIndex * allocation.BufferSize[ItemName.Ofmap];
                        dictionary.Add(ni.Op, ni.Nb.OutputsSize);
                        op = ni.Op;
                        if (((object)op == null || !(op.Target is GNNELoad)) &&
                            dictionary.ContainsKey((Call)ni.Op.Arguments[0]))
                        {
                            dictionary[(Call)ni.Op.Arguments[0]]--;
                            if (dictionary[(Call)ni.Op.Arguments[0]] == 0)
                            {
                                freeBufferIndices.Add(
                                    list.Find((NodeInfo nI) => nI.Op == (Call)ni.Op.Arguments[0]).Nb.OfBufferIndex);
                            }
                        }

                        op = ni.Op;
                        if ((object)op == null || !(op.Target is GNNEActivation) ||
                            !(ni.Op[GNNEActivation.InputB] != None.Default) ||
                            !dictionary.ContainsKey((Call)ni.Op.Arguments[1]))
                        {
                            continue;
                        }

                        dictionary[(Call)ni.Op.Arguments[1]]--;
                        if (dictionary[(Call)ni.Op.Arguments[1]] == 0)
                        {
                            freeBufferIndices.Add(list.Find((NodeInfo nI) => nI.Op == (Call)ni.Op.Arguments[1]).Nb.OfBufferIndex);
                        }

                        continue;
                    }
                }
            }

            NodeInfo preNi2 = reversedNodes.Find((NodeInfo nI) => nI.Op == ni.Op.Arguments[0]);
            ni.Nb.OfBufferIndex = preNi2.Nb.OfBufferIndex;
            ni.Nb.OutputsSize = ni.Children.Count;
            ni.Nb.OfmapOffset = preNi2.Nb.OfmapOffset;
            list.Find((NodeInfo nI) => nI.Op == preNi2.Op).Nb.AlignType = ni.Nb.AlignType;
            dictionary.Add(ni.Op, ni.Nb.OutputsSize);
        }

        return list;
    }

    public PrimFunction BuildSchedule(FusionType fusionType, FusionInfo fusionInfo)
    {
        return new TileLayerGroup().BuildSchedule(fusionInfo);
    }

    public PrimFunction VisitToPrimFunc(Fusion fusion)
    {
        return T.PrimFunc("nop", K230RtModule.Kind).Body().Build();
    }

    private void UpdateAlignType(List<NodeInfo> nodeInfos)
    {
        foreach (NodeInfo ni in nodeInfos)
        {
            if (ni.Children.Exists(delegate(NodeInfo subNi)
                {
                    Call op2 = subNi.Op;
                    return (object)op2 != null && op2.Target is GNNEActivation;
                }))
            {
                foreach (NodeInfo item in from child in ni.Children.Where(delegate(NodeInfo child)
                         {
                             Call op2 = child.Op;
                             return (object)op2 != null && op2.Target is GNNEActivation &&
                                    child.Op[GNNEActivation.InputB] != None.Default &&
                                    ((GNNEActivation)child.Op.Target).InputFromL1.Any((bool x) => x);
                         })
                         let index = ((GNNEActivation)child.Op.Target).InputFromL1[0] ? 1 : 0
                         where (object)(Call)child.Op.Arguments[index] == ni.Op
                         select child)
                {
                    _ = item;
                    if (ni.Op.CheckedShape[2] == 1)
                    {
                        ni.Nb.AlignType = AlignedType.FAligned;
                    }
                    else
                    {
                        ni.Nb.AlignType = AlignedType.EAligned;
                    }
                }
            }

            Call op = ni.Op;
            if ((object)op != null && op.Target is GNNEPdp1 or GNNETranspose)
            {
                ni.Nb.AlignType = AlignedType.FAligned;
            }

            if (ni.Children.Exists(delegate(NodeInfo subNi)
                {
                    Call op2 = subNi.Op;
                    if ((object)op2 == null || !(op2.Target is GNNEPdp1))
                    {
                        op2 = subNi.Op;
                        if ((object)op2 != null)
                        {
                            return op2.Target is GNNETranspose;
                        }

                        return false;
                    }

                    return true;
                }))
            {
                ni.Nb.AlignType = AlignedType.FAligned;
            }
        }
    }

    private void UpdateNodeInfo(List<NodeInfo> nodeInfos, Call call, NodeInfo currNode = null)
    {
        int nodeIndex = nodeInfos.FindIndex((NodeInfo ni) => ni.Op == call);
        if (nodeIndex >= 0)
        {
            NodeInfo nodeInfo = nodeInfos[nodeIndex];
            nodeInfos.RemoveAt(nodeIndex);
            if (currNode != null && !nodeInfo.Children.Contains(currNode))
            {
                nodeInfo.Children.Add(currNode);
            }

            nodeInfos.Add(nodeInfo);
        }
        else
        {
            nodeInfos.Add(new NodeInfo(
                call,
                (currNode != null) ? new List<NodeInfo> { currNode } : new List<NodeInfo>()));
        }
    }

    private void LowerGNNELoad(Call call, List<NodeInfo> nodeInfos)
    {
        if (!(call[GNNELoad.Input] is TensorConst))
        {
            UpdateNodeInfo(nodeInfos, call);
            if (!(call[GNNELoad.Input] is Var))
            {
                UpdateNodeInfo(nodeInfos, call[GNNELoad.Input] as Call, nodeInfos[nodeInfos.Count - 1]);
                Visit(call[GNNELoad.Input], nodeInfos);
            }
        }
    }

    private void LowerGnneStore(Call call, List<NodeInfo> nodeInfos)
    {
        UpdateNodeInfo(nodeInfos, call);
        UpdateNodeInfo(nodeInfos, call[GNNEStore.Input] as Call, nodeInfos[nodeInfos.Count - 1]);
        Visit(call[GNNEStore.Input], nodeInfos);
    }

    private void LowerGNNEConv2D(Call call, List<NodeInfo> nodeInfos)
    {
        int[] inputShape = call[GNNEConv2D.Input].CheckedShape.ToValueArray();
        int[] outputShape = call.CheckedShape.ToValueArray();
        int[] weightsShape = call[GNNEConv2D.Weights].CheckedShape.ToValueArray();
        int[] source = ((TensorConst)call[GNNEConv2D.Padding]).Value.ToArray<int>();
        int strideH = ((TensorConst)call[GNNEConv2D.Stride]).Value.ToArray<int>()[0];
        int strideW = ((TensorConst)call[GNNEConv2D.Stride]).Value.ToArray<int>()[1];
        int dilationH = ((TensorConst)call[GNNEConv2D.Dilation]).Value.ToArray<int>()[0];
        int dilationW = ((TensorConst)call[GNNEConv2D.Dilation]).Value.ToArray<int>()[1];
        int groups = ((TensorConst)call[GNNEConv2D.Groups]).Value.ToScalar<int>();
        if (inputShape[0] > 1 && outputShape[2] == 1 && outputShape[3] == 1 && inputShape[2] == 1 && inputShape[3] == 1 && weightsShape[2] == 1 &&
            weightsShape[3] == 1 && strideH == 1 && strideW == 1 && dilationH == 1 && dilationW == 1 && source.Sum() == 0 && groups == 1)
        {
            _containNotSupportedOp = true;
        }

        UpdateNodeInfo(nodeInfos, call);
        UpdateNodeInfo(nodeInfos, call[GNNEConv2D.Input] as Call, nodeInfos[nodeInfos.Count - 1]);
        Visit(call[GNNEConv2D.Input], nodeInfos);
    }

    private void LowerGNNEPdp0Dw(Call call, List<NodeInfo> nodeInfos)
    {
        UpdateNodeInfo(nodeInfos, call);
        UpdateNodeInfo(nodeInfos, call[GNNEPdp0DW.Input] as Call, nodeInfos[nodeInfos.Count - 1]);
        Visit(call[GNNEPdp0DW.Input], nodeInfos);
    }

    private void LowerUninitialized(Call call, List<NodeInfo> nodeInfos)
    {
    }

    private void LowerGnnePdp1(Call call, List<NodeInfo> nodeInfos)
    {
        UpdateNodeInfo(nodeInfos, call);
        UpdateNodeInfo(nodeInfos, call[GNNEPdp1.Input] as Call, nodeInfos[nodeInfos.Count - 1]);
        Visit(call[GNNEPdp1.Input], nodeInfos);
    }

    private void LowerGnnePdp0Reduce(Call call, List<NodeInfo> nodeInfos)
    {
        UpdateNodeInfo(nodeInfos, call);
        UpdateNodeInfo(nodeInfos, call[GNNEPdp0Reduce.Input] as Call, nodeInfos[nodeInfos.Count - 1]);
        Visit(call[GNNEPdp0Reduce.Input], nodeInfos);
    }

    private void LowerGnneTranspose(Call call, List<NodeInfo> nodeInfos)
    {
        UpdateNodeInfo(nodeInfos, call);
        UpdateNodeInfo(nodeInfos, call[GNNETranspose.Input] as Call, nodeInfos[nodeInfos.Count - 1]);
        Visit(call[GNNETranspose.Input], nodeInfos);
    }

    private void LowerGnneActivation(Call call, List<NodeInfo> nodeInfos)
    {
        UpdateNodeInfo(nodeInfos, call);
        NodeInfo currNode = nodeInfos[nodeInfos.Count - 1];
        bool visitedInputB = false;
        if (call[GNNEActivation.InputB] != None.Default && call[GNNEActivation.InputB] is Call call2 &&
            call2.Target is GNNEConv2D && ((GNNEActivation)call.Target).InputFromL1[1])
        {
            visitedInputB = true;
            UpdateNodeInfo(nodeInfos, call[GNNEActivation.InputB] as Call, currNode);
            Visit(call[GNNEActivation.InputB], nodeInfos);
        }

        if (call[GNNEActivation.InputB] == None.Default || !(call[GNNEActivation.InputA] is Call call3) ||
            !(call3.Target is GNNELoad) ||
            ((((Call)call[GNNEActivation.InputA])[GNNELoad.Input] is Var ||
              (((Call)call[GNNEActivation.InputA])[GNNELoad.Input] is Call call4 && call4.Target is Reshape)) &&
             call[GNNEActivation.InputB] is Call call5 && call5.Target is GNNELoad))
        {
            UpdateNodeInfo(nodeInfos, call[GNNEActivation.InputA] as Call, currNode);
            Visit(call[GNNEActivation.InputA], nodeInfos);
        }

        if (!visitedInputB && call[GNNEActivation.InputB] != None.Default && (!(call[GNNEActivation.InputB] is Call call6) ||
                                                                     !(call6.Target is GNNELoad) ||
                                                                     (((Call)call[GNNEActivation.InputB])[
                                                                          GNNELoad.Input] is Var &&
                                                                      call[GNNEActivation.InputA] is Call call7 &&
                                                                      call7.Target is GNNELoad &&
                                                                      ((Call)call[GNNEActivation.InputA])[
                                                                          GNNELoad.Input]
                                                                      is TensorConst)))
        {
            UpdateNodeInfo(nodeInfos, call[GNNEActivation.InputB] as Call, currNode);
            Visit(call[GNNEActivation.InputB], nodeInfos);
        }
    }

    private void LowerReshape(Call call, List<NodeInfo> nodeInfos)
    {
        UpdateNodeInfo(nodeInfos, call);
        if (!(call[Reshape.Input] is Var))
        {
            UpdateNodeInfo(nodeInfos, call[Reshape.Input] as Call, nodeInfos[nodeInfos.Count - 1]);
            Visit(call[Reshape.Input], nodeInfos);
        }
    }

    private void LowerGnneLoad(Call call, GNNELoad op, FusionType fusionType, List<NodeInfo> nodesInfo,
        GlbItemSize maxSizes, NodeInfo currNode)
    {
        int[] tileOutShape = currNode.OutShape;
        if (!(call[GNNELoad.Input] is TensorConst))
        {
            int[] tileShape = tileOutShape;
            int tileH = tileShape[2];
            int tileW = tileShape[3];
            int alignmentFactor = (TileUtilities.GetBytesPerElement(call.CheckedDataType) == 1) ? 32 : 16;
            if (currNode.Nb.AlignType == AlignedType.EAligned)
            {
                tileH = TileUtilities.GetAlignedNum(tileH, alignmentFactor);
            }

            if (currNode.Nb.AlignType == AlignedType.FAligned)
            {
                tileW = TileUtilities.GetAlignedNum(tileW, alignmentFactor);
            }

            int allocatedBytes = new TensorOnGlb(new int[4] { tileShape[0], tileShape[1], tileH, tileW }, call.CheckedDataType, 0)
                .AllocatedBytes;
            allocatedBytes = TileUtilities.GetAlignedNum(allocatedBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
            allocatedBytes *= _ofBufNum;
            if (allocatedBytes > maxSizes.MaxOfmapSize)
            {
                maxSizes.MaxOfmapSize = allocatedBytes;
            }
        }

        nodesInfo.Where((NodeInfo node) => node.Op == call[GNNELoad.Input]).ToList().ForEach(delegate(NodeInfo node)
        {
            node.OutShape = tileOutShape;
        });
    }

    private void LowerGnneStore(Call call, GNNEStore op, FusionType fusionType, List<NodeInfo> nodesInfo,
        GlbItemSize maxSizes, NodeInfo currNode)
    {
        int[] tileOutShape = currNode.OutShape;
        int[] tileShape = tileOutShape;
        int tileH = tileShape[2];
        int tileW = tileShape[3];
        int allocatedBytes =
            new TensorOnGlb(new int[4] { tileShape[0], tileShape[1], tileH, tileW }, call[GNNEStore.Input].CheckedDataType, 0)
                .AllocatedBytes;
        allocatedBytes = TileUtilities.GetAlignedNum(allocatedBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        allocatedBytes *= _ofBufNum;
        if (allocatedBytes > maxSizes.MaxOfmapSize)
        {
            maxSizes.MaxOfmapSize = allocatedBytes;
        }

        nodesInfo.Where((NodeInfo node) => node.Op == call[GNNEStore.Input]).ToList().ForEach(delegate(NodeInfo node)
        {
            node.OutShape = tileOutShape;
        });
    }

    private void LowerGnnePdp1(Call call, GNNEPdp1 op, FusionType fusionType, List<NodeInfo> nodesInfo,
        GlbItemSize maxSizes, NodeInfo currNode)
    {
        int[] outShape = currNode.OutShape;
        int[] inputShape = call[GNNEPdp1.Input].CheckedShape.ToValueArray();
        int[] outputShape = call.CheckedShape.ToValueArray();
        int[] padding = ((TensorConst)call[GNNEPdp1.Padding]).Value.ToArray<int>();
        int[] paddingH = new int[2] { padding[0], padding[1] };
        int[] paddingW = new int[2] { padding[2], padding[3] };
        int uh = ((TensorConst)call[GNNEPdp1.Stride]).Value.ToArray<int>()[0];
        int uh2 = ((TensorConst)call[GNNEPdp1.Stride]).Value.ToArray<int>()[1];
        int dh = 1;
        int dh2 = 1;
        int filterH = ((TensorConst)call[GNNEPdp1.Filter]).Value.ToArray<int>()[0];
        int filterW = ((TensorConst)call[GNNEPdp1.Filter]).Value.ToArray<int>()[1];
        _ = call[GNNEPdp1.Input].CheckedDataType;
        DataType checkedDataType = call.CheckedDataType;
        int r = filterH;
        int r2 = filterW;
        bool isGlobalPool = inputShape[2] == filterH && inputShape[3] == filterW &&
                    (inputShape[2] > 16 || inputShape[3] > 64 || inputShape[2] * inputShape[3] > 256);
        if (fusionType != FusionType.L2IfFullWPp && isGlobalPool)
        {
            maxSizes.MaxOfmapSize = GNNEEnv.GlbSize;
        }

        int[] tileShape = outShape;
        int tileH = tileShape[2];
        if (isGlobalPool)
        {
            int cappedFilterH = (filterH > 16) ? 16 : filterH;
            tileH = (int)Math.Ceiling(1f * (float)inputShape[2] / (float)cappedFilterH);
        }

        int tileW = tileShape[3];
        int alignmentFactor = (TileUtilities.GetBytesPerElement(isGlobalPool ? DataTypes.Float16 : checkedDataType) == 1)
            ? 32
            : 16;
        int alignedNum = TileUtilities.GetAlignedNum(tileW, alignmentFactor);
        int allocatedBytes = new TensorOnGlb(
            new int[4] { tileShape[0], tileShape[1], tileH, alignedNum },
            isGlobalPool ? DataTypes.Float16 : checkedDataType, 0).AllocatedBytes;
        allocatedBytes = TileUtilities.GetAlignedNum(allocatedBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        allocatedBytes *= _ofBufNum;
        int tileInH = SpaceSearcher.GetInputHeight(tileH, inputShape[2], r, outputShape[2], uh, dh,
            new Padding(paddingH[0], paddingH[1]));
        int tileInW = SpaceSearcher.GetInputHeight(tileW, inputShape[3], r2, outputShape[3], uh2, dh2,
            new Padding(paddingW[0], paddingW[1]));
        if (fusionType == FusionType.L2IfFullWPp)
        {
            int clampedInH = Math.Min(tileInH, inputShape[2]);
            tileInW = Math.Min(tileInW, inputShape[3]);
            tileInH = clampedInH;
        }

        int[] currentInputShape = new int[4] { inputShape[0], inputShape[1], tileInH, tileInW };
        if (allocatedBytes > maxSizes.MaxOfmapSize)
        {
            maxSizes.MaxOfmapSize = allocatedBytes;
        }

        nodesInfo.Where((NodeInfo node) => node.Op == call[GNNEPdp1.Input]).ToList().ForEach(delegate(NodeInfo node)
        {
            node.OutShape =
                (node.OutShape.Length == 0 || currentInputShape.Aggregate((int a, int b) => a * b) >
                    node.OutShape.Aggregate((int a, int b) => a * b))
                    ? currentInputShape
                    : node.OutShape;
        });
    }

    private void LowerGnnePdp0Reduce(Call call, GNNEPdp0Reduce op, FusionType fusionType, List<NodeInfo> nodesInfo,
        GlbItemSize maxSizes, NodeInfo currNode)
    {
        int[] outShape = currNode.OutShape;
        int[] inputShape = call[GNNEPdp0Reduce.Input].CheckedShape.ToValueArray();
        int[] outputShape = call.CheckedShape.ToValueArray();
        int[] padding = ((TensorConst)call[GNNEPdp0Reduce.Padding]).Value.ToArray<int>();
        int[] paddingH = new int[2] { padding[0], padding[1] };
        int[] paddingW = new int[2] { padding[2], padding[3] };
        int uh = ((TensorConst)call[GNNEPdp0Reduce.Stride]).Value.ToArray<int>()[0];
        int uh2 = ((TensorConst)call[GNNEPdp0Reduce.Stride]).Value.ToArray<int>()[1];
        int dh = 1;
        int dh2 = 1;
        int filterH = ((TensorConst)call[GNNEPdp0Reduce.Filter]).Value.ToArray<int>()[0];
        int filterW = ((TensorConst)call[GNNEPdp0Reduce.Filter]).Value.ToArray<int>()[1];
        _ = call[GNNEPdp0Reduce.Input].CheckedDataType;
        DataType checkedDataType = call.CheckedDataType;
        int outChannels = outputShape[1];
        int r = filterH;
        int r2 = filterW;
        int c = outChannels;
        int[] tileShape = outShape;
        int alignmentFactor = (TileUtilities.GetBytesPerElement(checkedDataType) == 1) ? 32 : 16;
        int tileH = tileShape[2];
        if (currNode.Nb.AlignType == AlignedType.EAligned)
        {
            tileH = TileUtilities.GetAlignedNum(tileH, alignmentFactor);
        }

        int tileW = tileShape[3];
        if (currNode.Nb.AlignType == AlignedType.FAligned)
        {
            tileW = TileUtilities.GetAlignedNum(tileW, alignmentFactor);
        }

        int allocatedBytes = new TensorOnGlb(new int[4] { tileShape[0], tileShape[1], tileH, tileW }, checkedDataType, 0)
            .AllocatedBytes;
        allocatedBytes = TileUtilities.GetAlignedNum(allocatedBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        allocatedBytes *= _ofBufNum;
        int tileInH = SpaceSearcher.GetInputHeight(tileShape[2], inputShape[2], r, outputShape[2], uh, dh,
            new Padding(paddingH[0], paddingH[1]));
        int tileInW = SpaceSearcher.GetInputHeight(tileShape[3], inputShape[3], r2, outputShape[3], uh2, dh2,
            new Padding(paddingW[0], paddingW[1]));
        if (fusionType == FusionType.L2IfFullWPp)
        {
            int clampedInH = Math.Min(tileInH, inputShape[2]);
            tileInW = Math.Min(tileInW, inputShape[3]);
            tileInH = clampedInH;
        }

        int[] currentInputShape = new int[4] { inputShape[0], inputShape[1], tileInH, tileInW };
        if (allocatedBytes > maxSizes.MaxOfmapSize)
        {
            maxSizes.MaxOfmapSize = allocatedBytes;
        }

        NodeBuffer nb = currNode.Nb;
        nb.Pdp0ActOffset = maxSizes.MaxPdp0ActSize;
        int alignedNum = TileUtilities.GetAlignedNum(
            SpaceSearcher.GetActSize(c, GNNEEnv.ActNumPerChan, 2),
            GNNEEnv.ActBankWidth * GNNEEnv.GlbBankWidth);
        maxSizes.MaxPdp0ActSize += alignedNum;
        currNode.Nb = nb;
        nodesInfo.Where((NodeInfo node) => node.Op == call[GNNEPdp0Reduce.Input]).ToList().ForEach(
            delegate(NodeInfo node)
            {
                node.OutShape =
                    (node.OutShape.Length == 0 || currentInputShape.Aggregate((int a, int b) => a * b) >
                        node.OutShape.Aggregate((int a, int b) => a * b))
                        ? currentInputShape
                        : node.OutShape;
            });
    }

    private void LowerGnneConv2D(Call call, GNNEConv2D op, FusionType fusionType, List<NodeInfo> nodesInfo,
        GlbItemSize maxSizes, NodeInfo currNode)
    {
        int[] outShape = currNode.OutShape;
        int[] inputShape = call[GNNEConv2D.Input].CheckedShape.ToValueArray();
        int[] outputShape = call.CheckedShape.ToValueArray();
        int[] weightsShape = call[GNNEConv2D.Weights].CheckedShape.ToValueArray();
        int[] padding = ((TensorConst)call[GNNEConv2D.Padding]).Value.ToArray<int>();
        int[] source = new int[2] { padding[0], padding[1] };
        _ = new int[2] { padding[2], padding[3] };
        int uh = ((TensorConst)call[GNNEConv2D.Stride]).Value.ToArray<int>()[0];
        int uh2 = ((TensorConst)call[GNNEConv2D.Stride]).Value.ToArray<int>()[1];
        int dh = ((TensorConst)call[GNNEConv2D.Dilation]).Value.ToArray<int>()[0];
        int dh2 = ((TensorConst)call[GNNEConv2D.Dilation]).Value.ToArray<int>()[1];
        int groups = ((TensorConst)call[GNNEConv2D.Groups]).Value.ToScalar<int>();
        DataType checkedDataType = call.CheckedDataType;
        DataType checkedDataType2 = call[GNNEConv2D.Weights].CheckedDataType;
        int weightsOutChannels = weightsShape[0];
        int weightsInChannels = weightsShape[1];
        int kernelH = weightsShape[2];
        int kernelW = weightsShape[3];
        int actChannels = weightsOutChannels;
        int[] tileShape = outShape;
        int alignmentFactor = (TileUtilities.GetBytesPerElement(checkedDataType) == 1) ? 32 : 16;
        int tileH = tileShape[2];
        if (currNode.Nb.AlignType == AlignedType.EAligned)
        {
            tileH = TileUtilities.GetAlignedNum(tileH, alignmentFactor);
        }

        int tileW = tileShape[3];
        if (currNode.Nb.AlignType == AlignedType.FAligned)
        {
            tileW = TileUtilities.GetAlignedNum(tileW, alignmentFactor);
        }

        int allocatedBytes = new TensorOnGlb(new int[4] { tileShape[0], tileShape[1], tileH, tileW }, checkedDataType, 0)
            .AllocatedBytes;
        allocatedBytes = TileUtilities.GetAlignedNum(allocatedBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        allocatedBytes *= _ofBufNum;
        int tileInH = SpaceSearcher.GetInputHeight(tileShape[2], inputShape[2], kernelH, outputShape[2], uh, dh, new Padding(0, 0));
        if (weightsShape[1] * weightsShape[2] <= GNNEEnv.PuHeight && weightsShape[2] != 1)
        {
            tileInH += source.Sum();
        }

        int tileInW = SpaceSearcher.GetInputHeight(tileShape[3], inputShape[3], kernelW, outputShape[3], uh2, dh2, new Padding(0, 0));
        if (fusionType == FusionType.L2IfFullWPp)
        {
            int clampedInH = Math.Min(tileInH, inputShape[2]);
            tileInW = Math.Min(tileInW, inputShape[3]);
            tileInH = clampedInH;
        }

        int[] currentInputShape = new int[4] { inputShape[0], (groups == 1) ? weightsShape[1] : inputShape[1], tileInH, tileInW };
        NodeBuffer nb = currNode.Nb;
        Call op2 = currNode.Children[0].Op;
        bool feedsActivationFromL1 = currNode.Children.Count == 1 &&
            (object)op2 != null && op2.Target is GNNEActivation &&
            ((((GNNEActivation)currNode.Children[0].Op.Target).InputFromL1[0] &&
              currNode.Children[0].Op[GNNEActivation.InputA] == call) ||
             (((GNNEActivation)currNode.Children[0].Op.Target).InputFromL1[1] &&
              currNode.Children[0].Op[GNNEActivation.InputB] == call));
        bool feedsPdp0 = (object)op2 != null && op2.Target is GNNEPdp0DW or GNNEPdp0Reduce;
        if (!feedsActivationFromL1 && !feedsPdp0 && allocatedBytes > maxSizes.MaxOfmapSize)
        {
            maxSizes.MaxOfmapSize = allocatedBytes;
        }

        _ = new TensorOnGlb(new int[4] { kernelH, kernelW, weightsInChannels, weightsOutChannels }, checkedDataType2, 0);
        if (fusionType == FusionType.L2IfFullWPp || fusionType == FusionType.L2IfSplitWPp)
        {
            int alignedNum = TileUtilities.GetAlignedNum(
                SpaceSearcher.GetWeightSize(kernelH, kernelW, weightsInChannels, weightsOutChannels, TileUtilities.GetBytesPerElement(checkedDataType2)),
                GNNEEnv.WBankWidth * GNNEEnv.GlbBankWidth);
            if (alignedNum > maxSizes.MaxWeightSize)
            {
                maxSizes.MaxWeightSize = alignedNum;
            }
        }
        else
        {
            nb.WeightOffset = maxSizes.MaxWeightSize;
            int alignedNum2 = TileUtilities.GetAlignedNum(
                SpaceSearcher.GetWeightSize(kernelH, kernelW, weightsInChannels, weightsOutChannels, TileUtilities.GetBytesPerElement(checkedDataType2)),
                GNNEEnv.WBankWidth * GNNEEnv.GlbBankWidth);
            maxSizes.MaxWeightSize += alignedNum2;
        }

        nb.ActOffset = maxSizes.MaxActSize;
        int alignedNum3 = TileUtilities.GetAlignedNum(
            SpaceSearcher.GetActSize(actChannels, GNNEEnv.ActNumPerChan, 2),
            GNNEEnv.ActBankWidth * GNNEEnv.GlbBankWidth);
        maxSizes.MaxActSize += alignedNum3;
        if (checkedDataType2 == DataTypes.UInt8 || checkedDataType2 == DataTypes.Int16)
        {
            nb.WeightQargOffset = maxSizes.MaxWeightQargSize;
            int alignedNum4 = TileUtilities.GetAlignedNum(
                SpaceSearcher.GetWQargSize(
                    (int)Math.Ceiling((double)actChannels * 1.0 / (double)GNNEEnv.PuHeight) * GNNEEnv.PuWidth, 1),
                GNNEEnv.WQargBankWidth * GNNEEnv.GlbBankWidth);
            maxSizes.MaxWeightQargSize += alignedNum4;
        }

        currNode.Nb = nb;
        nodesInfo.Where((NodeInfo node) => node.Op == call[GNNEConv2D.Input]).ToList().ForEach(delegate(NodeInfo node)
        {
            node.OutShape =
                (node.OutShape.Length == 0 || currentInputShape.Aggregate((int a, int b) => a * b) >
                    node.OutShape.Aggregate((int a, int b) => a * b))
                    ? currentInputShape
                    : node.OutShape;
        });
    }

    private void LowerGnnePdp0Dw(Call call, GNNEPdp0DW op, FusionType fusionType, List<NodeInfo> nodesInfo,
        GlbItemSize maxSizes, NodeInfo currNode)
    {
        int[] outShape = currNode.OutShape;
        int[] inputShape = call[GNNEPdp0DW.Input].CheckedShape.ToValueArray();
        int[] outputShape = call.CheckedShape.ToValueArray();
        int[] weightsShape = call[GNNEPdp0DW.Weights].CheckedShape.ToValueArray();
        int[] padding = ((TensorConst)call[GNNEPdp0DW.Padding]).Value.ToArray<int>();
        int[] source = new int[2] { padding[0], padding[1] };
        _ = new int[2] { padding[2], padding[3] };
        int uh = ((TensorConst)call[GNNEPdp0DW.Stride]).Value.ToArray<int>()[0];
        int uh2 = ((TensorConst)call[GNNEPdp0DW.Stride]).Value.ToArray<int>()[1];
        int dh = ((TensorConst)call[GNNEPdp0DW.Dilation]).Value.ToArray<int>()[0];
        int dh2 = ((TensorConst)call[GNNEPdp0DW.Dilation]).Value.ToArray<int>()[1];
        int groups = ((TensorConst)call[GNNEPdp0DW.Groups]).Value.ToScalar<int>();
        DataType checkedDataType = call.CheckedDataType;
        DataType checkedDataType2 = call[GNNEPdp0DW.Weights].CheckedDataType;
        weightsShape[0] /= inputShape[1];
        weightsShape[1] = TileUtilities.GetAlignedNum(inputShape[1], GNNEEnv.PuWidth);
        int weightsOutChannels = weightsShape[0];
        int weightsInChannels = weightsShape[1];
        int kernelH = weightsShape[2];
        int kernelW = weightsShape[3];
        int actChannels = weightsInChannels * weightsOutChannels;
        int[] tileShape = outShape;
        int alignmentFactor = (TileUtilities.GetBytesPerElement(checkedDataType) == 1) ? 32 : 16;
        int tileH = tileShape[2];
        if (currNode.Nb.AlignType == AlignedType.EAligned)
        {
            tileH = TileUtilities.GetAlignedNum(tileH, alignmentFactor);
        }

        int tileW = tileShape[3];
        if (currNode.Nb.AlignType == AlignedType.FAligned)
        {
            tileW = TileUtilities.GetAlignedNum(tileW, alignmentFactor);
        }

        int allocatedBytes = new TensorOnGlb(new int[4] { tileShape[0], tileShape[1], tileH, tileW }, checkedDataType, 0)
            .AllocatedBytes;
        allocatedBytes = TileUtilities.GetAlignedNum(allocatedBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        allocatedBytes *= _ofBufNum;
        int tileInH = SpaceSearcher.GetInputHeight(tileShape[2], inputShape[2], kernelH, outputShape[2], uh, dh, new Padding(0, 0));
        if (weightsShape[1] * weightsShape[2] <= GNNEEnv.PuHeight && weightsShape[2] != 1)
        {
            tileInH += source.Sum();
        }

        int tileInW = SpaceSearcher.GetInputHeight(tileShape[3], inputShape[3], kernelW, outputShape[3], uh2, dh2, new Padding(0, 0));
        if (fusionType == FusionType.L2IfFullWPp)
        {
            int clampedInH = Math.Min(tileInH, inputShape[2]);
            tileInW = Math.Min(tileInW, inputShape[3]);
            tileInH = clampedInH;
        }

        int[] currentInputShape = new int[4] { inputShape[0], (groups == 1) ? weightsShape[1] : inputShape[1], tileInH, tileInW };
        NodeBuffer nb = currNode.Nb;
        if (allocatedBytes > maxSizes.MaxOfmapSize)
        {
            maxSizes.MaxOfmapSize = allocatedBytes;
        }

        _ = new TensorOnGlb(new int[4] { kernelH, kernelW, weightsInChannels, weightsOutChannels }, checkedDataType2, 0);
        nb.DwWeightOffset = maxSizes.MaxDwWeightSize;
        int alignedNum = TileUtilities.GetAlignedNum(
            SpaceSearcher.GetWeightSize(kernelH, kernelW, TileUtilities.GetAlignedNum(weightsInChannels, GNNEEnv.PuWidth), weightsOutChannels,
                TileUtilities.GetBytesPerElement(checkedDataType2)), GNNEEnv.WBankWidth * GNNEEnv.GlbBankWidth);
        maxSizes.MaxDwWeightSize += alignedNum;
        nb.DwActOffset = maxSizes.MaxDwActSize;
        int alignedNum2 = TileUtilities.GetAlignedNum(
            SpaceSearcher.GetActSize(actChannels, GNNEEnv.ActNumPerChan, 2),
            GNNEEnv.ActBankWidth * GNNEEnv.GlbBankWidth);
        maxSizes.MaxDwActSize += alignedNum2;
        if (checkedDataType2 == DataTypes.UInt8 || checkedDataType2 == DataTypes.Int16)
        {
            nb.DwWeightQargOffset = maxSizes.MaxDwWeightQargSize;
            int alignedNum3 = TileUtilities.GetAlignedNum(
                SpaceSearcher.GetWQargSize(actChannels, 1),
                GNNEEnv.WQargBankWidth * GNNEEnv.GlbBankWidth);
            maxSizes.MaxDwWeightQargSize += alignedNum3;
        }

        currNode.Nb = nb;
        nodesInfo.Where((NodeInfo node) => node.Op == call[GNNEPdp0DW.Input]).ToList().ForEach(delegate(NodeInfo node)
        {
            node.OutShape =
                (node.OutShape.Length == 0 || currentInputShape.Aggregate((int a, int b) => a * b) >
                    node.OutShape.Aggregate((int a, int b) => a * b))
                    ? currentInputShape
                    : node.OutShape;
        });
    }

    private void LowerGnneTranspose(Call call, GNNETranspose op, FusionType fusionType, List<NodeInfo> nodesInfo,
        GlbItemSize maxSizes, NodeInfo currNode)
    {
        int[] outShape = currNode.OutShape;
        if (fusionType != FusionType.L2IfFullWPp)
        {
            maxSizes.MaxOfmapSize = GNNEEnv.GlbSize;
        }

        _ = call[GNNETranspose.Input].CheckedDataType;
        DataType checkedDataType = call.CheckedDataType;
        int[] tileShape = outShape;
        int value = tileShape[3];
        int alignmentFactor = (TileUtilities.GetBytesPerElement(checkedDataType) == 1) ? 32 : 16;
        int alignedNum = TileUtilities.GetAlignedNum(value, alignmentFactor);
        int allocatedBytes =
            new TensorOnGlb(new int[4] { tileShape[0], tileShape[1], tileShape[2], alignedNum }, checkedDataType, 0).AllocatedBytes;
        allocatedBytes = TileUtilities.GetAlignedNum(allocatedBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        allocatedBytes *= _ofBufNum;
        MFU_TRANS_PERMUTE perm = op.Perm;
        int[] tileInShape = GetInputShapes(tileShape[0], tileShape[1], tileShape[2], tileShape[3], perm);
        int[] currentInputShape = new int[4] { tileInShape[0], tileInShape[1], tileInShape[2], tileInShape[3] };
        if (allocatedBytes > maxSizes.MaxOfmapSize)
        {
            maxSizes.MaxOfmapSize = allocatedBytes;
        }

        nodesInfo.Where((NodeInfo node) => node.Op == call[GNNETranspose.Input]).ToList().ForEach(
            delegate(NodeInfo node)
            {
                node.OutShape =
                    (node.OutShape.Length == 0 || currentInputShape.Aggregate((int a, int b) => a * b) >
                        node.OutShape.Aggregate((int a, int b) => a * b))
                        ? currentInputShape
                        : node.OutShape;
            });

        static int[] GetInputShapes(int d0, int d1, int d2, int d3, MFU_TRANS_PERMUTE permute)
        {
            return permute switch
            {
                MFU_TRANS_PERMUTE.NCHW => new int[4] { d0, d1, d2, d3 },
                MFU_TRANS_PERMUTE.NCWH => new int[4] { d0, d1, d3, d2 },
                MFU_TRANS_PERMUTE.NHCW => new int[4] { d0, d2, d1, d3 },
                MFU_TRANS_PERMUTE.NHWC => new int[4] { d0, d3, d1, d2 },
                MFU_TRANS_PERMUTE.NWCH => new int[4] { d0, d2, d3, d1 },
                MFU_TRANS_PERMUTE.NWHC => new int[4] { d0, d3, d2, d1 },
                MFU_TRANS_PERMUTE.CNHW => new int[4] { d1, d0, d2, d3 },
                MFU_TRANS_PERMUTE.CNWH => new int[4] { d1, d0, d3, d2 },
                MFU_TRANS_PERMUTE.CHNW => new int[4] { d2, d0, d1, d3 },
                MFU_TRANS_PERMUTE.CHWN => new int[4] { d3, d0, d1, d2 },
                MFU_TRANS_PERMUTE.CWNH => new int[4] { d2, d0, d3, d1 },
                MFU_TRANS_PERMUTE.CWHN => new int[4] { d3, d0, d2, d1 },
                MFU_TRANS_PERMUTE.HNCW => new int[4] { d1, d2, d0, d3 },
                MFU_TRANS_PERMUTE.HNWC => new int[4] { d1, d3, d0, d2 },
                MFU_TRANS_PERMUTE.HCNW => new int[4] { d2, d1, d0, d3 },
                MFU_TRANS_PERMUTE.HCWN => new int[4] { d3, d1, d0, d2 },
                MFU_TRANS_PERMUTE.HWNC => new int[4] { d2, d3, d0, d1 },
                MFU_TRANS_PERMUTE.HWCN => new int[4] { d3, d2, d0, d1 },
                MFU_TRANS_PERMUTE.WNCH => new int[4] { d1, d2, d3, d0 },
                MFU_TRANS_PERMUTE.WNHC => new int[4] { d1, d3, d2, d0 },
                MFU_TRANS_PERMUTE.WCNH => new int[4] { d2, d1, d3, d0 },
                MFU_TRANS_PERMUTE.WCHN => new int[4] { d3, d1, d2, d0 },
                MFU_TRANS_PERMUTE.WHNC => new int[4] { d2, d3, d1, d0 },
                MFU_TRANS_PERMUTE.WHCN => new int[4] { d3, d2, d1, d0 },
                _ => new int[4] { d0, d1, d2, d3 },
            };
        }
    }

    private void LowerUninitialized(Call call, Uninitialized op, FusionType fusionType, List<NodeInfo> nodesInfo,
        GlbItemSize max_sizes, NodeInfo currNode)
    {
        _ = currNode.OutShape;
    }

    private void LowerGnneActivation(Call call, GNNEActivation op, FusionType fusionType, List<NodeInfo> nodesInfo,
        GlbItemSize maxSizes, NodeInfo currNode)
    {
        int[] outShape = currNode.OutShape;
        bool isUnary = call[GNNEActivation.InputB] == None.Default;
        int[] inputAShape = call[GNNEActivation.InputA].CheckedShape.ToValueArray();
        int[] inputBShape = inputAShape;
        if (!isUnary)
        {
            inputBShape = call[GNNEActivation.InputB].CheckedShape.ToValueArray();
        }

        int[] outputShape = call.CheckedShape.ToValueArray();
        if ((fusionType == FusionType.L2IfSplitWPp || fusionType == FusionType.L2IfSplitWFull) &&
            (!inputAShape.SequenceEqual(outputShape) || !inputBShape.SequenceEqual(outputShape)))
        {
            maxSizes.MaxOfmapSize = GNNEEnv.GlbSize;
        }

        DataType checkedDataType = call[GNNEActivation.InputA].CheckedDataType;
        DataType dataType = checkedDataType;
        if (!isUnary)
        {
            dataType = call[GNNEActivation.InputB].CheckedDataType;
        }

        DataType checkedDataType2 = call.CheckedDataType;
        int outChannels = outputShape[1];
        int[] tileShape = outShape;
        int alignmentFactor = (TileUtilities.GetBytesPerElement(checkedDataType2) == 1) ? 32 : 16;
        int tileH = tileShape[2];
        if (currNode.Nb.AlignType == AlignedType.EAligned)
        {
            tileH = TileUtilities.GetAlignedNum(tileH, alignmentFactor);
        }

        int tileW = tileShape[3];
        if (currNode.Nb.AlignType == AlignedType.FAligned)
        {
            tileW = TileUtilities.GetAlignedNum(tileW, alignmentFactor);
        }

        int allocatedBytes = new TensorOnGlb(new int[4] { tileShape[0], tileShape[1], tileH, tileW }, checkedDataType2, 0)
            .AllocatedBytes;
        allocatedBytes = TileUtilities.GetAlignedNum(allocatedBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        allocatedBytes *= _ofBufNum;
        int tileInH = tileShape[2];
        int tileInW = tileShape[3];
        int[] currentInputShape = new int[4] { outputShape[0], outputShape[1], tileInH, tileInW };
        NodeBuffer nb = currNode.Nb;
        if (allocatedBytes > maxSizes.MaxOfmapSize)
        {
            maxSizes.MaxOfmapSize = allocatedBytes;
        }

        nb.Ifmap2Offset = maxSizes.MaxIfmap2Size;
        int ifmap2Bytes = 0;
        if (!isUnary && call[GNNEActivation.InputA] is Call call2 && call2.Target is GNNELoad &&
            nodesInfo.Find((NodeInfo ni) => ni.Op == call[GNNEActivation.InputA]) == null)
        {
            tileInH = inputAShape[2];
            tileInW = inputAShape[3];
            ifmap2Bytes = new TensorOnGlb(
                new int[4] { inputAShape[0], inputAShape[1], TileUtilities.GetAlignedNum(tileInH, 32), tileInW },
                checkedDataType, 0).AllocatedBytes;
            ifmap2Bytes = TileUtilities.GetAlignedNum(ifmap2Bytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        }

        if (!isUnary && call[GNNEActivation.InputB] is Call call3 && call3.Target is GNNELoad &&
            nodesInfo.Find((NodeInfo ni) => ni.Op == call[GNNEActivation.InputB]) == null)
        {
            tileInH = inputBShape[2];
            tileInW = inputBShape[3];
            ifmap2Bytes = new TensorOnGlb(
                new int[4] { inputBShape[0], inputBShape[1], TileUtilities.GetAlignedNum(tileInH, 32), tileInW },
                dataType, 0).AllocatedBytes;
            ifmap2Bytes = TileUtilities.GetAlignedNum(ifmap2Bytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        }

        maxSizes.MaxIfmap2Size += ifmap2Bytes;
        nb.Act1Offset = maxSizes.MaxAct1Size;
        bool is16Segments = ((TensorConst)call[GNNEActivation.Is16Segments]).Value.ToScalar<bool>();
        int c = is16Segments ? 1 : outChannels;
        int nActParam = is16Segments ? 49 : GNNEEnv.ActNumPerChan;
        int alignedNum = TileUtilities.GetAlignedNum(
            SpaceSearcher.GetActSize(c, nActParam, 2),
            GNNEEnv.ActBankWidth * GNNEEnv.GlbBankWidth);
        maxSizes.MaxAct1Size += alignedNum;
        currNode.Nb = nb;
        if (fusionType != FusionType.L2IfFullWPp && !inputAShape.SequenceEqual(inputBShape))
        {
            maxSizes.MaxOfmapSize = GNNEEnv.GlbSize;
        }

        nodesInfo.Where((NodeInfo node) => node.Op == call[GNNEActivation.InputA]).ToList().ForEach(
            delegate(NodeInfo node)
            {
                node.OutShape =
                    (node.OutShape.Length == 0 || currentInputShape.Aggregate((int a, int b) => a * b) >
                        node.OutShape.Aggregate((int a, int b) => a * b))
                        ? currentInputShape
                        : node.OutShape;
            });
        if (isUnary)
        {
            return;
        }

        nodesInfo.Where((NodeInfo node) => node.Op == call[GNNEActivation.InputB]).ToList().ForEach(
            delegate(NodeInfo node)
            {
                node.OutShape =
                    (node.OutShape.Length == 0 || currentInputShape.Aggregate((int a, int b) => a * b) >
                        node.OutShape.Aggregate((int a, int b) => a * b))
                        ? currentInputShape
                        : node.OutShape;
            });
    }
}
