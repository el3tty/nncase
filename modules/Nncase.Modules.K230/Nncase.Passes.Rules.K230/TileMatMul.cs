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
public class TileMatMul : RewriteRule<Pattern>
{
    private static int _count = -1;

    private readonly List<Tuple<SegmentND, TensorStat>>[] _ofmapRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _g2LIfRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _g2RWRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly List<Tuple<SegmentND, TensorStat>>[] _l2GOfRec = new List<Tuple<SegmentND, TensorStat>>[2];

    private readonly Dictionary<Var, Nncase.TIR.Buffer> _ifBufferMap =
        new Dictionary<Var, Nncase.TIR.Buffer>(ReferenceEqualityComparer.Instance);

    private CcrHandler _ccrHandler = new CcrHandler();

    private WeightGroupHandler _weightGroup;

    private GprHandler _gpr = new GprHandler();

    private SsrHandler _ssr = new SsrHandler();

    private DataType? _inputType;

    private DataType? _outputType;

    private DataType? _weightType;

    private Call? _mm;

    private Call? _lif;

    private Call? _lw;

    private Call? _lact;

    private Call? _lwQarg;

    private Call? _sof;

    private GNNEShape? _inputShape;

    private GNNEShape? _outputShape;

    private GNNEShape? _weightsShape;

    private int _strideH;

    private int _strideW;

    private int _dilationH;

    private int _dilationW;

    private bool _broadcastW;

    private bool _broadcastIf;

    public override Pattern Pattern { get; } = FusionPattern.IsGNNEMatMulFusion();

    private PrimFunction GetReplace(Call call, Call ld, Call st)
    {
        _count++;
        InitParameters(call, ld, st);
        TileMatMulGlb glb = SearchGlbParameters();
        int[] inputBShape = ((Call)_mm[GNNEMatMul.InputB])[GNNELoad.Input].CheckedShape.ToValueArray();
        List<Nncase.TIR.Buffer> buffers = new List<Nncase.TIR.Buffer>();
        Nncase.TIR.Buffer buffer;
        if (((Call)_mm[GNNEMatMul.InputB])[GNNELoad.Input] is TensorConst)
        {
            T.AttachBuffer((TensorConst)((Call)call[GNNEMatMul.InputB])[GNNELoad.Input], out buffer, "ddrIf");
        }
        else
        {
            T.CreateBuffer(new TensorType(((Call)_mm[GNNEMatMul.InputB])[GNNELoad.Input].CheckedDataType, inputBShape),
                MemoryLocation.Input, out buffer, "ddrIf");
            buffers.Add(buffer);
            _ifBufferMap.Add((Var)((Call)_mm[GNNEMatMul.InputB])[GNNELoad.Input], buffer);
        }

        int[] inputAShape = ((Call)_mm[GNNEMatMul.InputA])[GNNELoad.Input].CheckedShape.ToValueArray();
        Nncase.TIR.Buffer buffer2;
        if (((Call)_mm[GNNEMatMul.InputA])[GNNELoad.Input] is TensorConst)
        {
            T.AttachBuffer((TensorConst)((Call)call[GNNEMatMul.InputA])[GNNELoad.Input], out buffer2, "ddrW");
        }
        else
        {
            T.CreateBuffer(new TensorType(((Call)_mm[GNNEMatMul.InputA])[GNNELoad.Input].CheckedDataType, inputAShape),
                MemoryLocation.Input, out buffer2, "ddrW");
            buffers.Add(buffer2);
            _ifBufferMap.Add((Var)((Call)_mm[GNNEMatMul.InputA])[GNNELoad.Input], buffer2);
        }

        if (buffers.Count == 2)
        {
            buffers.Reverse();
        }

        T.CreateBuffer(new TensorType(st.CheckedDataType, st.CheckedShape), MemoryLocation.Output,
            out Nncase.TIR.Buffer buffer3, "var ddrOf");
        buffers.Add(buffer3);
        T.AttachBuffer((TensorConst)((Call)call[GNNEMatMul.InputABias])[GNNELoadW.Input], out Nncase.TIR.Buffer buffer4,
            "var ddrWQarg");
        T.AttachBuffer((TensorConst)((Call)call[GNNEMatMul.Act])[GNNELoadW.Input], out Nncase.TIR.Buffer buffer5,
            "var ddrAct");
        List<GnneAction> actions = new List<GnneAction>();
        ItemRecStatusInit();
        BuildSchedule(glb, call, ld, st, buffer, buffer3, buffer2, buffer4, buffer5, weightGroupOnly: true,
            _weightGroup);
        actions = BuildSchedule(glb, call, ld, st, buffer, buffer3, buffer2, buffer4, buffer5, weightGroupOnly: false,
            _weightGroup);
        ActionToInstruct actionToInstruct = new ActionToInstruct();
        return T.PrimFunc($"TileMatMul_{_count}", K230RtModule.Kind, buffers.ToArray())
            .Body(actionToInstruct.Instructions(actions), I.END(GP_REGISTER.x0, 0)).Build();
    }

    private List<GnneAction> BuildSchedule(TiledGlb glb, Call conv, Call ld, Call st, Nncase.TIR.Buffer ddrIf,
        Nncase.TIR.Buffer ddrOf, Nncase.TIR.Buffer ddrW, Nncase.TIR.Buffer ddrWQarg, Nncase.TIR.Buffer ddrAct,
        bool weightGroupOnly, WeightGroupHandler weightGroup)
    {
        List<GnneAction> actions = new List<GnneAction>();
        if (!weightGroupOnly)
        {
            _gpr = new GprHandler(GNNEEnv.GprNum);
            _ssr = new SsrHandler(GNNEEnv.SsrNum);
            _ccrHandler = new CcrHandler();
        }

        GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, glb, _ccrHandler, _gpr, _ssr);
        if (!weightGroupOnly)
        {
            UpdateCcrRecStat();
        }

        if (!weightGroupOnly)
        {
            gnneActionUpdater.UpdateMmuConf();
        }

        int ofmapPp = 0;
        int ifmapPp = -1;
        int weightPp = -1;
        int nPingPongSplit = GNNEEnv.NPingPongSplit;
        List<Segment1D> segmentsN =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[0], _outputShape[1]);
        List<Segment1D> segmentsW =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[3], _outputShape[3]);
        List<Segment1D> segmentsC =
            TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[1], _outputShape[2]);
        List<Segment1D> icSegments =
            TileUtilities.GetSegmentStartEndLength(0, glb.GlbMap[ItemName.Ifmap].Dimensions[1], _weightsShape[3]);
        if (!weightGroupOnly)
        {
            gnneActionUpdater.UpdateLoadAct(_lact, ddrAct);
            if (_weightType == DataTypes.UInt8 || _weightType == DataTypes.Int16)
            {
                for (int i = 0; i < (_broadcastW ? 1 : _outputShape[1]); i++)
                {
                    gnneActionUpdater.UpdateLoadWQargMatmul(_lwQarg, weightGroup, ddrWQarg, i);
                }
            }
        }

        Segment1D emptySegment = new Segment1D(..0, new Padding(0, 0));
        Segment1D oneSegment = new Segment1D(..1, new Padding(0, 0));
        List<SegmentND> segmentHistory = new List<SegmentND>();
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        List<SegmentND> ifmapHistory = segmentHistory;
        segmentHistory = new List<SegmentND>();
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        segmentHistory.Add(new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment));
        List<SegmentND> weightHistory = segmentHistory;
        Segment1D oneSegmentRef = oneSegment;
        Segment1D oneSegmentRef2 = oneSegmentRef;
        foreach (Segment1D segmentC in segmentsC)
        {
            foreach (Segment1D segmentN in segmentsN)
            {
                foreach (Segment1D segmentW in segmentsW)
                {
                    SegmentND psum = new SegmentND(segmentN, segmentC, oneSegmentRef, segmentW);
                    SegmentND ofmapSegment = new SegmentND(oneSegment, segmentN, segmentC, segmentW);
                    foreach (Segment1D icSegment in icSegments)
                    {
                        Segment1D segmentWRef = segmentW;
                        SegmentND weight = new SegmentND(segmentC, icSegment, oneSegment, oneSegment);
                        SegmentND weightSegment =
                            new SegmentND(oneSegment, _broadcastW ? oneSegment : segmentN, segmentC, icSegment);
                        if (weightHistory[0] == weightSegment)
                        {
                            weightPp = 0;
                        }
                        else if (weightHistory[1] == weightSegment)
                        {
                            weightPp = 1;
                        }
                        else
                        {
                            weightPp = (weightPp + 1) % 2;
                            if (!weightGroupOnly && _g2RWRec[weightPp].Count > 0 && _g2RWRec[weightPp][0].Item1 == weightSegment)
                            {
                                TensorStat loadStat = _g2RWRec[weightPp][0].Item2;
                                int weightCcrCount = ((loadStat != null && loadStat.IsFirstSlice && loadStat.IsLastSlice) ? 1 : 2);
                                List<CcrSet> ccrsToSet = new List<CcrSet>
                                {
                                    new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Weight, weightPp)),
                                        weightCcrCount)
                                };
                                List<int> stridesD = new List<int>
                                {
                                    weightSegment[1].Length,
                                    segmentC.Length,
                                    TileUtilities.GetAlignedNum(icSegment.Length, GNNEEnv.WAlignNum)
                                };
                                gnneActionUpdater.UpdateLoadIf(weightSegment, _lw, weightPp, ddrW, 0, stridesD,
                                    ItemName.Weight, ccrsToSet);
                            }

                            weightHistory[weightPp] = weightSegment;
                        }

                        SegmentND ifmap = new SegmentND(_broadcastIf ? oneSegment : segmentN, icSegment, oneSegmentRef2,
                            segmentWRef);
                        SegmentND ifmapSegment = new SegmentND(oneSegment, _broadcastIf ? oneSegment : segmentN, icSegment,
                            segmentWRef);
                        if (ifmapHistory[0] == ifmapSegment)
                        {
                            ifmapPp = 0;
                        }
                        else if (ifmapHistory[1] == ifmapSegment)
                        {
                            ifmapPp = 1;
                        }
                        else
                        {
                            ifmapPp = (ifmapPp + 1) % 2;
                            if (!weightGroupOnly && _g2LIfRec[ifmapPp].Count > 0 && _g2LIfRec[ifmapPp][0].Item1 == ifmapSegment)
                            {
                                TensorStat loadStat = _g2LIfRec[ifmapPp][0].Item2;
                                int ifmapCcrCount = ((loadStat != null && loadStat.IsFirstSlice && loadStat.IsLastSlice) ? 1 : 2);
                                List<CcrSet> ccrsToSet2 = new List<CcrSet>
                                {
                                    new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap, ifmapPp)),
                                        ifmapCcrCount)
                                };
                                List<int> stridesD2 = new List<int>
                                {
                                    glb.GlbMap[ItemName.Ifmap].Dimensions[0],
                                    glb.GlbMap[ItemName.Ifmap].Dimensions[1],
                                    glb.GlbMap[ItemName.Ifmap].Dimensions[3]
                                };
                                gnneActionUpdater.UpdateLoadIf(ifmapSegment, _lif, ifmapPp, ddrIf, 0, stridesD2,
                                    ItemName.Ifmap, ccrsToSet2);
                            }

                            ifmapHistory[ifmapPp] = ifmapSegment;
                        }

                        BuildL1Schedule(gnneActionUpdater, glb, ifmap, weight, psum, ifmapPp, ofmapPp, weightPp, weightGroupOnly,
                            weightGroup);
                    }

                    if (!weightGroupOnly)
                    {
                        bool notLastSlice = !_ofmapRec[ofmapPp][0].Item2.IsLastSlice;
                        _ofmapRec[ofmapPp].RemoveAt(0);
                        List<CcrSet> ofmapFakeCcrsToSet = new List<CcrSet>();
                        if ((notLastSlice ? 1 : 0) > (false ? 1 : 0))
                        {
                            ofmapFakeCcrsToSet.Add(new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.OfmapFake, ofmapPp)),
                                1));
                        }

                        List<CcrClr> ccrsToClr = new List<CcrClr>
                        {
                            new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, ofmapPp)))
                        };
                        List<int> stridesS = new List<int>
                        {
                            glb.GlbMap[ItemName.Ofmap].Dimensions[0],
                            glb.GlbMap[ItemName.Ofmap].Dimensions[1],
                            glb.GlbMap[ItemName.Ofmap].Dimensions[3]
                        };
                        gnneActionUpdater.UpdateStoreT(ofmapSegment, _sof, ofmapPp, ddrOf, 0, null, ofmapFakeCcrsToSet, ccrsToClr,
                            ItemName.Ofmap, stridesS);
                    }
                    else
                    {
                        _ofmapRec[ofmapPp].Add(new Tuple<SegmentND, TensorStat>(ofmapSegment,
                            new TensorStat(isFirstSlice: false, isLastSlice: false)));
                    }

                    ofmapPp = (ofmapPp + 1) % nPingPongSplit;
                }
            }
        }

        TileUtilities.Assert(_ccrHandler.CcrSanityCheck(), "_ccrHandler.CcrSanityCheck()",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileMatMul.cs",
            309);
        return actions;
    }

    private List<int> L1Search(TiledGlb glb, SegmentND weight, SegmentND psum)
    {
        int puHeight = Math.Min(GNNEEnv.PuHeight, _weightsShape[3]);
        int puWidth = Math.Min(GNNEEnv.PuWidth, _weightsShape[2]);
        int h = 1;
        int w = 1;
        int e = 1;
        int f = 1;
        int stepE = 1;
        int stepF = 1;
        int psumPingPangSplit = 1;
        int ifBytesPerElementGlb = TileUtilities.GetBytesPerElement(_inputType);
        if (!HandleL1Allocate(h, w, e, f, psumPingPangSplit, ifBytesPerElementGlb))
        {
            throw new NotSupportedException("exceeds L1 size");
        }

        while (f < psum[3].Length && h * w * ifBytesPerElementGlb < GNNEEnv.IfL1Size / GNNEEnv.PuHeight / 2 &&
               IncreaseFBy())
        {
        }

        return new List<int>
        {
            puHeight,
            puWidth,
            e,
            f,
            stepE,
            stepF
        };

        bool IncreaseFBy()
        {
            int nextF = f + 1;
            int nextW = nextF;
            bool allocated = HandleL1Allocate(h, nextW, e, nextF, psumPingPangSplit, ifBytesPerElementGlb);
            if (allocated)
            {
                f = nextF;
                w = nextW;
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
        SegmentND psum, int iPp, int ofPp, int wPp, bool weightGroupOnly, WeightGroupHandler weightGroup)
    {
        int l1Pp = 0;
        List<int> l1Params = L1Search(glb, weight, psum);
        TileUtilities.GetSegmentStartEndLength(psum[2].Start, l1Params[2], psum[2].End);
        List<Segment1D> psumColSegments =
            TileUtilities.GetSegmentStartEndLength(psum[3].Start, l1Params[3], psum[3].End);
        List<Segment1D> psumBatchSegments = TileUtilities.GetSegmentStartEndLength(psum[0].Start, 1, psum[0].End);
        TileUtilities.GetSegmentStartEndLength(weight[2].Start, l1Params[4], weight[2].End);
        TileUtilities.GetSegmentStartEndLength(weight[3].Start, l1Params[5], weight[3].End);
        List<List<Segment1D>> l1McSeg = GetL1McSeg(ifmap, psum, l1Params[1], l1Params[0]);
        List<Segment1D> ocSegments = l1McSeg[0];
        List<Segment1D> icSegments = l1McSeg[1];
        Segment1D unitSegment = new Segment1D(..1, new Padding(0, 0));
        SegmentND weightRecSegment = new SegmentND(unitSegment, _broadcastW ? unitSegment : psum[0], weight[0], weight[1]);
        SegmentND ifmapRecSegment = new SegmentND(unitSegment, _broadcastIf ? unitSegment : psum[0], weight[1], ifmap[3]);
        SegmentND ofmapRecSegment = new SegmentND(unitSegment, psum[0], psum[1], psum[3]);
        foreach (Segment1D ocSegment in ocSegments)
        {
            foreach (Segment1D batchSegment in psumBatchSegments)
            {
                Segment1D unitSegment2 = unitSegment;
                Segment1D unitSegment3 = unitSegment;
                foreach (Segment1D colSegment in psumColSegments)
                {
                    Segment1D colSegmentCopy = colSegment;
                    new SegmentND(batchSegment, ocSegment, unitSegment2, colSegment);
                    SegmentND convSegment = new SegmentND(batchSegment, ocSegment, unitSegment3, colSegmentCopy);
                    bool isFirstLoop = true;
                    foreach (Segment1D icSegment in icSegments)
                    {
                        Segment1D current4;
                        Segment1D icSegmentCopy = (current4 = icSegment);
                        Segment1D firstKernelRowSegment = unitSegment;
                        Segment1D firstKernelColSegment = unitSegment;
                        Segment1D ifmapIcSegment = icSegmentCopy;
                        Segment1D ifmapBatchSegment = batchSegment;
                        Segment1D weightOcSegment = ocSegment;
                        SegmentND x = new SegmentND(_broadcastIf ? unitSegment : ifmapBatchSegment, ifmapIcSegment, unitSegment3,
                            colSegmentCopy);
                        SegmentND w = new SegmentND(ocSegment, current4, firstKernelRowSegment, firstKernelColSegment);
                        bool hasIfmapLoad = false;
                        SegmentND shiftedIfmap = TileUtilities.ShiftInputTensor(in x, in w, 1, 1, _strideH, _strideW,
                            _dilationH, _dilationW);
                        if (shiftedIfmap[0].Length > 0 && shiftedIfmap[1].Length > 0 && shiftedIfmap[2].Length > 0 &&
                            shiftedIfmap[3].Length > 0)
                        {
                            hasIfmapLoad = true;
                            if (!weightGroupOnly)
                            {
                                int ifmapStatCount = _g2LIfRec[iPp][0].Item2.Stat_cnt();
                                _g2LIfRec[iPp].RemoveAt(0);
                                List<CcrClr> ifmapCcrsToClr = new List<CcrClr>();
                                if (ifmapStatCount > 0)
                                {
                                    ifmapCcrsToClr.Add(new CcrClr(
                                        _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap, iPp))));
                                }

                                actionUpdater.UpdateG2LIf(shiftedIfmap, ifmap, _lif, iPp, null, ifmapCcrsToClr, 0, _inputType,
                                    ItemName.Ifmap, restored: false, 0, 0, 0, h2c: false, 1, _strideH);
                            }
                            else
                            {
                                _g2LIfRec[iPp].Add(new Tuple<SegmentND, TensorStat>(ifmapRecSegment,
                                    new TensorStat(isFirstSlice: false, isLastSlice: false)));
                            }
                        }

                        Segment1D kernelRowSegment = firstKernelRowSegment;
                        Segment1D kernelColSegment = firstKernelColSegment;
                        SegmentND w2 = new SegmentND(weightOcSegment, current4, kernelRowSegment, kernelColSegment);
                        SegmentND ifmapLoadSegment = TileUtilities.ShiftInputTensor(in x, in w2, 1, 1, _strideH, _strideW,
                            _dilationH, _dilationW);
                        int weightPasses = ((!(_weightType == DataTypes.Int16)) ? 1 : 2);
                        int inputPasses = ((!(_inputType == DataTypes.Int16)) ? 1 : 2);
                        for (int i = 0; i < weightPasses; i++)
                        {
                            for (int j = 0; j < inputPasses; j++)
                            {
                                if (!weightGroupOnly)
                                {
                                    int weightStatCount = _g2RWRec[wPp][0].Item2.Stat_cnt();
                                    _g2RWRec[wPp].RemoveAt(0);
                                    List<CcrClr> weightCcrsToClr = new List<CcrClr>();
                                    if (weightStatCount > 0)
                                    {
                                        weightCcrsToClr.Add(new CcrClr(
                                            _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Weight, wPp))));
                                    }

                                    actionUpdater.UpdateG2RWMatmul(w2, weight, weightGroup, _weightsShape[2], _lw, wPp,
                                        i, null, weightCcrsToClr,
                                        (!_broadcastW) ? (batchSegment.Start - psumBatchSegments[0].Start) : 0);
                                }
                                else
                                {
                                    if (batchSegment.Start == 0)
                                    {
                                        weightGroup.UpdateWeightGroup(w2);
                                    }

                                    _g2RWRec[wPp].Add(new Tuple<SegmentND, TensorStat>(weightRecSegment,
                                        new TensorStat(isFirstSlice: false, isLastSlice: false)));
                                }

                                if (!weightGroupOnly)
                                {
                                    actionUpdater.UpdateL2RIf(ifmapLoadSegment, shiftedIfmap, _strideH, _strideW,
                                        _weightsShape[3], _lif, 0, j,
                                        ((TensorConst)_mm[GNNEMatMul.DeqBBias]).Value.ToArray<byte>()[
                                            ifmapLoadSegment[0].Start], _inputType, h2c: false, 1);
                                }

                                bool releaseIf = false;
                                if (i == weightPasses - 1 && j == inputPasses - 1 && hasIfmapLoad)
                                {
                                    releaseIf = true;
                                    hasIfmapLoad = false;
                                }

                                bool loopStart = false;
                                if (isFirstLoop && i == 0 && j == 0)
                                {
                                    loopStart = true;
                                    isFirstLoop = false;
                                }

                                bool isLastKernelSlice = current4.End == _weightsShape[3] && kernelColSegment.End == weight[3].End &&
                                             kernelRowSegment.End == weight[2].End && i == weightPasses - 1 && j == inputPasses - 1;
                                DataType outputType = _outputType;
                                ACT0_OUTPUT_DEST act0OutputDest = ACT0_OUTPUT_DEST.dm;
                                if (!weightGroupOnly)
                                {
                                    int shift = ((TensorConst)_mm[GNNEMatMul.ShiftBits]).Value.ToScalar<int>();
                                    int setOfmapCcr = 0;
                                    int clearOfmapCcr = 0;
                                    if (isLastKernelSlice)
                                    {
                                        Tuple<SegmentND, TensorStat> l2GOfRecStat = _l2GOfRec[ofPp][0];
                                        _l2GOfRec[ofPp].RemoveAt(0);
                                        if (l2GOfRecStat.Item2.IsFirstSlice && !_ofmapRec[ofPp][0].Item2.IsFirstSlice)
                                        {
                                            clearOfmapCcr = 1;
                                        }

                                        if (l2GOfRecStat.Item2.IsLastSlice)
                                        {
                                            setOfmapCcr = 1;
                                        }
                                    }

                                    List<CcrSet> ofmapCcrsToSet = new List<CcrSet>();
                                    if (setOfmapCcr > 0)
                                    {
                                        ofmapCcrsToSet.Add(new CcrSet(
                                            _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, ofPp)), 1));
                                    }

                                    List<CcrClr> ofmapCcrsToClr = new List<CcrClr>();
                                    if (clearOfmapCcr > 0)
                                    {
                                        ofmapCcrsToClr.Add(new CcrClr(
                                            _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.OfmapFake, ofPp))));
                                    }

                                    int offsetAct = batchSegment.Start * glb.GlbMap[ItemName.Act].GlbNByte / _outputShape[1];
                                    act0OutputDest = ACT0_OUTPUT_DEST.dm;
                                    actionUpdater.UpdateR2LPsum(shift, convSegment, convSegment, psum, ofPp, l1Pp,
                                        act0OutputDest, releaseIf, Math.Max(i, j), TcuComputeMode.MatMul, loopStart,
                                        isLastKernelSlice, _strideH, _strideW, _weightsShape[2], _inputType, _weightType,
                                        outputType, _lact.CheckedDataType, ofmapCcrsToSet, ofmapCcrsToClr, offsetAct);
                                }
                                else if (isLastKernelSlice && act0OutputDest == ACT0_OUTPUT_DEST.dm)
                                {
                                    _l2GOfRec[ofPp].Add(new Tuple<SegmentND, TensorStat>(ofmapRecSegment,
                                        new TensorStat(isFirstSlice: false, isLastSlice: false)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private List<List<Segment1D>> GetL1McSeg(SegmentND ifmap, SegmentND psum, int mInloop, int cInloop)
    {
        List<Segment1D> ocSegments = new List<Segment1D>();
        List<Segment1D> icSegments = new List<Segment1D>();
        ocSegments = TileUtilities.GetSegmentStartEndLength(psum[1].Start, mInloop, psum[1].End);
        icSegments = TileUtilities.GetSegmentStartEndLength(ifmap[1].Start, cInloop, ifmap[1].End);
        return new List<List<Segment1D>> { ocSegments, icSegments };
    }

    private void UpdateCcrRecStat()
    {
        List<Tuple<SegmentND, TensorStat>>[] g2LIfRec = _g2LIfRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in g2LIfRec)
        {
            for (int j = 0; j < recList.Count; j++)
            {
                if (j == 0)
                {
                    recList[j].Item2.IsFirstSlice = true;
                }

                if (j == recList.Count - 1)
                {
                    recList[j].Item2.IsLastSlice = true;
                }

                if (j < recList.Count - 1 && recList[j].Item1 != recList[j + 1].Item1)
                {
                    recList[j].Item2.IsLastSlice = true;
                    recList[j + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        g2LIfRec = _g2RWRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in g2LIfRec)
        {
            for (int k = 0; k < recList.Count; k++)
            {
                if (k == 0)
                {
                    recList[k].Item2.IsFirstSlice = true;
                }

                if (k == recList.Count - 1)
                {
                    recList[k].Item2.IsLastSlice = true;
                }

                if (k < recList.Count - 1 && recList[k].Item1 != recList[k + 1].Item1)
                {
                    recList[k].Item2.IsLastSlice = true;
                    recList[k + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        g2LIfRec = _l2GOfRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in g2LIfRec)
        {
            for (int l = 0; l < recList.Count; l++)
            {
                if (l == 0)
                {
                    recList[l].Item2.IsFirstSlice = true;
                }

                if (l == recList.Count - 1)
                {
                    recList[l].Item2.IsLastSlice = true;
                }

                if (l < recList.Count - 1 && recList[l].Item1 != recList[l + 1].Item1)
                {
                    recList[l].Item2.IsLastSlice = true;
                    recList[l + 1].Item2.IsFirstSlice = true;
                }
            }
        }

        g2LIfRec = _ofmapRec;
        foreach (List<Tuple<SegmentND, TensorStat>> recList in g2LIfRec)
        {
            if (recList.Count > 0)
            {
                recList[0].Item2.IsFirstSlice = true;
                recList[recList.Count - 1].Item2.IsLastSlice = true;
            }
        }
    }

    private void ItemRecStatusInit()
    {
        for (int i = 0; i < _ofmapRec.Length; i++)
        {
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
        }
    }

    private void InitParameters(Call convNode, Call ld, Call st)
    {
        _mm = convNode;
        _lif = (Call)_mm[GNNEMatMul.InputB];
        _lw = (Call)_mm[GNNEMatMul.InputA];
        _lact = (Call)_mm[GNNEMatMul.Act];
        _lwQarg = (Call)_mm[GNNEMatMul.InputABias];
        _sof = st;
        _inputShape = new GNNEShape(_lif.CheckedShape[0].FixedValue, _lif.CheckedShape[1].FixedValue,
            _lif.CheckedShape[2].FixedValue, _lif.CheckedShape[3].FixedValue);
        _outputShape = new GNNEShape(_mm.CheckedShape[0].FixedValue, _mm.CheckedShape[1].FixedValue,
            _mm.CheckedShape[2].FixedValue, _mm.CheckedShape[3].FixedValue);
        _weightsShape = new GNNEShape(_lw.CheckedShape[0].FixedValue, _lw.CheckedShape[1].FixedValue,
            _lw.CheckedShape[2].FixedValue, _lw.CheckedShape[3].FixedValue);
        _strideH = 1;
        _strideW = 1;
        _dilationH = 1;
        _dilationW = 1;
        _inputType = _lif.CheckedDataType;
        _weightType = _lw.CheckedDataType;
        _weightGroup = new WeightGroupHandler(_weightType, _weightType);
        _outputType = _mm.CheckedDataType;
        _broadcastW = _weightsShape[1] == 1;
        _broadcastIf = _inputShape[1] == 1;
    }

    private TileMatMulGlb SearchGlbParameters()
    {
        int i = 1;
        int j = 1;
        int r = 1;
        int s = 1;
        int ofmapRows = 1;
        int ofmapCols = Math.Min(GNNEEnv.PuWidth, _outputShape[3]);
        int h = 1;
        int w = ofmapCols;
        int k = 1;
        AllocateResult allocateResult;
        for (; j < _weightsShape[3]; j++)
        {
            allocateResult = HandleAllocate(i, j + 1, h, w, r, s, k, ofmapRows, ofmapCols);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; k < GNNEEnv.PuWidth && k < _weightsShape[2]; k++)
        {
            allocateResult = HandleAllocate(i, j, h, w, r, s, k + 1, ofmapRows, ofmapCols);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        while (ofmapCols < _outputShape[3])
        {
            int nextOfmapCols = ofmapCols + 1;
            int nextWidth = nextOfmapCols;
            allocateResult = HandleAllocate(i, j, h, nextWidth, r, s, k, ofmapRows, nextOfmapCols);
            if (!allocateResult.IsOk)
            {
                break;
            }

            ofmapCols = nextOfmapCols;
            w = nextWidth;
        }

        for (; k < _weightsShape[2]; k++)
        {
            allocateResult = HandleAllocate(i, j, h, w, r, s, k + 1, ofmapRows, ofmapCols);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        for (; i < _outputShape[1]; i++)
        {
            allocateResult = HandleAllocate(i + 1, j, h, w, r, s, k, ofmapRows, ofmapCols);
            if (!allocateResult.IsOk)
            {
                break;
            }
        }

        if (k % GNNEEnv.PuWidth != 0 && k > GNNEEnv.PuWidth)
        {
            k = k / GNNEEnv.PuWidth * GNNEEnv.PuWidth;
        }

        allocateResult = HandleAllocate(i, j, h, w, r, s, k, ofmapRows, ofmapCols, isFinal: true);
        TileUtilities.Assert(allocateResult.IsOk, "allocation.IsOk",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileMatMul.cs",
            820);
        GNNEShape glbShape = new GNNEShape(i, k, ofmapRows, ofmapCols);
        return new TileMatMulGlb(allocateResult.GlbMap, allocateResult.Items, glbShape.Dims, GNNEEnv.NPingPongSplit,
            allocateResult.GlbMap[ItemName.Ifmap], allocateResult.GlbMap[ItemName.Ofmap]);
    }

    private AllocateResult HandleAllocate(int n, int c, int h, int w, int r, int s, int m, int e, int f,
        bool isFinal = false)
    {
        List<float> glbUsage = new List<float> { 0f, 0f };
        AllocateResult boxes = GetBoxes(n, c, h, w, r, s, m, e, f, glbUsage);
        AllocateResult allocateResult = SpaceSearcher.TryAllocate(new BoxPacker(16) { Boxes = boxes.Boxes });
        allocateResult.GlbMap = boxes.GlbMap;
        if (allocateResult.IsOk)
        {
            foreach (KeyValuePair<ItemName, TensorOnGlb> entry in boxes.GlbMap)
            {
                if (allocateResult.Items.ContainsKey(entry.Key))
                {
                    entry.Value.Mmu = allocateResult.Items[entry.Key];
                }
            }
        }

        return allocateResult;
    }

    private AllocateResult GetBoxes(int n, int c, int h, int w, int r, int s, int m, int e, int f, List<float> glbUsage)
    {
        List<BoxOnGlb> boxes = new List<BoxOnGlb>();
        Dictionary<ItemName, TensorOnGlb> glbMap = new Dictionary<ItemName, TensorOnGlb>();
        List<float> glbUsageBytes = new List<float> { 0f, 0f };
        int pingPongSplit = GNNEEnv.NPingPongSplit;
        if (n == _inputShape[0] && c == _inputShape[1] && h == _inputShape[2] && w == _inputShape[3])
        {
            pingPongSplit = 1;
        }

        TensorOnGlb ifmapTensor = new TensorOnGlb(new int[4] { _broadcastIf ? 1 : n, c, h, w }, _inputType, 0);
        int ifmapBytes = ifmapTensor.AllocatedBytes * pingPongSplit;
        glbUsageBytes[0] += ifmapBytes;
        ifmapBytes = TileUtilities.GetAlignedNum(ifmapBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        glbUsageBytes[1] += ifmapBytes;
        int weightOcCount = (_broadcastW ? m : (m * n));
        TensorOnGlb weightTensor = new TensorOnGlb(new int[4] { weightOcCount, c, r, s }, _weightType, 0);
        int weightBytes = SpaceSearcher.GetWeightSize(r, s, c, weightOcCount, TileUtilities.GetBytesPerElement(_weightType)) *
                   GNNEEnv.NPingPongSplit;
        weightTensor.AllocatedBytes = weightBytes / GNNEEnv.NPingPongSplit;
        glbUsageBytes[0] += weightBytes;
        weightBytes = TileUtilities.GetAlignedNum(weightBytes, GNNEEnv.WBankWidth * GNNEEnv.GlbBankWidth);
        glbUsageBytes[1] += weightBytes;
        TensorOnGlb ofmapTensor = new TensorOnGlb(new int[4] { n, m, e, f }, _outputType, 0);
        int ofmapBytes = ofmapTensor.AllocatedBytes * GNNEEnv.NPingPongSplit;
        glbUsageBytes[0] += ofmapBytes;
        ofmapBytes = TileUtilities.GetAlignedNum(ofmapBytes, GNNEEnv.OfmapBankWidth * GNNEEnv.GlbBankWidth);
        glbUsageBytes[1] += ofmapBytes;
        TensorOnGlb value = new TensorOnGlb(new int[4] { 1, _outputShape[1], _outputShape[2], GNNEEnv.ActNumPerChan },
            _lact.CheckedDataType, 0);
        int actSize = SpaceSearcher.GetActSize(_outputShape[1] * _outputShape[2], GNNEEnv.ActNumPerChan,
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
        glbMap.Add(ItemName.Act, value);
        if (_weightType == DataTypes.UInt8 || _weightType == DataTypes.Int16)
        {
            TensorOnGlb wQargTensor =
                new TensorOnGlb(new int[4] { 1, 1, _broadcastW ? 1 : _outputShape[1], _weightsShape[2] },
                    _lwQarg.CheckedDataType, 0);
            int wQargBytes = (_broadcastW ? 1 : _outputShape[1]) * SpaceSearcher.GetWQargSize(_weightsShape[2],
                TileUtilities.GetBytesPerElement(_lwQarg.CheckedDataType));
            glbUsageBytes[0] += wQargBytes;
            wQargBytes = TileUtilities.GetAlignedNum(wQargBytes, GNNEEnv.WQargBankWidth * GNNEEnv.GlbBankWidth);
            glbUsageBytes[1] += wQargBytes;
            boxes.Add(new BoxOnGlb(
                new int[2] { GNNEEnv.WQargBankWidth, wQargBytes / GNNEEnv.WQargBankWidth / GNNEEnv.GlbBankWidth },
                ItemName.WQarg));
            glbMap.Add(ItemName.WQarg, wQargTensor);
        }

        glbUsage[0] = glbUsageBytes[0] / (float)GNNEEnv.GlbSize;
        glbUsage[1] = glbUsageBytes[1] / (float)GNNEEnv.GlbSize;
        return new AllocateResult { Boxes = boxes, GlbMap = glbMap };
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Call call = (Call)__result["call"];
        Call ld = (Call)__result["ld"];
        Call st = (Call)__result["st"];
        return GetReplace(call, ld, st);
    }
}
