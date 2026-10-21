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
public class TileAct1 : RewriteRule<Pattern>
{
    private static int _count = -1;

    private DataType? _inputAType;

    private DataType? _inputBType;

    private DataType? _outputType;

    private Call? _act1;

    private Call? _lif1;

    private Call? _lif2;

    private Call? _lact;

    private Call? _sof;

    private GNNEShape? _inputAShape;

    private GNNEShape? _inputBShape;

    private GNNEShape? _outputShape;

    public override Pattern Pattern { get; } = FusionPattern.IsGNNEActivationFusion();

    private List<GnneAction> BuildSchedule(TiledGlb glb, Call act, Call ld, Call st, Nncase.TIR.Buffer ddrIf,
        Nncase.TIR.Buffer ddrOf, Nncase.TIR.Buffer ddrAct, Nncase.TIR.Buffer ddrIf2)
    {
        List<GnneAction> actions = new List<GnneAction>();
        GprHandler gpr = new GprHandler(GNNEEnv.GprNum);
        SsrHandler ssr = new SsrHandler(GNNEEnv.SsrNum);
        CcrHandler ccrHandler = new CcrHandler();
        GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, glb, ccrHandler, gpr, ssr);
        gnneActionUpdater.UpdateMmuConf();
        gnneActionUpdater.UpdateLoadAct(_lact, ddrAct, ItemName.MfuAct1);
        if (act[GNNEActivation.InputB] != None.Default)
        {
            GNNEShape minShape = new GNNEShape(0, 0, 0, 0);
            for (int i = 0; i < 4; i++)
            {
                minShape[i] = Math.Min(_inputAShape[i], _inputBShape[i]);
            }

            GNNEShape repeatA = new GNNEShape(0, 0, 0, 0);
            GNNEShape repeatB = new GNNEShape(0, 0, 0, 0);
            GNNEShape repeatOut = new GNNEShape(0, 0, 0, 0);
            for (int j = 0; j < 4; j++)
            {
                repeatA[j] = ((minShape[j] == 1) ? 1 : (_inputAShape[j] / minShape[j]));
                repeatB[j] = ((minShape[j] == 1) ? 1 : (_inputBShape[j] / minShape[j]));
                repeatOut[j] = ((minShape[j] == 1) ? 1 : (_outputShape[j] / minShape[j]));
            }

            foreach (Segment1D segmentN in TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[0], _outputShape[0]))
            {
                Segment1D inputASegmentN = segmentN / (repeatOut[0] / repeatA[0]);
                Segment1D inputBSegmentN = segmentN / (repeatOut[0] / repeatB[0]);
                if (_inputAShape[0] == 1)
                {
                    inputASegmentN = new Segment1D(..1, new Padding(0, 0));
                }

                if (_inputBShape[0] == 1)
                {
                    inputBSegmentN = new Segment1D(..1, new Padding(0, 0));
                }

                foreach (Segment1D segmentC in TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[1],
                             _outputShape[1]))
                {
                    Segment1D inputASegmentC = segmentC / (repeatOut[1] / repeatA[1]);
                    Segment1D inputBSegmentC = segmentC / (repeatOut[1] / repeatB[1]);
                    if (_inputAShape[1] == 1)
                    {
                        inputASegmentC = new Segment1D(..1, Padding.Zero());
                    }

                    if (_inputBShape[1] == 1)
                    {
                        inputBSegmentC = new Segment1D(..1, new Padding(0, 0));
                    }

                    foreach (Segment1D segmentH in TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[2],
                                 _outputShape[2]))
                    {
                        Segment1D inputASegmentH = segmentH / (repeatOut[2] / repeatA[2]);
                        Segment1D inputBSegmentH = segmentH / (repeatOut[2] / repeatB[2]);
                        if (_inputAShape[2] == 1)
                        {
                            inputASegmentH = new Segment1D(..1, Padding.Zero());
                        }

                        if (_inputBShape[2] == 1)
                        {
                            inputBSegmentH = new Segment1D(..1, new Padding(0, 0));
                        }

                        foreach (Segment1D segmentW in TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[3],
                                     _outputShape[3]))
                        {
                            Segment1D inputASegmentW = segmentW / (repeatOut[3] / repeatA[3]);
                            Segment1D inputBSegmentW = segmentW / (repeatOut[3] / repeatB[3]);
                            if (_inputAShape[3] == 1)
                            {
                                inputASegmentW = new Segment1D(..1, Padding.Zero());
                            }

                            if (_inputBShape[3] == 1)
                            {
                                inputBSegmentW = new Segment1D(..1, Padding.Zero());
                            }

                            SegmentND ofmapSegment = new SegmentND(segmentN.Start..segmentN.End, segmentC, segmentH, segmentW);
                            SegmentND ifmapASegment = new SegmentND(inputASegmentN, inputASegmentC, inputASegmentH, inputASegmentW);
                            SegmentND ifmapBSegment = new SegmentND(inputBSegmentN, inputBSegmentC, inputBSegmentH, inputBSegmentW);
                            int minChannels = Math.Min(inputASegmentC.Length, inputBSegmentC.Length);
                            int chunkChannels = (int)Math.Ceiling(1f * (float)minChannels / (float)glb.NPingPongSplit);
                            List<Segment1D> channelSegmentsA =
                                TileUtilities.GetSegmentStartEndLength(inputASegmentC.Start, chunkChannels * inputASegmentC.Length / minChannels,
                                    inputASegmentC.End);
                            List<Segment1D> channelSegmentsB =
                                TileUtilities.GetSegmentStartEndLength(inputBSegmentC.Start, chunkChannels * inputBSegmentC.Length / minChannels,
                                    inputBSegmentC.End);
                            List<Segment1D> channelSegmentsOut =
                                TileUtilities.GetSegmentStartEndLength(segmentC.Start, chunkChannels * segmentC.Length / minChannels,
                                    segmentC.End);
                            for (int k = 0; k < channelSegmentsOut.Count; k++)
                            {
                                Segment1D loadSegmentA = channelSegmentsA[k];
                                Segment1D loadSegmentB = channelSegmentsB[k];
                                Segment1D storeSegmentC = channelSegmentsOut[k];
                                SegmentND ifmapLoadA = new SegmentND(ifmapASegment[0], loadSegmentA, ifmapASegment[2],
                                    ifmapASegment[3]);
                                SegmentND ifmapLoadB = new SegmentND(ifmapBSegment[0], loadSegmentB, ifmapBSegment[2],
                                    ifmapBSegment[3]);
                                SegmentND ofmap = new SegmentND(ofmapSegment[0], storeSegmentC, ofmapSegment[2], ofmapSegment[3]);
                                List<CcrSet> ccrsToSet = new List<CcrSet>
                                {
                                    new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, k)), 1)
                                };
                                gnneActionUpdater.UpdateLoadIf(ifmapLoadA, _lif1, k, ddrIf, 0, null, ItemName.Ifmap,
                                    ccrsToSet);
                                List<CcrSet> ccrsToSet2 = new List<CcrSet>
                                {
                                    new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap2, k)), 1)
                                };
                                gnneActionUpdater.UpdateLoadIf(ifmapLoadB, _lif2, k, ddrIf2, 0, null, ItemName.Ifmap2,
                                    ccrsToSet2);
                                List<CcrSet> ccrsToSet3 = new List<CcrSet>
                                {
                                    new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, k)), 1)
                                };
                                List<CcrClr> ofmapCcrsToClr = new List<CcrClr>
                                {
                                    new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, k)))
                                };
                                ofmapCcrsToClr.Add(new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap2, k))));
                                List<int> src1Stride = new List<int>
                                {
                                    ifmapLoadA[1].Length, ifmapLoadA[2].Length, ifmapLoadA[3].Length
                                };
                                List<int> src2Stride = new List<int>
                                {
                                    ifmapLoadB[1].Length, ifmapLoadB[2].Length, ifmapLoadB[3].Length
                                };
                                gnneActionUpdater.UpdateMfuAct1(ifmapLoadA, ifmapLoadB, ofmap, ACT1_SOURCE_TYPE.l2,
                                    ACT1_SOURCE_TYPE.l2, _inputAType, _inputBType, _outputType,
                                    (_inputAType == DataTypes.Float16)
                                        ? new DeQuantizeParam(0, 1f)
                                        : ((TensorConst)act[GNNEActivation.DeqAParams]).Value
                                        .ToScalar<DeQuantizeParam>(),
                                    (_inputBType == DataTypes.Float16)
                                        ? new DeQuantizeParam(0, 1f)
                                        : ((TensorConst)act[GNNEActivation.DeqBParams]).Value
                                        .ToScalar<DeQuantizeParam>(),
                                    ((TensorConst)act[GNNEActivation.InAShiftBits]).Value.ToScalar<int>(),
                                    ((TensorConst)act[GNNEActivation.InBShiftBits]).Value.ToScalar<int>(),
                                    ((TensorConst)act[GNNEActivation.OutShiftBits]).Value.ToScalar<int>(),
                                    ((TensorConst)act[GNNEActivation.Is16Segments]).Value.ToScalar<bool>(), k,
                                    ccrsToSet3, ofmapCcrsToClr, 0, 0, 0, 0, ItemName.Ifmap2, ItemName.Ifmap, ItemName.MfuAct1,
                                    (((GNNEActivation)act.Target).Type == GnneActivationType.Mul)
                                        ? MFU_ACT1_FUNCTION.mul
                                        : MFU_ACT1_FUNCTION.add, ItemName.Ofmap, src1Stride, src2Stride);
                                List<CcrClr> ccrsToClr = new List<CcrClr>
                                {
                                    new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, k)))
                                };
                                gnneActionUpdater.UpdateStoreT(ofmap, _sof, k, ddrOf, 0, null, null, ccrsToClr);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            GNNEShape unitShape = new GNNEShape(0, 0, 0, 0);
            GNNEShape repeatShape = new GNNEShape(0, 0, 0, 0);
            for (int l = 0; l < 4; l++)
            {
                unitShape[l] = 1;
                repeatShape[l] = ((_inputAShape[l] == 1) ? 1 : (_outputShape[l] / _inputAShape[l]));
            }

            foreach (Segment1D segmentN2 in TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[0], _outputShape[0]))
            {
                Segment1D inputASegmentN2 = segmentN2 / (repeatShape[0] / unitShape[0]);
                if (_inputAShape[0] == 1)
                {
                    inputASegmentN2 = new Segment1D(0..1, Padding.Zero());
                }

                foreach (Segment1D segmentC2 in TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[1],
                             _outputShape[1]))
                {
                    Segment1D inputASegmentC2 = segmentC2 / (repeatShape[1] / unitShape[1]);
                    if (_inputAShape[1] == 1)
                    {
                        inputASegmentC2 = new Segment1D(0..1, Padding.Zero());
                    }

                    foreach (Segment1D segmentH2 in TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[2],
                                 _outputShape[2]))
                    {
                        Segment1D inputASegmentH2 = segmentH2 / (repeatShape[2] / unitShape[2]);
                        if (_inputAShape[2] == 1)
                        {
                            inputASegmentH2 = new Segment1D(0..1, Padding.Zero());
                        }

                        foreach (Segment1D segmentW2 in TileUtilities.GetSegmentStartEndLength(0, glb.LastOutShape[3],
                                     _outputShape[3]))
                        {
                            Segment1D inputASegmentW2 = segmentW2 / (repeatShape[3] / unitShape[3]);
                            if (_inputAShape[3] == 1)
                            {
                                inputASegmentW2 = new Segment1D(0..1, Padding.Zero());
                            }

                            SegmentND ofmapSegment2 = new SegmentND(segmentN2, segmentC2, segmentH2, segmentW2);
                            SegmentND ifmapSegment2 = new SegmentND(inputASegmentN2, inputASegmentC2, inputASegmentH2, inputASegmentW2);
                            int chunkSize =
                                (int)Math.Ceiling(1.0 * (double)inputASegmentC2.Length / (double)glb.NPingPongSplit);
                            List<Segment1D> channelSegments2 =
                                TileUtilities.GetSegmentStartEndLength(inputASegmentC2.Start, chunkSize, inputASegmentC2.End);
                            for (int m = 0; m < channelSegments2.Count; m++)
                            {
                                Segment1D loadSegment2 = channelSegments2[m];
                                SegmentND ifmapLoad2 = new SegmentND(ifmapSegment2[0], loadSegment2, ifmapSegment2[2],
                                    ifmapSegment2[3]);
                                Segment1D emptySegment = new Segment1D(0..0, Padding.Zero());
                                SegmentND ifmap = new SegmentND(emptySegment, emptySegment, emptySegment, emptySegment);
                                SegmentND ofmap2 = new SegmentND(ofmapSegment2[0], loadSegment2, ofmapSegment2[2],
                                    ofmapSegment2[3]);
                                List<CcrSet> ccrsToSet4 = new List<CcrSet>
                                {
                                    new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, m)), 1)
                                };
                                List<int> stridesD = new List<int>
                                {
                                    glb.GlbMap[ItemName.Ifmap].Dimensions[1],
                                    glb.GlbMap[ItemName.Ifmap].Dimensions[2],
                                    glb.GlbMap[ItemName.Ifmap].Dimensions[3]
                                };
                                gnneActionUpdater.UpdateLoadIf(ifmapLoad2, _lif1, m, ddrIf, 0, stridesD, ItemName.Ifmap,
                                    ccrsToSet4);
                                List<CcrSet> ccrsToSet5 = new List<CcrSet>
                                {
                                    new CcrSet(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, m)), 1)
                                };
                                List<CcrClr> ccrsToClr2 = new List<CcrClr>
                                {
                                    new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ifmap, m)))
                                };
                                gnneActionUpdater.UpdateMfuAct1(ifmapLoad2, ifmap, ofmap2, ACT1_SOURCE_TYPE.l2,
                                    ACT1_SOURCE_TYPE.l2, _inputAType, _inputAType, _outputType,
                                    (_inputAType == DataTypes.Float16)
                                        ? new DeQuantizeParam(0, 1f)
                                        : ((TensorConst)act[GNNEActivation.DeqAParams]).Value
                                        .ToScalar<DeQuantizeParam>(),
                                    (_inputBType == DataTypes.Float16)
                                        ? new DeQuantizeParam(0, 1f)
                                        : ((TensorConst)act[GNNEActivation.DeqBParams]).Value
                                        .ToScalar<DeQuantizeParam>(),
                                    ((TensorConst)act[GNNEActivation.InAShiftBits]).Value.ToScalar<int>(),
                                    ((TensorConst)act[GNNEActivation.InBShiftBits]).Value.ToScalar<int>(),
                                    ((TensorConst)act[GNNEActivation.OutShiftBits]).Value.ToScalar<int>(),
                                    ((TensorConst)act[GNNEActivation.Is16Segments]).Value.ToScalar<bool>(), m,
                                    ccrsToSet5, ccrsToClr2, 0, 0, 0, 0, ItemName.Ifmap2, ItemName.Ifmap,
                                    ItemName.MfuAct1,
                                    (((GNNEActivation)act.Target).Type == GnneActivationType.Mul)
                                        ? MFU_ACT1_FUNCTION.mul
                                        : MFU_ACT1_FUNCTION.add);
                                List<CcrClr> ccrsToClr3 = new List<CcrClr>
                                {
                                    new CcrClr(ccrHandler.GetCcrItem(ccrHandler.GetName(ItemName.Ofmap, m)))
                                };
                                gnneActionUpdater.UpdateStoreT(ofmap2, _sof, m, ddrOf, 0, null, null, ccrsToClr3);
                            }
                        }
                    }
                }
            }
        }

        TileUtilities.Assert(ccrHandler.CcrSanityCheck(), "ccrHandler.CcrSanityCheck()",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileAct1.cs",
            249);
        return actions;
    }

    private TileAct1Glb SearchGlbParameters(Call ld, Call st, Call act)
    {
        GNNEShape scaleA;
        GNNEShape scaleB;
        GNNEShape scaleOut;
        GNNEShape o;
        GNNEShape a;
        GNNEShape b;
        if (act[GNNEActivation.InputB] != None.Default)
        {
            GNNEShape minShape = new GNNEShape(0, 0, 0, 0);
            for (int i = 0; i < 4; i++)
            {
                minShape[i] = Math.Min(_inputAShape[i], _inputBShape[i]);
            }

            scaleA = new GNNEShape(0, 0, 0, 0);
            scaleB = new GNNEShape(0, 0, 0, 0);
            scaleOut = new GNNEShape(0, 0, 0, 0);
            for (int j = 0; j < 4; j++)
            {
                scaleA[j] = ((minShape[j] == 1) ? 1 : (_inputAShape[j] / minShape[j]));
                scaleB[j] = ((minShape[j] == 1) ? 1 : (_inputBShape[j] / minShape[j]));
                scaleOut[j] = ((minShape[j] == 1) ? 1 : (_outputShape[j] / minShape[j]));
            }

            int pingPongSplit = GNNEEnv.NPingPongSplit;
            int outN = scaleOut[0];
            int outC = Math.Min(scaleOut[1] * pingPongSplit, _outputShape[1]);
            if (outC == 1)
            {
                pingPongSplit = 1;
            }

            int outH = scaleOut[2];
            int outW = scaleOut[3];
            o = new GNNEShape(outN, outC, outH, outW);
            a = new GNNEShape(0, 0, 0, 0);
            b = new GNNEShape(0, 0, 0, 0);
            CalcInputShape();
            AllocateResult allocateResult;
            while (o[3] < _outputShape[3])
            {
                o[3] = Math.Min(o[3] + scaleOut[3], _outputShape[3]);
                CalcInputShape();
                allocateResult = HandleAllocate(o, a, b, pingPongSplit);
                if (allocateResult.IsOk)
                {
                    outW = o[3];
                    continue;
                }

                o[3] = outW;
                break;
            }

            while (o[2] < _outputShape[2])
            {
                o[2] = Math.Min(o[2] + scaleOut[2] * pingPongSplit, _outputShape[2]);
                CalcInputShape();
                allocateResult = HandleAllocate(o, a, b, pingPongSplit);
                if (allocateResult.IsOk)
                {
                    outH = o[2];
                    continue;
                }

                o[2] = outH;
                break;
            }

            while (o[1] < _outputShape[1])
            {
                o[1] = Math.Min(o[1] + scaleOut[1], _outputShape[1]);
                CalcInputShape();
                allocateResult = HandleAllocate(o, a, b, pingPongSplit);
                if (allocateResult.IsOk)
                {
                    outC = o[1];
                    continue;
                }

                o[1] = outC;
                break;
            }

            while (o[0] < _outputShape[0])
            {
                o[0] = Math.Min(o[0] + scaleOut[0], _outputShape[0]);
                CalcInputShape();
                allocateResult = HandleAllocate(o, a, b, pingPongSplit);
                if (allocateResult.IsOk)
                {
                    outN = o[0];
                    continue;
                }

                o[0] = outN;
                break;
            }

            CalcInputShape();
            allocateResult = HandleAllocate(o, a, b, pingPongSplit);
            TileUtilities.Assert(allocateResult.IsOk, "allocation.IsOk",
                "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileAct1.cs",
                368);
            GNNEShape glbShape = new GNNEShape(outN, outC, outH, outW);
            return new TileAct1Glb(allocateResult.GlbMap, allocateResult.Items, glbShape.Dims, GNNEEnv.NPingPongSplit,
                allocateResult.GlbMap[ItemName.Ifmap], allocateResult.GlbMap[ItemName.Ofmap]);
        }

        int k = 1;
        int l = 1;
        int ifmapRows = 1;
        int w = _inputAShape[3];
        int ofmapRows = _outputShape[2] / _inputAShape[2];
        int ofmapCols = _outputShape[3];
        int nPingPongSplit = GNNEEnv.NPingPongSplit;
        AllocateResult allocateResult2 = HandleAllocate(k, l, ifmapRows, w, ofmapRows, ofmapCols, nPingPongSplit);
        if (!allocateResult2.IsOk)
        {
            nPingPongSplit = 1;
        }

        for (; l < _inputAShape[1]; l++)
        {
            allocateResult2 = HandleAllocate(k, l + 1, ifmapRows, w, ofmapRows, ofmapCols, nPingPongSplit);
            if (!allocateResult2.IsOk)
            {
                break;
            }
        }

        while (ifmapRows < _inputAShape[2])
        {
            int nextIfmapRows = ifmapRows + 1;
            int e = nextIfmapRows * (_outputShape[2] / _inputAShape[2]);
            allocateResult2 = HandleAllocate(k, l, nextIfmapRows, w, e, ofmapCols, nPingPongSplit);
            if (!allocateResult2.IsOk)
            {
                break;
            }

            ifmapRows++;
            ofmapRows = ifmapRows * (_outputShape[2] / _inputAShape[2]);
        }

        for (; k < _inputAShape[0]; k++)
        {
            allocateResult2 = HandleAllocate(k + 1, l, ifmapRows, w, ofmapRows, ofmapCols, nPingPongSplit);
            if (!allocateResult2.IsOk)
            {
                break;
            }
        }

        allocateResult2 = HandleAllocate(k, l, ifmapRows, w, ofmapRows, ofmapCols, nPingPongSplit, isFinal: true);
        TileUtilities.Assert(allocateResult2.IsOk, "allocation.IsOk",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileAct1.cs",
            437);
        GNNEShape glbShape2 = new GNNEShape(k, l, ofmapRows, ofmapCols);
        return new TileAct1Glb(allocateResult2.GlbMap, allocateResult2.Items, glbShape2.Dims, GNNEEnv.NPingPongSplit,
            allocateResult2.GlbMap[ItemName.Ifmap], allocateResult2.GlbMap[ItemName.Ofmap]);

        void CalcInputShape()
        {
            for (int m = 0; m < 4; m++)
            {
                a[m] = Math.Min(o[m] / scaleOut[m] * scaleA[m], _inputAShape[m]);
                b[m] = Math.Min(o[m] / scaleOut[m] * scaleB[m], _inputBShape[m]);
            }
        }
    }

    private AllocateResult HandleAllocate(int n, int c, int h, int w, int e, int f, int nPingPongSplit,
        bool isFinal = false)
    {
        BoxPacker boxPacker = new BoxPacker(16);
        Dictionary<ItemName, TensorOnGlb> glbMap = new Dictionary<ItemName, TensorOnGlb>();
        int channelsPerPass = (int)Math.Ceiling(1f * (float)c / (float)nPingPongSplit);
        TensorOnGlb ifmapTensor = new TensorOnGlb(new int[4] { n, channelsPerPass, h, w }, _inputAType, 0);
        int ifmapBytes = ifmapTensor.AllocatedBytes * nPingPongSplit;
        ifmapBytes = TileUtilities.GetAlignedNum(ifmapBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        TensorOnGlb ofmapTensor = new TensorOnGlb(new int[4] { n, channelsPerPass, e, f }, _outputType, 0);
        int ofmapBytes = ofmapTensor.AllocatedBytes * nPingPongSplit;
        ofmapBytes = TileUtilities.GetAlignedNum(ofmapBytes, GNNEEnv.OfmapBankWidth * GNNEEnv.GlbBankWidth);
        bool is16Segments = ((TensorConst)_act1[GNNEActivation.Is16Segments]).Value.ToScalar<bool>();
        int c2 = (is16Segments ? 1 : _inputAShape[1]);
        int nActParam = (is16Segments ? 49 : GNNEEnv.ActNumPerChan);
        int actSize = SpaceSearcher.GetActSize(c2, nActParam, TileUtilities.GetBytesPerElement(_lact.CheckedDataType));
        actSize = TileUtilities.GetAlignedNum(actSize, GNNEEnv.ActBankWidth * GNNEEnv.GlbBankWidth);
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
        boxPacker.Boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.ActBankWidth, actSize / GNNEEnv.ActBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.MfuAct1));
        AllocateResult allocateResult = SpaceSearcher.TryAllocate(boxPacker);
        glbMap.Add(ItemName.Ifmap, ifmapTensor);
        glbMap.Add(ItemName.Ofmap, ofmapTensor);
        glbMap.Add(ItemName.MfuAct1, new TensorOnGlb(new int[4] { 0, 0, 0, 0 }, _lact.CheckedDataType, 0));
        if (allocateResult.IsOk)
        {
            foreach (KeyValuePair<ItemName, TensorOnGlb> entry in glbMap)
            {
                entry.Value.Mmu = allocateResult.Items[entry.Key];
            }
        }

        return new AllocateResult { IsOk = allocateResult.IsOk, Items = allocateResult.Items, GlbMap = glbMap };
    }

    private AllocateResult HandleAllocate(GNNEShape o, GNNEShape a, GNNEShape b, int nPingPongSplit,
        bool isFinal = false)
    {
        BoxPacker boxPacker = new BoxPacker(16);
        Dictionary<ItemName, TensorOnGlb> glbMap = new Dictionary<ItemName, TensorOnGlb>();
        int minChannels = Math.Min(a[1], b[1]);
        int channelsPerPass = (int)Math.Ceiling(1f * (float)minChannels / (float)nPingPongSplit);
        TensorOnGlb ifmapATensor = new TensorOnGlb(new int[4] { a[0], a[1] / minChannels * channelsPerPass, a[2], a[3] }, _inputAType, 0);
        int ifmapABytes = ifmapATensor.AllocatedBytes * nPingPongSplit;
        ifmapABytes = TileUtilities.GetAlignedNum(ifmapABytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        TensorOnGlb ifmapBTensor = new TensorOnGlb(new int[4] { b[0], b[1] / minChannels * channelsPerPass, b[2], b[3] }, _inputBType, 0);
        int ifmapBBytes = ifmapBTensor.AllocatedBytes * nPingPongSplit;
        ifmapBBytes = TileUtilities.GetAlignedNum(ifmapBBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        TensorOnGlb ofmapTensor = new TensorOnGlb(new int[4] { o[0], o[1] / minChannels * channelsPerPass, o[2], o[3] }, _outputType, 0);
        int ofmapBytes = ofmapTensor.AllocatedBytes * nPingPongSplit;
        ofmapBytes = TileUtilities.GetAlignedNum(ofmapBytes, GNNEEnv.OfmapBankWidth * GNNEEnv.GlbBankWidth);
        bool is16Segments = ((TensorConst)_act1[GNNEActivation.Is16Segments]).Value.ToScalar<bool>();
        int c = (is16Segments ? 1 : _inputAShape[1]);
        int nActParam = (is16Segments ? 49 : GNNEEnv.ActNumPerChan);
        int actSize = SpaceSearcher.GetActSize(c, nActParam, TileUtilities.GetBytesPerElement(_lact.CheckedDataType));
        actSize = TileUtilities.GetAlignedNum(actSize, GNNEEnv.ActBankWidth * GNNEEnv.GlbBankWidth);
        int basementSize = SpaceSearcher.GetBasementSize();
        basementSize = TileUtilities.GetAlignedNum(basementSize, GNNEEnv.GlbWidth * GNNEEnv.GlbBankWidth);
        boxPacker.Boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.GlbWidth, basementSize / GNNEEnv.GlbWidth / GNNEEnv.GlbBankWidth },
            ItemName.Basement));
        boxPacker.Boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.IfmapBankWidth, ifmapABytes / GNNEEnv.IfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.Ifmap));
        boxPacker.Boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.IfmapBankWidth, ifmapBBytes / GNNEEnv.IfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.Ifmap2));
        boxPacker.Boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.OfmapBankWidth, ofmapBytes / GNNEEnv.OfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.Ofmap));
        boxPacker.Boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.ActBankWidth, actSize / GNNEEnv.ActBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.MfuAct1));
        AllocateResult allocateResult = SpaceSearcher.TryAllocate(boxPacker);
        glbMap.Add(ItemName.Ifmap, ifmapATensor);
        glbMap.Add(ItemName.Ifmap2, ifmapBTensor);
        glbMap.Add(ItemName.Ofmap, ofmapTensor);
        glbMap.Add(ItemName.MfuAct1, new TensorOnGlb(new int[4] { 0, 0, 0, 0 }, _lact.CheckedDataType, 0));
        if (allocateResult.IsOk)
        {
            foreach (KeyValuePair<ItemName, TensorOnGlb> entry in glbMap)
            {
                entry.Value.Mmu = allocateResult.Items[entry.Key];
            }
        }

        return new AllocateResult { IsOk = allocateResult.IsOk, Items = allocateResult.Items, GlbMap = glbMap };
    }

    private AllocateResult HandleAllocate(int n, int c, int h, int w, int e, int f, DataType inType, DataType outType,
        bool isFinal = false)
    {
        BoxPacker boxPacker = new BoxPacker(16);
        Dictionary<ItemName, TensorOnGlb> glbMap = new Dictionary<ItemName, TensorOnGlb>();
        int alignmentFactor = ((TileUtilities.GetBytesPerElement(inType) == 1) ? 32 : 16);
        int alignedNum = TileUtilities.GetAlignedNum(w, alignmentFactor);
        TensorOnGlb ifmapTensor = new TensorOnGlb(new int[4] { n, c, h, alignedNum }, inType, 0);
        int ifmapBytes = ifmapTensor.GlbNByte * GNNEEnv.NPingPongSplit;
        ifmapBytes = TileUtilities.GetAlignedNum(ifmapBytes, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        alignmentFactor = ((TileUtilities.GetBytesPerElement(outType) == 1) ? 32 : 16);
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

    private void InitParameters(Call act, Call ld, Call st)
    {
        _act1 = act;
        _inputAType = _act1[GNNEActivation.InputA].CheckedDataType;
        _inputBType = ((_act1[GNNEActivation.InputB] != None.Default)
            ? _act1[GNNEActivation.InputB].CheckedDataType
            : DataTypes.Float16);
        _outputType = act.CheckedDataType;
        _lif1 = ld;
        if (_act1[GNNEActivation.InputB] != None.Default)
        {
            _lif2 = (Call)_act1[GNNEActivation.InputB];
        }

        _lact = (Call)_act1[GNNEActivation.Act];
        _sof = st;
        _inputAShape = new GNNEShape(_lif1.CheckedShape[0].FixedValue, _lif1.CheckedShape[1].FixedValue,
            _lif1.CheckedShape[2].FixedValue, _lif1.CheckedShape[3].FixedValue);
        _inputBShape = ((_act1[GNNEActivation.InputB] != None.Default)
            ? new GNNEShape(_lif2.CheckedShape[0].FixedValue, _lif2.CheckedShape[1].FixedValue,
                _lif2.CheckedShape[2].FixedValue, _lif2.CheckedShape[3].FixedValue)
            : _inputAShape);
        _outputShape = new GNNEShape(st.CheckedShape[0].FixedValue, st.CheckedShape[1].FixedValue,
            st.CheckedShape[2].FixedValue, st.CheckedShape[3].FixedValue);
    }

    private PrimFunction GetReplace(Call call, GNNEActivation callOp, Call ld, Call st)
    {
        _count++;
        InitParameters(call, ld, st);
        TileAct1Glb glb = SearchGlbParameters(ld, st, call);
        int[] inputAShape = ld.Arguments[0].CheckedShape.ToValueArray();
        int[] inputBShape = inputAShape;
        if (call[GNNEActivation.InputB] != None.Default)
        {
            inputBShape = call[GNNEActivation.InputB].CheckedShape.ToValueArray();
        }

        T.CreateBuffer(new TensorType(ld[GNNELoad.Input].CheckedDataType, inputAShape), MemoryLocation.Input,
            out Nncase.TIR.Buffer buffer, "var ddrIf");
        Nncase.TIR.Buffer buffer2 = buffer;
        List<Nncase.TIR.Buffer> buffers = new List<Nncase.TIR.Buffer> { buffer };
        if (call[GNNEActivation.InputB] != None.Default)
        {
            if (!(((Call)call[GNNEActivation.InputB])[GNNELoad.Input] is TensorConst))
            {
                T.CreateBuffer(
                    new TensorType(((Call)call[GNNEActivation.InputB])[GNNELoad.Input].CheckedDataType, inputBShape),
                    MemoryLocation.Input, out buffer2, "ddrIf2");
                buffers.Add(buffer2);
            }
            else
            {
                T.AttachBuffer((TensorConst)((Call)call[GNNEActivation.InputB])[GNNELoad.Input], out buffer2, "ddrIf2");
            }
        }

        T.CreateBuffer(new TensorType(st.CheckedDataType, st.CheckedShape), MemoryLocation.Output,
            out Nncase.TIR.Buffer buffer3, "var ddrOf");
        buffers.Add(buffer3);
        T.AttachBuffer((TensorConst)((Call)call[GNNEActivation.Act])[GNNELoadW.Input], out Nncase.TIR.Buffer buffer4,
            "var ddrAct");
        List<GnneAction> actions = BuildSchedule(glb, call, ld, st, buffer, buffer3, buffer4, buffer2);
        ActionToInstruct actionToInstruct = new ActionToInstruct();
        return T.PrimFunc($"TileAct1_{_count}", K230RtModule.Kind, buffers.ToArray())
            .Body(actionToInstruct.Instructions(actions), I.END(GP_REGISTER.x0, 0)).Build();
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Call call = (Call)__result["call"];
        GNNEActivation callOp = (GNNEActivation)__result["callOp"];
        Call ld = (Call)__result["ld"];
        Call st = (Call)__result["st"];
        return GetReplace(call, callOp, ld, st);
    }
}
