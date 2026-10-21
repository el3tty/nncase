// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.K230;
using Nncase.PatternMatch;
using Nncase.Runtime.K230;
using Nncase.TIR;
using Nncase.TIR.Instructions;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class TileLSTM : RewriteRule<Pattern>
{
    private static int _count = -1;

    private readonly List<WeightGroupHandler> _weightGroups = new List<WeightGroupHandler>(2);

    private CcrHandler _ccrHandler = new CcrHandler();

    private GprHandler _gpr = new GprHandler();

    private SsrHandler _ssr = new SsrHandler();

    private GNNEShape? _inputShape;

    private GNNEShape? _wXcShape;

    private GNNEShape? _wXcShapeReshape;

    private GNNEShape? _actXcShape;

    private GNNEShape? _wRcShape;

    private GNNEShape? _wRcShapeReshape;

    private GNNEShape? _actRcShape;

    private GNNEShape? _initHShape;

    private GNNEShape? _initCShape;

    private GNNEShape? _outputShape;

    private GNNEShape? _outputHShape;

    private GNNEShape? _outputCShape;

    private Call? _lIf;

    private Call? _lWXc;

    private Call? _lActXc;

    private Call? _lWRc;

    private Call? _lActRc0;

    private Call? _lActRc1;

    private Call? _lInitH;

    private Call? _lInitC;

    private Call? _lSegFt;

    private Call? _lSegGt;

    private Call? _of;

    private Call? _ofH;

    private Call? _ofC;

    private Call? _lwXcQarg;

    private Call? _lwRcQarg;

    private Call? _lwBinAct;

    private Call? _lwBinQAct;

    private DataType? _inputType;

    private DataType? _wXcType;

    private DataType? _actXcType;

    private DataType? _wRcType;

    private DataType? _actRcType;

    private DataType? _initHType;

    private DataType? _initCType;

    private DataType? _outputType;

    private DataType? _outputHType;

    private DataType? _outputCType;

    private Call? _oldLstm;

    public override Pattern Pattern { get; } =
        Nncase.PatternMatch.Utility.IsFusion("k230", new GNNELSTMFusion().Pattern);

    private PrimFunction GetReplace(Expr output, Call midCall, IReadOnlyList<Expr> midCallParams)
    {
        _count++;
        Call call = (Call)midCallParams[0];
        Call call2 = (Call)((Nncase.IR.Tuple)output)[0];
        int outputCount = ((TensorConst)midCall[GNNELSTM.OutputSize]).Value.ToScalar<int>();
        Call call3 = ((outputCount >= 2) ? ((Call)((Nncase.IR.Tuple)output)[1]) : null);
        Call call4 = ((outputCount >= 3) ? ((Call)((Nncase.IR.Tuple)output)[2]) : null);
        Nncase.TIR.Buffer buffer = null;
        Nncase.TIR.Buffer buffer2 = null;
        List<Nncase.TIR.Buffer> buffers = new List<Nncase.TIR.Buffer>();
        InitParameters(midCall, call, call2, call3, call4);
        TileLSTMGlb glb = SearchGlbParameters();
        int[] inputShape = _oldLstm[GNNELSTM.Input].CheckedShape.ToValueArray();
        T.CreateBuffer(new TensorType(call[GNNELoad.Input].CheckedDataType, inputShape), MemoryLocation.Input,
            out Nncase.TIR.Buffer buffer3, "var ddrIf");
        buffers.Add(buffer3);
        T.AttachBuffer((TensorConst)_lWXc[GNNELoadW.Input], out Nncase.TIR.Buffer buffer4, "var ddrLWXC");
        T.AttachBuffer((TensorConst)_lWRc[GNNELoadW.Input], out Nncase.TIR.Buffer buffer5, "var ddrLWRC");
        T.AttachBuffer((TensorConst)_lwXcQarg[GNNELoadW.Input], out Nncase.TIR.Buffer buffer6, "var ddrLWXCQARG");
        T.AttachBuffer((TensorConst)_lwRcQarg[GNNELoadW.Input], out Nncase.TIR.Buffer buffer7, "var ddrLWRCQARG");
        T.AttachBuffer((TensorConst)_lActXc[GNNELoadW.Input], out Nncase.TIR.Buffer buffer8, "var ddrLACTXC");
        T.AttachBuffer((TensorConst)_lActRc0[GNNELoadW.Input], out Nncase.TIR.Buffer buffer9, "var ddrLACTRC0");
        T.AttachBuffer((TensorConst)_lActRc1[GNNELoadW.Input], out Nncase.TIR.Buffer buffer10, "var ddrLACTRC1");
        T.AttachBuffer((TensorConst)_lSegFt[GNNELoadW.Input], out Nncase.TIR.Buffer buffer11, "var ddrLSEGFT");
        T.AttachBuffer((TensorConst)_lSegGt[GNNELoadW.Input], out Nncase.TIR.Buffer buffer12, "var ddrLSEGGT");
        T.AttachBuffer((TensorConst)_lwBinAct[GNNELoadW.Input], out Nncase.TIR.Buffer buffer13, "var ddrLWBINACT");
        T.AttachBuffer((TensorConst)_lwBinQAct[GNNELoadW.Input], out Nncase.TIR.Buffer buffer14, "var ddrLWBINQACT");
        Nncase.TIR.Buffer buffer15;
        if (_lInitH[GNNELoad.Input] is TensorConst)
        {
            T.AttachBuffer((TensorConst)_lInitH[GNNELoad.Input], out buffer15, "ddrLInitH");
        }
        else
        {
            T.CreateBuffer(
                new TensorType(_lInitH[GNNELoad.Input].CheckedDataType, _lInitH[GNNELoad.Input].CheckedShape),
                MemoryLocation.Input, out buffer15, "ddrLInitH");
            buffers.Add(buffer15);
        }

        Nncase.TIR.Buffer buffer16;
        if (_lInitC[GNNELoad.Input] is TensorConst)
        {
            T.AttachBuffer((TensorConst)_lInitC[GNNELoad.Input], out buffer16, "ddrLInitC");
        }
        else
        {
            T.CreateBuffer(
                new TensorType(_lInitC[GNNELoad.Input].CheckedDataType, _lInitC[GNNELoad.Input].CheckedShape),
                MemoryLocation.Input, out buffer16, "ddrLInitC");
            buffers.Add(buffer16);
        }

        T.CreateBuffer(new TensorType(call2.CheckedDataType, call2.CheckedShape), MemoryLocation.Output,
            out Nncase.TIR.Buffer buffer17, "var ddrOf");
        buffers.Add(buffer17);
        if ((object)call3 != null)
        {
            T.CreateBuffer(new TensorType(call3.CheckedDataType, call3.CheckedShape), MemoryLocation.Output, out buffer,
                "ddrOfH");
            buffers.Add(buffer);
        }

        if ((object)call4 != null)
        {
            T.CreateBuffer(new TensorType(call4.CheckedDataType, call4.CheckedShape), MemoryLocation.Output,
                out buffer2, "ddrOfC");
            buffers.Add(buffer2);
        }

        List<GnneAction> actions = BuildSchedule(glb, midCall, call, call2, call3, call4, buffer3, buffer4, buffer5,
            buffer6, buffer7, buffer8, buffer9, buffer10, buffer11, buffer12, buffer13, buffer14, buffer15, buffer16,
            buffer17, buffer, buffer2);
        Span<byte> bytesBuffer = ((TensorConst)((Call)buffer4.MemSpan.Start)[DDrOf.Input]).Value.BytesBuffer;
        ArrangeWeights(_wXcType, _wXcShape, bytesBuffer, _weightGroups[0], _wXcShapeReshape);
        Span<byte> bytesBuffer2 = ((TensorConst)((Call)buffer5.MemSpan.Start)[DDrOf.Input]).Value.BytesBuffer;
        ArrangeWeights(_wRcType, _wRcShape, bytesBuffer2, _weightGroups[1], _wRcShapeReshape);
        ActionToInstruct actionToInstruct = new ActionToInstruct();
        return T.PrimFunc($"TileLSTM_{_count}", K230RtModule.Kind, buffers.ToArray())
            .Body(actionToInstruct.Instructions(actions), I.END(GP_REGISTER.x0, 0)).Build();
    }

    private void ArrangeWeights(DataType weightsType, GNNEShape weightsShape, Span<byte> oldWeights,
        WeightGroupHandler weightGroup, GNNEShape weightsReshape)
    {
        int bytesPerElement = TileUtilities.GetBytesPerElement(weightsType);
        List<SegmentND> weightSlices = weightGroup.WeightGroupSlice();
        byte[] arranged = new byte[oldWeights.Length];
        int srcBase = 0;
        int writeIndex = 0;
        for (int i = 0; i < weightsShape[1]; i++)
        {
            int sliceOffset = 0;
            foreach (SegmentND weightSlice in weightSlices)
            {
                TileUtilities.Assert(sliceOffset == weightGroup.WeightGroupOffset(weightSlice) * bytesPerElement,
                    "offset == weightGroup.WeightGroupOffset(slice) * bytesPerElement",
                    "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLSTM.cs",
                    165);
                for (int j = 0; j < bytesPerElement; j++)
                {
                    for (int k = 0; k < weightSlice[0].Length; k++)
                    {
                        for (int l = 0; l < weightSlice[2].Length; l++)
                        {
                            for (int m = 0; m < weightSlice[3].Length; m++)
                            {
                                for (int n = 0; n < weightSlice[1].Length; n++)
                                {
                                    int srcIndex = (srcBase +
                                                (((k + weightSlice[0].Start) * weightsReshape[1] + n + weightSlice[1].Start) *
                                                    weightsReshape[2] + l + weightSlice[2].Start) * weightsReshape[3] + m +
                                                weightSlice[3].Start) * bytesPerElement;
                                    arranged[writeIndex++] = oldWeights[srcIndex + j];
                                    sliceOffset++;
                                }
                            }
                        }
                    }
                }
            }

            srcBase += sliceOffset / bytesPerElement;
        }

        arranged.CopyTo(oldWeights);
    }

    private TileLSTMGlb SearchGlbParameters()
    {
        int nPingPongSplit = 1;
        AllocateResult allocateResult = HandleAllocate(nPingPongSplit, isFinal: true);
        TileUtilities.Assert(allocateResult.IsOk, "allocation.IsOk",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLSTM.cs",
            201);
        GNNEShape glbShape = new GNNEShape(1, 1, 1, _outputShape[3]);
        return new TileLSTMGlb(allocateResult.GlbMap, allocateResult.Items, glbShape.Dims, nPingPongSplit,
            allocateResult.GlbMap[ItemName.Ifmap], allocateResult.GlbMap[ItemName.Ofmap]);
    }

    private List<GnneAction> BuildSchedule(TiledGlb glb, Call call, Call ld, Call st, Call stH, Call stC,
        Nncase.TIR.Buffer ddrIf, Nncase.TIR.Buffer ddrLWXC, Nncase.TIR.Buffer ddrLWRC, Nncase.TIR.Buffer ddrLWXCQARG,
        Nncase.TIR.Buffer ddrLWRCQARG, Nncase.TIR.Buffer ddrLACTXC, Nncase.TIR.Buffer ddrLACTRC0,
        Nncase.TIR.Buffer ddrLACTRC1, Nncase.TIR.Buffer ddrLSEGFT, Nncase.TIR.Buffer ddrLSEGGT,
        Nncase.TIR.Buffer ddrLWBINACT, Nncase.TIR.Buffer ddrLWBINQACT, Nncase.TIR.Buffer ddrLInitH,
        Nncase.TIR.Buffer ddrLInitC, Nncase.TIR.Buffer ddrOf, Nncase.TIR.Buffer ddrOfH, Nncase.TIR.Buffer ddrOfC)
    {
        List<GnneAction> actions = new List<GnneAction>();
        _gpr = new GprHandler(GNNEEnv.GprNum);
        _ssr = new SsrHandler(GNNEEnv.SsrNum);
        _ccrHandler = new CcrHandler();
        GnneActionUpdater gnneActionUpdater = new GnneActionUpdater(actions, glb, _ccrHandler, _gpr, _ssr);
        gnneActionUpdater.UpdateMmuConf();
        int nPingPongSplit = glb.NPingPongSplit;
        int[] ifmapCcrValue = new int[1];
        int[] initHCcrValue = new int[1];
        SegmentND inputFeatureSegment = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
            new Segment1D(.._inputShape[3], new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
            new Segment1D(..1, new Padding(0, 0)));
        SegmentND gateOutputSegment = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
            new Segment1D(..(_outputShape[3] * 4), new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
            new Segment1D(..1, new Padding(0, 0)));
        SegmentND weight = new SegmentND(gateOutputSegment[1], inputFeatureSegment[1], new Segment1D(..1, new Padding(0, 0)),
            new Segment1D(..1, new Padding(0, 0)));
        BuildL1Schedule(isInitH: false, gnneActionUpdater, glb, inputFeatureSegment, weight, gateOutputSegment, 0, 8, 0,
            weightGroupOnly: true, _weightGroups[0], ifmapCcrValue, ddrIf, ddrLWXC);
        SegmentND ori = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
            new Segment1D(.._outputShape[3], new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
            new Segment1D(..1, new Padding(0, 0)));
        TileUtilities.GlbTensorIndexShift(in ori, in ori);
        weight = new SegmentND(gateOutputSegment[1], ori[1], new Segment1D(..1, new Padding(0, 0)),
            new Segment1D(..1, new Padding(0, 0)));
        BuildL1Schedule(isInitH: true, gnneActionUpdater, glb, ori, weight, gateOutputSegment, 0, 9, 0, weightGroupOnly: true,
            _weightGroups[1], initHCcrValue, ddrIf, ddrLWRC);
        if (_wXcType == DataTypes.UInt8 || _wXcType == DataTypes.Int16)
        {
            for (int i = 0; i < _outputShape[1]; i++)
            {
                gnneActionUpdater.UpdateLoadWQarg(_lwXcQarg, _weightGroups[0], ddrLWXCQARG, _outputShape[3] * 4 * i,
                    null, null, ItemName.LstmWXcQarg, _outputShape[3] * 4 * i);
            }
        }

        if (_wRcType == DataTypes.UInt8 || _wRcType == DataTypes.Int16)
        {
            for (int j = 0; j < _outputShape[1]; j++)
            {
                gnneActionUpdater.UpdateLoadWQarg(_lwRcQarg, _weightGroups[1], ddrLWRCQARG, _outputShape[3] * 4 * j,
                    null, null, ItemName.LstmWRcQarg, _outputShape[3] * 4 * j);
            }
        }

        gnneActionUpdater.UpdateLoadAct(_lActXc, ddrLACTXC, ItemName.LstmActXc);
        gnneActionUpdater.UpdateLoadAct(_lActRc0, ddrLACTRC0, ItemName.LstmActRc);
        gnneActionUpdater.UpdateLoadAct(_lActRc1, ddrLACTRC1, ItemName.LstmActRc,
            glb.GlbMap[ItemName.LstmActRc].AllocatedBytes);
        gnneActionUpdater.UpdateLoadAct(_lSegFt, ddrLSEGFT, ItemName.LstmSegFittingParamFt);
        gnneActionUpdater.UpdateLoadAct(_lSegGt, ddrLSEGGT, ItemName.LstmSegFittingParamGt);
        gnneActionUpdater.UpdateLoadAct(_lwBinAct, ddrLWBINACT, ItemName.LstmBinAct);
        gnneActionUpdater.UpdateLoadAct(_lwBinQAct, ddrLWBINQACT, ItemName.LstmBinQAct);
        int ifmapIndex = 0;
        List<int> timeSteps = new List<int>();
        for (int k = 0; k < _inputShape[1]; k++)
        {
            timeSteps.Add(k);
        }

        if (((GNNELSTM)_oldLstm.Target).Direction == LSTMDirection.Reverse)
        {
            timeSteps.Reverse();
        }

        for (int l = 0; l < _outputShape[1]; l++)
        {
            if (l == 1)
            {
                timeSteps.Reverse();
            }

            for (int m = 0; m < _inputShape[2]; m++)
            {
                for (int n = 0; n != timeSteps.Count; n++)
                {
                    int timeStep = timeSteps[n];
                    int l2 = ((n != timeSteps.Count - 1) ? timeSteps[n + 1] : 0);
                    if (timeStep == timeSteps[0])
                    {
                        SegmentND ifmapLoadSegment = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                            new Segment1D(timeStep..(timeStep + 1), new Padding(0, 0)),
                            new Segment1D(m..(m + 1), new Padding(0, 0)),
                            new Segment1D(.._inputShape[3], new Padding(0, 0)));
                        List<CcrSet> ccrsToSet = new List<CcrSet>
                        {
                            new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap, ifmapIndex)), ifmapCcrValue[0])
                        };
                        List<int> stridesD = new List<int>
                        {
                            ifmapLoadSegment[1].Length, ifmapLoadSegment[2].Length, ifmapLoadSegment[3].Length
                        };
                        List<CcrClr> ccrsToClr = new List<CcrClr>();
                        if (timeStep != timeSteps[0])
                        {
                            _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.IfmapFake, ifmapIndex));
                        }

                        gnneActionUpdater.UpdateLoadIf(ifmapLoadSegment, _lIf, ifmapIndex, ddrIf, 0, stridesD, ItemName.Ifmap,
                            ccrsToSet, ccrsToClr);
                    }

                    SegmentND initHSegment = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(l..(l + 1), new Padding(0, 0)), new Segment1D(m..(m + 1), new Padding(0, 0)),
                        new Segment1D(.._outputShape[3], new Padding(0, 0)));
                    SegmentND initCSegment = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(l..(l + 1), new Padding(0, 0)), new Segment1D(m..(m + 1), new Padding(0, 0)),
                        new Segment1D(.._outputShape[3], new Padding(0, 0)));
                    List<CcrSet> initHCcrsToSet = new List<CcrSet>
                    {
                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.LstmOfH)), 0)
                    };
                    List<CcrSet> initCCcrsToSet = new List<CcrSet>
                    {
                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.LstmOfC)), 0)
                    };
                    if (timeStep == timeSteps[0])
                    {
                        initHCcrsToSet[0].Value = initHCcrValue[0];
                        initCCcrsToSet[0].Value = 1;
                        List<int> stridesD2 = new List<int>
                        {
                            initHSegment[1].Length, initHSegment[2].Length, initHSegment[3].Length
                        };
                        gnneActionUpdater.UpdateLoadIf(initHSegment, _lInitH, 0, ddrLInitH, 0, stridesD2,
                            ItemName.LstmOfH, initHCcrsToSet);
                        stridesD2 = new List<int> { initCSegment[1].Length, initCSegment[2].Length, initCSegment[3].Length };
                        gnneActionUpdater.UpdateLoadIf(initCSegment, _lInitC, 0, ddrLInitC, 0, stridesD2,
                            ItemName.LstmOfC, initCCcrsToSet);
                    }

                    SegmentND loopInputSegment = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(.._inputShape[3], new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)));
                    SegmentND loopGateSegment = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..(_outputShape[3] * 4), new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)));
                    SegmentND weight2 = new SegmentND(loopGateSegment[1], loopInputSegment[1],
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)));
                    BuildL1Schedule(isInitH: false, gnneActionUpdater, glb, loopInputSegment, weight2, loopGateSegment, ifmapIndex, 8, 0,
                        weightGroupOnly: false, _weightGroups[0], ifmapCcrValue, ddrIf, ddrLWXC, timeStep == timeSteps[0], l,
                        timeStep == timeSteps[timeSteps.Count - 1], l2, m);
                    SegmentND ori2 = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(.._outputShape[3], new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)));
                    SegmentND ori3 = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(.._outputShape[3], new Padding(0, 0)),
                        new Segment1D(l..(l + 1), new Padding(0, 0)), new Segment1D(m..(m + 1), new Padding(0, 0)));
                    SegmentND oriShift2 = TileUtilities.GlbTensorIndexShift(in ori2, in ori2);
                    SegmentND oriShift3 = TileUtilities.GlbTensorIndexShift(in ori3, in ori3);
                    weight2 = new SegmentND(loopGateSegment[1], ori2[1], new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)));
                    BuildL1Schedule(isInitH: true, gnneActionUpdater, glb, ori2, weight2, loopGateSegment, 0, 9, 0,
                        weightGroupOnly: false, _weightGroups[1], initHCcrValue, ddrIf, ddrLWRC, timeStep == timeSteps[0], l);
                    List<CcrSet> ccrsToSet2 = new List<CcrSet>
                    {
                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 10)), 4)
                    };
                    List<CcrClr> ofmapCcrsToClrA = new List<CcrClr>
                    {
                        new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 8)))
                    };
                    ofmapCcrsToClrA.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 9))));
                    int offsetS = 0;
                    int allocatedBytes = glb.GlbMap[ItemName.Ofmap].AllocatedBytes;
                    int offsetD = 0;
                    int offsetAct = 0;
                    SegmentND mfuSegmentA = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(loopGateSegment[1].Start..loopGateSegment[1].End, new Padding(0, 0)));
                    List<int> src1Stride = new List<int>
                    {
                        mfuSegmentA[1].Length, mfuSegmentA[2].Length, mfuSegmentA[3].Length
                    };
                    List<int> src2Stride = new List<int>
                    {
                        mfuSegmentA[1].Length, mfuSegmentA[2].Length, mfuSegmentA[3].Length
                    };
                    List<int> dstStride = new List<int>
                    {
                        mfuSegmentA[1].Length, mfuSegmentA[2].Length, mfuSegmentA[3].Length
                    };
                    gnneActionUpdater.UpdateMfuAct1(mfuSegmentA, mfuSegmentA, mfuSegmentA, ACT1_SOURCE_TYPE.l2,
                        ACT1_SOURCE_TYPE.l2, DataTypes.Float16, DataTypes.Float16, DataTypes.Float16,
                        new DeQuantizeParam(0, 1f), new DeQuantizeParam(0, 1f), 0, 0, 0, is16Segments: false, 0,
                        ccrsToSet2, ofmapCcrsToClrA, offsetS, allocatedBytes, offsetD, offsetAct, ItemName.Ofmap, ItemName.Ofmap,
                        ItemName.LstmBinAct, MFU_ACT1_FUNCTION.add, ItemName.Ofmap, src1Stride, src2Stride, dstStride,
                        isByChannel: false);
                    List<CcrSet> ccrsToSet3 = new List<CcrSet>
                    {
                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 6)), 1)
                    };
                    List<CcrClr> ccrsToClr2 = new List<CcrClr>
                    {
                        new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 10)))
                    };
                    SegmentND gateSliceA = new SegmentND(loopGateSegment);
                    gateSliceA[1] = new Segment1D((_outputShape[3] * 2)..(_outputShape[3] * 3), new Padding(0, 0));
                    Segment1D emptySegmentA = new Segment1D(..0, new Padding(0, 0));
                    SegmentND emptyNdA = new SegmentND(emptySegmentA, emptySegmentA, emptySegmentA, emptySegmentA);
                    SegmentND srcCopyA = new SegmentND(ori2);
                    SegmentND gateVecA = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(gateSliceA[1].Start..gateSliceA[1].End, new Padding(0, 0)));
                    SegmentND srcVecA = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(srcCopyA[1].Start..srcCopyA[1].End, new Padding(0, 0)));
                    int addr =
                        new TensorOnGlb(
                            new int[4]
                            {
                                loopGateSegment[0].Length, loopGateSegment[1].Length, loopGateSegment[2].Length,
                                loopGateSegment[3].Length
                            }, DataTypes.Float16, 0).GetAddr(gateSliceA[0].Start, gateSliceA[1].Start,
                            gateSliceA[2].Start,
                            gateSliceA[3].Start, 0, 1);
                    int offsetS2 = 0;
                    int offsetD2 = glb.GlbMap[ItemName.Ofmap].AllocatedBytes +
                                   glb.GlbMap[ItemName.Ofmap].GlbNByte / 4 * 2;
                    int offsetAct2 = 0;
                    List<int> src1Stride2 = new List<int>
                    {
                        gateVecA[1].Length, gateVecA[2].Length, gateVecA[3].Length
                    };
                    List<int> src2Stride2 = new List<int>
                    {
                        emptyNdA[1].Length, emptyNdA[2].Length, emptyNdA[3].Length
                    };
                    List<int> dstStride2 = new List<int>
                    {
                        srcVecA[1].Length, srcVecA[2].Length, srcVecA[3].Length
                    };
                    gnneActionUpdater.UpdateMfuAct1(gateVecA, emptyNdA, srcVecA, ACT1_SOURCE_TYPE.l2,
                        ACT1_SOURCE_TYPE.l2, DataTypes.Float16, DataTypes.Float16, DataTypes.Float16,
                        new DeQuantizeParam(0, 1f), new DeQuantizeParam(0, 1f), 0, 0, 0, is16Segments: true, 0,
                        ccrsToSet3, ccrsToClr2, addr, offsetS2, offsetD2, offsetAct2, ItemName.Ofmap, ItemName.Ofmap,
                        ItemName.LstmSegFittingParamFt, MFU_ACT1_FUNCTION.add, ItemName.Ofmap, src1Stride2, src2Stride2,
                        dstStride2);
                    List<CcrClr> ofmapCcrsToClrB = new List<CcrClr> { new CcrClr(initCCcrsToSet[0].Ccr) };
                    ofmapCcrsToClrB.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 6))));
                    List<CcrSet> ccrsToSet4 = new List<CcrSet>
                    {
                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 2)), 1)
                    };
                    int offsetS3 = 0;
                    int offsetS4 = glb.GlbMap[ItemName.Ofmap].AllocatedBytes +
                                   glb.GlbMap[ItemName.Ofmap].GlbNByte / 4 * 2;
                    int offsetD3 = glb.GlbMap[ItemName.Ofmap].GlbNByte / 4 * 2;
                    int offsetAct3 = 0;
                    SegmentND mfuSegmentB = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(oriShift3[1].Start..oriShift3[1].End, new Padding(0, 0)));
                    List<int> src1Stride3 = new List<int>
                    {
                        mfuSegmentB[1].Length, mfuSegmentB[2].Length, mfuSegmentB[3].Length
                    };
                    List<int> src2Stride3 = new List<int>
                    {
                        mfuSegmentB[1].Length, mfuSegmentB[2].Length, mfuSegmentB[3].Length
                    };
                    List<int> dstStride3 = new List<int>
                    {
                        mfuSegmentB[1].Length, mfuSegmentB[2].Length, mfuSegmentB[3].Length
                    };
                    gnneActionUpdater.UpdateMfuAct1(mfuSegmentB, mfuSegmentB, mfuSegmentB, ACT1_SOURCE_TYPE.l2,
                        ACT1_SOURCE_TYPE.l2, _initCType, DataTypes.Float16, DataTypes.Float16,
                        new DeQuantizeParam(0, 1f), new DeQuantizeParam(0, 1f), 0, 0, 0, is16Segments: false, 0,
                        ccrsToSet4, ofmapCcrsToClrB, offsetS3, offsetS4, offsetD3, offsetAct3, ItemName.Ofmap, ItemName.LstmOfC,
                        ItemName.LstmBinAct, MFU_ACT1_FUNCTION.mul, ItemName.Ofmap, src1Stride3, src2Stride3,
                        dstStride3, isByChannel: false);
                    List<CcrSet> ccrsToSet5 = new List<CcrSet>
                    {
                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 4)), 1)
                    };
                    List<CcrClr> ccrsToClr3 = new List<CcrClr>
                    {
                        new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 10)))
                    };
                    SegmentND gateSliceB = new SegmentND(loopGateSegment);
                    _ = _outputShape[3];
                    gateSliceB[1] = new Segment1D(0.._outputShape[3], new Padding(0, 0));
                    Segment1D emptySegmentB = new Segment1D(..0, new Padding(0, 0));
                    SegmentND emptyNdB = new SegmentND(emptySegmentB, emptySegmentB, emptySegmentB, emptySegmentB);
                    SegmentND srcCopyB = new SegmentND(ori2);
                    SegmentND gateVecB = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(gateSliceB[1].Start..gateSliceB[1].End, new Padding(0, 0)));
                    SegmentND srcVecB = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(srcCopyB[1].Start..srcCopyB[1].End, new Padding(0, 0)));
                    int addr2 =
                        new TensorOnGlb(
                            new int[4]
                            {
                                loopGateSegment[0].Length, loopGateSegment[1].Length, loopGateSegment[2].Length,
                                loopGateSegment[3].Length
                            }, DataTypes.Float16, 0).GetAddr(gateSliceB[0].Start, gateSliceB[1].Start,
                            gateSliceB[2].Start,
                            gateSliceB[3].Start, 0, 1);
                    int offsetS5 = 0;
                    int allocatedBytes2 = glb.GlbMap[ItemName.Ofmap].AllocatedBytes;
                    _ = glb.GlbMap[ItemName.Ofmap].GlbNByte / 4;
                    int offsetD4 = allocatedBytes2 + 0;
                    int offsetAct4 = 0;
                    List<int> src1Stride4 = new List<int>
                    {
                        gateVecB[1].Length, gateVecB[2].Length, gateVecB[3].Length
                    };
                    List<int> src2Stride4 = new List<int>
                    {
                        emptyNdB[1].Length, emptyNdB[2].Length, emptyNdB[3].Length
                    };
                    List<int> dstStride4 = new List<int>
                    {
                        srcVecB[1].Length, srcVecB[2].Length, srcVecB[3].Length
                    };
                    gnneActionUpdater.UpdateMfuAct1(gateVecB, emptyNdB, srcVecB, ACT1_SOURCE_TYPE.l2,
                        ACT1_SOURCE_TYPE.l2, DataTypes.Float16, DataTypes.Float16, DataTypes.Float16,
                        new DeQuantizeParam(0, 1f), new DeQuantizeParam(0, 1f), 0, 0, 0, is16Segments: true, 0,
                        ccrsToSet5, ccrsToClr3, addr2, offsetS5, offsetD4, offsetAct4, ItemName.Ofmap, ItemName.Ofmap,
                        ItemName.LstmSegFittingParamFt, MFU_ACT1_FUNCTION.add, ItemName.Ofmap, src1Stride4, src2Stride4,
                        dstStride4);
                    List<CcrSet> ccrsToSet6 = new List<CcrSet>
                    {
                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 7)), 1)
                    };
                    List<CcrClr> ccrsToClr4 = new List<CcrClr>
                    {
                        new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 10)))
                    };
                    SegmentND gateSliceC = new SegmentND(loopGateSegment);
                    gateSliceC[1] = new Segment1D((_outputShape[3] * 3)..(_outputShape[3] * 4), new Padding(0, 0));
                    Segment1D emptySegmentC = new Segment1D(..0, new Padding(0, 0));
                    SegmentND emptyNdC = new SegmentND(emptySegmentC, emptySegmentC, emptySegmentC, emptySegmentC);
                    SegmentND srcCopyC = new SegmentND(ori2);
                    SegmentND gateVecC = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(gateSliceC[1].Start..gateSliceC[1].End, new Padding(0, 0)));
                    SegmentND srcVecC = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(srcCopyC[1].Start..srcCopyC[1].End, new Padding(0, 0)));
                    int addr3 =
                        new TensorOnGlb(
                            new int[4]
                            {
                                loopGateSegment[0].Length, loopGateSegment[1].Length, loopGateSegment[2].Length,
                                loopGateSegment[3].Length
                            }, DataTypes.Float16, 0).GetAddr(gateSliceC[0].Start, gateSliceC[1].Start,
                            gateSliceC[2].Start,
                            gateSliceC[3].Start, 0, 1);
                    int offsetS6 = 0;
                    int offsetD5 = glb.GlbMap[ItemName.Ofmap].AllocatedBytes +
                                   glb.GlbMap[ItemName.Ofmap].GlbNByte / 4 * 3;
                    int offsetAct5 = 0;
                    List<int> src1Stride5 = new List<int>
                    {
                        gateVecC[1].Length, gateVecC[2].Length, gateVecC[3].Length
                    };
                    List<int> src2Stride5 = new List<int>
                    {
                        emptyNdC[1].Length, emptyNdC[2].Length, emptyNdC[3].Length
                    };
                    List<int> dstStride5 = new List<int>
                    {
                        srcVecC[1].Length, srcVecC[2].Length, srcVecC[3].Length
                    };
                    gnneActionUpdater.UpdateMfuAct1(gateVecC, emptyNdC, srcVecC, ACT1_SOURCE_TYPE.l2,
                        ACT1_SOURCE_TYPE.l2, DataTypes.Float16, DataTypes.Float16, DataTypes.Float16,
                        new DeQuantizeParam(0, 1f), new DeQuantizeParam(0, 1f), 0, 0, 0, is16Segments: true, 0,
                        ccrsToSet6, ccrsToClr4, addr3, offsetS6, offsetD5, offsetAct5, ItemName.Ofmap, ItemName.Ofmap,
                        ItemName.LstmSegFittingParamGt, MFU_ACT1_FUNCTION.add, ItemName.Ofmap, src1Stride5, src2Stride5,
                        dstStride5);
                    List<CcrClr> ofmapCcrsToClrC = new List<CcrClr>
                    {
                        new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 4)))
                    };
                    ofmapCcrsToClrC.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 7))));
                    List<CcrSet> ccrsToSet7 = new List<CcrSet>
                    {
                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 0)), 1)
                    };
                    int allocatedBytes3 = glb.GlbMap[ItemName.Ofmap].AllocatedBytes;
                    _ = glb.GlbMap[ItemName.Ofmap].GlbNByte / 4;
                    int offsetS7 = allocatedBytes3 + 0;
                    int offsetS8 = glb.GlbMap[ItemName.Ofmap].AllocatedBytes +
                                   glb.GlbMap[ItemName.Ofmap].GlbNByte / 4 * 3;
                    _ = glb.GlbMap[ItemName.Ofmap].AllocatedBytes;
                    int offsetD6 = 0;
                    int offsetAct6 = 0;
                    SegmentND mfuSegmentC = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(oriShift3[1].Start..oriShift3[1].End, new Padding(0, 0)));
                    List<int> src1Stride6 = new List<int>
                    {
                        mfuSegmentC[1].Length, mfuSegmentC[2].Length, mfuSegmentC[3].Length
                    };
                    List<int> src2Stride6 = new List<int>
                    {
                        mfuSegmentC[1].Length, mfuSegmentC[2].Length, mfuSegmentC[3].Length
                    };
                    List<int> dstStride6 = new List<int>
                    {
                        mfuSegmentC[1].Length, mfuSegmentC[2].Length, mfuSegmentC[3].Length
                    };
                    gnneActionUpdater.UpdateMfuAct1(mfuSegmentC, mfuSegmentC, mfuSegmentC, ACT1_SOURCE_TYPE.l2,
                        ACT1_SOURCE_TYPE.l2, DataTypes.Float16, DataTypes.Float16, DataTypes.Float16,
                        new DeQuantizeParam(0, 1f), new DeQuantizeParam(0, 1f), 0, 0, 0, is16Segments: false, 0,
                        ccrsToSet7, ofmapCcrsToClrC, offsetS7, offsetS8, offsetD6, offsetAct6, ItemName.Ofmap, ItemName.Ofmap,
                        ItemName.LstmBinAct, MFU_ACT1_FUNCTION.mul, ItemName.Ofmap, src1Stride6, src2Stride6,
                        dstStride6, isByChannel: false);
                    List<CcrClr> ofmapCcrsToClrD = new List<CcrClr>
                    {
                        new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 2)))
                    };
                    ofmapCcrsToClrD.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 0))));
                    List<CcrSet> ofmapCcrsToSetD = new List<CcrSet>();
                    ofmapCcrsToSetD.Add(
                        new CcrSet(initCCcrsToSet[0].Ccr, (timeStep == timeSteps[timeSteps.Count - 1] && (object)_ofC == null) ? 1 : 2));
                    List<CcrSet> ccrsToSet8 = ofmapCcrsToSetD;
                    int offsetS9 = glb.GlbMap[ItemName.Ofmap].GlbNByte / 4 * 2;
                    _ = glb.GlbMap[ItemName.Ofmap].GlbNByte / 4;
                    int offsetS10 = 0;
                    int offsetD7 = 0;
                    int offsetAct7 = 0;
                    SegmentND mfuSegmentD = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(oriShift3[1].Start..oriShift3[1].End, new Padding(0, 0)));
                    List<int> src1Stride7 = new List<int>
                    {
                        mfuSegmentD[1].Length, mfuSegmentD[2].Length, mfuSegmentD[3].Length
                    };
                    List<int> src2Stride7 = new List<int>
                    {
                        mfuSegmentD[1].Length, mfuSegmentD[2].Length, mfuSegmentD[3].Length
                    };
                    List<int> dstStride7 = new List<int>
                    {
                        mfuSegmentD[1].Length, mfuSegmentD[2].Length, mfuSegmentD[3].Length
                    };
                    gnneActionUpdater.UpdateMfuAct1(mfuSegmentD, mfuSegmentD, mfuSegmentD, ACT1_SOURCE_TYPE.l2,
                        ACT1_SOURCE_TYPE.l2, DataTypes.Float16, DataTypes.Float16, DataTypes.Float16,
                        new DeQuantizeParam(0, 1f), new DeQuantizeParam(0, 1f), 0, 0, 0, is16Segments: false, 0,
                        ccrsToSet8, ofmapCcrsToClrD, offsetS9, offsetS10, offsetD7, offsetAct7, ItemName.Ofmap, ItemName.Ofmap,
                        ItemName.LstmBinAct, MFU_ACT1_FUNCTION.add, ItemName.LstmOfC, src1Stride7, src2Stride7,
                        dstStride7, isByChannel: false);
                    List<CcrSet> ccrsToSet9 = new List<CcrSet>
                    {
                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 5)), 1)
                    };
                    List<CcrClr> ccrsToClr5 = new List<CcrClr>
                    {
                        new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 10)))
                    };
                    SegmentND gateSliceD = new SegmentND(loopGateSegment);
                    gateSliceD[1] = new Segment1D(_outputShape[3]..(_outputShape[3] * 2), new Padding(0, 0));
                    Segment1D emptySegmentD = new Segment1D(..0, new Padding(0, 0));
                    SegmentND emptyNdD = new SegmentND(emptySegmentD, emptySegmentD, emptySegmentD, emptySegmentD);
                    SegmentND srcCopyD = new SegmentND(ori2);
                    SegmentND gateVecD = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(gateSliceD[1].Start..gateSliceD[1].End, new Padding(0, 0)));
                    SegmentND srcVecD = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(srcCopyD[1].Start..srcCopyD[1].End, new Padding(0, 0)));
                    int addr4 =
                        new TensorOnGlb(
                            new int[4]
                            {
                                loopGateSegment[0].Length, loopGateSegment[1].Length, loopGateSegment[2].Length,
                                loopGateSegment[3].Length
                            }, DataTypes.Float16, 0).GetAddr(gateSliceD[0].Start, gateSliceD[1].Start,
                            gateSliceD[2].Start,
                            gateSliceD[3].Start, 0, 1);
                    int offsetS11 = 0;
                    int offsetD8 = glb.GlbMap[ItemName.Ofmap].AllocatedBytes + glb.GlbMap[ItemName.Ofmap].GlbNByte / 4;
                    int offsetAct8 = 0;
                    List<int> src1Stride8 = new List<int>
                    {
                        gateVecD[1].Length, gateVecD[2].Length, gateVecD[3].Length
                    };
                    List<int> src2Stride8 = new List<int>
                    {
                        emptyNdD[1].Length, emptyNdD[2].Length, emptyNdD[3].Length
                    };
                    List<int> dstStride8 = new List<int>
                    {
                        srcVecD[1].Length, srcVecD[2].Length, srcVecD[3].Length
                    };
                    gnneActionUpdater.UpdateMfuAct1(gateVecD, emptyNdD, srcVecD, ACT1_SOURCE_TYPE.l2,
                        ACT1_SOURCE_TYPE.l2, DataTypes.Float16, DataTypes.Float16, DataTypes.Float16,
                        new DeQuantizeParam(0, 1f), new DeQuantizeParam(0, 1f), 0, 0, 0, is16Segments: true, 0,
                        ccrsToSet9, ccrsToClr5, addr4, offsetS11, offsetD8, offsetAct8, ItemName.Ofmap, ItemName.Ofmap,
                        ItemName.LstmSegFittingParamFt, MFU_ACT1_FUNCTION.add, ItemName.Ofmap, src1Stride8, src2Stride8,
                        dstStride8);
                    List<CcrSet> ccrsToSet10 = new List<CcrSet>
                    {
                        new CcrSet(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 3)), 1)
                    };
                    List<CcrClr> ccrsToClr6 = new List<CcrClr> { new CcrClr(initCCcrsToSet[0].Ccr) };
                    SegmentND gateSliceE = new SegmentND(oriShift2);
                    SegmentND srcCopyE = new SegmentND(oriShift2);
                    Segment1D emptySegmentE = new Segment1D(..0, new Padding(0, 0));
                    SegmentND emptyNdE = new SegmentND(emptySegmentE, emptySegmentE, emptySegmentE, emptySegmentE);
                    SegmentND gateVecE = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(gateSliceE[1].Start..gateSliceE[1].End, new Padding(0, 0)));
                    SegmentND srcVecE = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(srcCopyE[1].Start..srcCopyE[1].End, new Padding(0, 0)));
                    int offsetS12 = 0;
                    int offsetS13 = 0;
                    int offsetD9 = glb.GlbMap[ItemName.Ofmap].GlbNByte / 4 * 3;
                    int offsetAct9 = 0;
                    List<int> src1Stride9 = new List<int>
                    {
                        gateVecE[1].Length, gateVecE[2].Length, gateVecE[3].Length
                    };
                    List<int> src2Stride9 = new List<int>
                    {
                        emptyNdE[1].Length, emptyNdE[2].Length, emptyNdE[3].Length
                    };
                    List<int> dstStride9 = new List<int>
                    {
                        srcVecE[1].Length, srcVecE[2].Length, srcVecE[3].Length
                    };
                    gnneActionUpdater.UpdateMfuAct1(gateVecE, emptyNdE, srcVecE, ACT1_SOURCE_TYPE.l2,
                        ACT1_SOURCE_TYPE.l2, DataTypes.Float16, DataTypes.Float16, DataTypes.Float16,
                        new DeQuantizeParam(0, 1f), new DeQuantizeParam(0, 1f), 0, 0, 0, is16Segments: true, 0,
                        ccrsToSet10, ccrsToClr6, offsetS12, offsetS13, offsetD9, offsetAct9, ItemName.Ofmap,
                        ItemName.LstmOfC, ItemName.LstmSegFittingParamGt, MFU_ACT1_FUNCTION.add, ItemName.Ofmap,
                        src1Stride9, src2Stride9, dstStride9);
                    int value = initHCcrValue[0] + 1;
                    if (timeStep == timeSteps[timeSteps.Count - 1])
                    {
                        value = (((object)_ofH == null) ? 1 : 2);
                    }

                    List<CcrClr> ofmapCcrsToClrE = new List<CcrClr>
                    {
                        new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 5)))
                    };
                    ofmapCcrsToClrE.Add(new CcrClr(_ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ofmap, 3))));
                    List<CcrSet> ccrsToSet11 = new List<CcrSet> { new CcrSet(initHCcrsToSet[0].Ccr, value) };
                    int offsetS14 = glb.GlbMap[ItemName.Ofmap].AllocatedBytes + glb.GlbMap[ItemName.Ofmap].GlbNByte / 4;
                    int offsetS15 = glb.GlbMap[ItemName.Ofmap].GlbNByte / 4 * 3;
                    int offsetD10 = 0;
                    int offsetAct10 = 0;
                    SegmentND mfuSegmentE = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(..1, new Padding(0, 0)), new Segment1D(..1, new Padding(0, 0)),
                        new Segment1D(oriShift2[1].Start..oriShift2[1].End, new Padding(0, 0)));
                    List<int> src1Stride10 = new List<int>
                    {
                        mfuSegmentE[1].Length, mfuSegmentE[2].Length, mfuSegmentE[3].Length
                    };
                    List<int> src2Stride10 = new List<int>
                    {
                        mfuSegmentE[1].Length, mfuSegmentE[2].Length, mfuSegmentE[3].Length
                    };
                    List<int> dstStride10 = new List<int>
                    {
                        mfuSegmentE[1].Length, mfuSegmentE[2].Length, mfuSegmentE[3].Length
                    };
                    gnneActionUpdater.UpdateMfuAct1(mfuSegmentE, mfuSegmentE, mfuSegmentE, ACT1_SOURCE_TYPE.l2,
                        ACT1_SOURCE_TYPE.l2, DataTypes.Float16, DataTypes.Float16, _outputType,
                        new DeQuantizeParam(0, 1f), new DeQuantizeParam(0, 1f), 0, 0, 0, is16Segments: false, 0,
                        ccrsToSet11, ofmapCcrsToClrE, offsetS14, offsetS15, offsetD10, offsetAct10, ItemName.Ofmap,
                        ItemName.Ofmap, ItemName.LstmBinQAct, MFU_ACT1_FUNCTION.mul, ItemName.LstmOfH, src1Stride10,
                        src2Stride10, dstStride10, isByChannel: false);
                    SegmentND ofmap = new SegmentND(new Segment1D(timeStep..(timeStep + 1), new Padding(0, 0)),
                        new Segment1D(l..(l + 1), new Padding(0, 0)), new Segment1D(m..(m + 1), new Padding(0, 0)),
                        new Segment1D(.._outputShape[3], new Padding(0, 0)));
                    List<CcrClr> ccrsToClr7 = new List<CcrClr> { new CcrClr(initHCcrsToSet[0].Ccr) };
                    gnneActionUpdater.UpdateStoreT(ofmap, _of, 0, ddrOf, 0, null, null, ccrsToClr7, ItemName.LstmOfH,
                        new List<int> { 1, 1, _outputShape[3] });
                    if (timeStep == timeSteps[timeSteps.Count - 1])
                    {
                        SegmentND ofmap2 = new SegmentND(new Segment1D(..1, new Padding(0, 0)),
                            new Segment1D(l..(l + 1), new Padding(0, 0)), new Segment1D(m..(m + 1), new Padding(0, 0)),
                            new Segment1D(.._outputShape[3], new Padding(0, 0)));
                        if ((object)_ofH != null)
                        {
                            List<CcrClr> ccrsToClr8 = new List<CcrClr> { new CcrClr(initHCcrsToSet[0].Ccr) };
                            gnneActionUpdater.UpdateStoreT(ofmap2, _ofH, 0, ddrOfH, 0, null, null, ccrsToClr8,
                                ItemName.LstmOfH, new List<int> { 1, 1, _outputShape[3] });
                        }

                        if ((object)_ofC != null)
                        {
                            List<CcrClr> ccrsToClr9 = new List<CcrClr> { new CcrClr(initCCcrsToSet[0].Ccr) };
                            gnneActionUpdater.UpdateStoreT(ofmap2, _ofC, 0, ddrOfC, 0, null, null, ccrsToClr9,
                                ItemName.LstmOfC, new List<int> { 1, 1, _outputShape[3] });
                        }
                    }

                    ifmapIndex = (ifmapIndex + 1) % nPingPongSplit;
                }
            }
        }

        TileUtilities.Assert(_ccrHandler.CcrSanityCheck(), "_ccrHandler.CcrSanityCheck()",
            "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\Transform\\Rules\\Tile\\TileLSTM.cs",
            665);
        return actions;
    }

    private void BuildL1Schedule(bool isInitH, GnneActionUpdater actionUpdater, TiledGlb glb, SegmentND ifmap,
        SegmentND weight, SegmentND psum, int iPp, int ofPp, int wPp, bool weightGroupOnly,
        WeightGroupHandler weightGroup, int[] ifLives, Nncase.TIR.Buffer ddrIf, Nncase.TIR.Buffer ddrW,
        bool isFirstL = false, int d = 0, bool isLastL = false, int l = 0, int b = 0)
    {
        int l1Pp = 0;
        bool restored = ReshapeIfmapWeightsOfmap(isInitH, ref ifmap, ref psum, ref weight);
        List<int> l1Params = L1Search(weight, psum, (!isInitH) ? _inputType : (isFirstL ? _initHType : _outputType));
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
        foreach (Segment1D ocSegment in ocSegments)
        {
            foreach (Segment1D batchSegment in psumBatchSegments)
            {
                foreach (Segment1D rowSegment in psumRowSegments)
                {
                    Padding p = new Padding(0, 0);
                    Segment1D currentRowSegment = rowSegment;
                    Segment1D inputRowSegment = TileUtilities.GetInputRowSegment(currentRowSegment.Start, currentRowSegment.Length,
                        ifmap[2].Length, weight[2].Length, 1, 1, in p);
                    foreach (Segment1D colSegment in psumColSegments)
                    {
                        Padding p2 = new Padding(0, 0);
                        Segment1D currentColSegment = colSegment;
                        Segment1D inputRowSegment2 = TileUtilities.GetInputRowSegment(currentColSegment.Start,
                            currentColSegment.Length, ifmap[3].Length, weight[3].Length, 1, 1, in p2);
                        new SegmentND(batchSegment, ocSegment, rowSegment, colSegment);
                        SegmentND convSegment = new SegmentND(batchSegment, ocSegment, currentRowSegment, currentColSegment);
                        bool isFirstLoop = true;
                        foreach (Segment1D icSegment in icSegments)
                        {
                            Segment1D weightIcSegment = icSegment;
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
                                    SegmentND segments = new SegmentND(ifmapBatchSegment, ifmapIcSegment, inputRowSegment,
                                        inputRowSegment2);
                                    new SegmentND(ocSegment, weightIcSegment, kernelRowSegment, kernelColSegment);
                                    bool hasIfmapLoad = false;
                                    SegmentND shiftedIfmap = new SegmentND(segments);
                                    if (shiftedIfmap[0].Length > 0 && shiftedIfmap[1].Length > 0 &&
                                        shiftedIfmap[2].Length > 0 && shiftedIfmap[3].Length > 0)
                                    {
                                        hasIfmapLoad = true;
                                        if (!weightGroupOnly)
                                        {
                                            int setIfmapCcr = 0;
                                            List<CcrSet> ifmapCcrsToSet = new List<CcrSet>();
                                            if ((ocSegment == ocSegments[ocSegments.Count - 1] &&
                                                 TileUtilities.IsLastSlice(shiftedIfmap, ifmap)) || (ocSegment == ocSegments[0] &&
                                                    TileUtilities.IsFirstSlice(shiftedIfmap, ifmap)))
                                            {
                                                setIfmapCcr = 1;
                                                if (ocSegment == ocSegments[ocSegments.Count - 1] &&
                                                    TileUtilities.IsLastSlice(shiftedIfmap, ifmap) && !isLastL &&
                                                    !isInitH)
                                                {
                                                    ifmapCcrsToSet.Add(new CcrSet(
                                                        _ccrHandler.GetCcrItem(
                                                            _ccrHandler.GetName(ItemName.IfmapFake, iPp)), 1));
                                                }
                                            }

                                            int ccrItem =
                                                _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.Ifmap, iPp));
                                            if (isInitH)
                                            {
                                                ccrItem = _ccrHandler.GetCcrItem(_ccrHandler.GetName(ItemName.LstmOfH));
                                            }

                                            List<CcrClr> ifmapCcrsToClr = new List<CcrClr>();
                                            if (setIfmapCcr > 0)
                                            {
                                                ifmapCcrsToClr.Add(new CcrClr(ccrItem));
                                            }

                                            int length = ifmap[1].Length;
                                            int length2 = ifmap[2].Length;
                                            int length3 = ifmap[3].Length;
                                            actionUpdater.UpdateG2LIf(shiftedIfmap, ifmap,
                                                (isInitH && isFirstL) ? _lInitH : _lIf, iPp, ifmapCcrsToSet, ifmapCcrsToClr, 0,
                                                (!isInitH) ? _inputType : (isFirstL ? _initHType : _outputType),
                                                isInitH ? ItemName.LstmOfH : ItemName.Ifmap, restored, length, length2,
                                                length3);
                                            if (ocSegment == ocSegments[ocSegments.Count - 1] &&
                                                TileUtilities.IsLastSlice(shiftedIfmap, ifmap) && !isLastL && !isInitH)
                                            {
                                                SegmentND ifmapLoadSegment = new SegmentND(
                                                    new Segment1D(..1, new Padding(0, 0)),
                                                    new Segment1D(l..(l + 1), new Padding(0, 0)),
                                                    new Segment1D(b..(b + 1), new Padding(0, 0)),
                                                    new Segment1D(.._inputShape[3], new Padding(0, 0)));
                                                List<CcrSet> ccrsToSet = new List<CcrSet>
                                                {
                                                    new CcrSet(
                                                        _ccrHandler.GetCcrItem(
                                                            _ccrHandler.GetName(ItemName.Ifmap, iPp)), ifLives[0])
                                                };
                                                List<int> stridesD = new List<int>
                                                {
                                                    ifmapLoadSegment[1].Length, ifmapLoadSegment[2].Length, ifmapLoadSegment[3].Length
                                                };
                                                List<CcrClr> ccrsToClr = new List<CcrClr>
                                                {
                                                    new CcrClr(_ccrHandler.GetCcrItem(
                                                        _ccrHandler.GetName(ItemName.IfmapFake, iPp)))
                                                };
                                                actionUpdater.UpdateLoadIf(ifmapLoadSegment, _lIf, iPp, ddrIf, 0, stridesD,
                                                    ItemName.Ifmap, ccrsToSet, ccrsToClr);
                                            }
                                        }
                                        else
                                        {
                                            ifLives[0]++;
                                        }
                                    }

                                    foreach (Segment1D kernelRowSlice in kernelRowSlices)
                                    {
                                        foreach (Segment1D kernelColSlice in kernelColSlices)
                                        {
                                            SegmentND weightSegment = new SegmentND(weightOcSegment, weightIcSegment, kernelRowSlice, kernelColSlice);
                                            SegmentND slice = new SegmentND(segments);
                                            int weightPasses = ((!isInitH)
                                                ? ((!(_wXcType == DataTypes.Int16)) ? 1 : 2)
                                                : ((!(_wRcType == DataTypes.Int16)) ? 1 : 2));
                                            int inputPasses = ((!isInitH)
                                                ? ((!(_inputType == DataTypes.Int16)) ? 1 : 2)
                                                : ((!isFirstL)
                                                    ? ((!(_outputType == DataTypes.Int16)) ? 1 : 2)
                                                    : ((!(_initHType == DataTypes.Int16)) ? 1 : 2)));
                                            for (int i = 0; i < weightPasses; i++)
                                            {
                                                for (int inputPass = 0; inputPass < inputPasses; inputPass++)
                                                {
                                                    if (!weightGroupOnly)
                                                    {
                                                        List<CcrSet> weightCcrsToSet = new List<CcrSet>();
                                                        List<CcrClr> weightCcrsToClr = new List<CcrClr>();
                                                        if (b == 0 && isFirstL && inputPass == 0)
                                                        {
                                                            if (!isInitH)
                                                            {
                                                                weightCcrsToSet.Add(new CcrSet(
                                                                    _ccrHandler.GetCcrItem(
                                                                        _ccrHandler.GetName(ItemName.LstmWXc)), 1));
                                                                weightCcrsToClr.Add(new CcrClr(
                                                                    _ccrHandler.GetCcrItem(
                                                                        _ccrHandler.GetName(ItemName.LstmWXc))));
                                                                actionUpdater.UpdateLoadW(weightSegment, _lWXc,
                                                                    weightGroup, 0, ddrW, weightCcrsToSet, null,
                                                                    glb.GlbMap[ItemName.LstmWXc].AllocatedBytes * d,
                                                                    h2c: false,
                                                                    weightGroup.CurrentOffset() * d *
                                                                    TileUtilities.GetBytesPerElement(_wXcType),
                                                                    ItemName.LstmWXc, i);
                                                            }
                                                            else
                                                            {
                                                                weightCcrsToSet.Add(new CcrSet(
                                                                    _ccrHandler.GetCcrItem(
                                                                        _ccrHandler.GetName(ItemName.LstmWRc)), 1));
                                                                weightCcrsToClr.Add(new CcrClr(
                                                                    _ccrHandler.GetCcrItem(
                                                                        _ccrHandler.GetName(ItemName.LstmWRc))));
                                                                actionUpdater.UpdateLoadW(weightSegment, _lWRc,
                                                                    weightGroup, 0, ddrW, weightCcrsToSet, null,
                                                                    glb.GlbMap[ItemName.LstmWRc].AllocatedBytes * d,
                                                                    h2c: false,
                                                                    weightGroup.CurrentOffset() * d *
                                                                    TileUtilities.GetBytesPerElement(_wRcType),
                                                                    ItemName.LstmWRc, i);
                                                            }
                                                        }

                                                        int offsetWS = (isInitH
                                                            ? (glb.GlbMap[ItemName.LstmWRc].AllocatedBytes * d)
                                                            : (glb.GlbMap[ItemName.LstmWXc].AllocatedBytes * d));
                                                        actionUpdater.UpdateG2RW(weightSegment, weightGroup,
                                                            weight[0].Length, isInitH ? _lWRc : _lWXc, wPp, i, null,
                                                            weightCcrsToClr, offsetWS, _outputShape[3] * 4 * d,
                                                            isInitH ? ItemName.LstmWRc : ItemName.LstmWXc,
                                                            isInitH ? ItemName.LstmWRcQarg : ItemName.LstmWXcQarg);
                                                    }
                                                    else
                                                    {
                                                        weightGroup.UpdateWeightGroup(weightSegment);
                                                    }

                                                    if (!weightGroupOnly)
                                                    {
                                                        int bias = ((TensorConst)_oldLstm[GNNELSTM.IfDeqBias]).Value
                                                            .ToScalar<int>();
                                                        if (isInitH)
                                                        {
                                                            bias = (isFirstL
                                                                ? ((TensorConst)_oldLstm[GNNELSTM.HDeqBias0]).Value
                                                                .ToScalar<int>()
                                                                : ((TensorConst)_oldLstm[GNNELSTM.HDeqBias1]).Value
                                                                .ToScalar<int>());
                                                        }

                                                        actionUpdater.UpdateL2RIf(slice, shiftedIfmap, 1, 1,
                                                            weight[1].Length, (isInitH && isFirstL) ? _lInitH : _lIf, 0,
                                                            inputPass, bias);
                                                    }

                                                    bool releaseIf = false;
                                                    int isLastPass;
                                                    if (kernelColSlice == kernelColSlices[
                                                            kernelColSlices.Count - 1])
                                                    {
                                                        if (kernelRowSlice == kernelRowSlices[
                                                                kernelRowSlices.Count - 1] && i == weightPasses - 1)
                                                        {
                                                            isLastPass = ((inputPass == inputPasses - 1) ? 1 : 0);
                                                            goto SetReleaseIf;
                                                        }
                                                    }

                                                    isLastPass = 0;
                                                    goto SetReleaseIf;
                                                    SetReleaseIf:
                                                    if (((uint)isLastPass & (hasIfmapLoad ? 1u : 0u)) != 0)
                                                    {
                                                        releaseIf = true;
                                                        hasIfmapLoad = false;
                                                    }

                                                    bool loopStart = false;
                                                    if (isFirstLoop && i == 0 && inputPass == 0)
                                                    {
                                                        loopStart = true;
                                                        isFirstLoop = false;
                                                    }

                                                    bool isLastKernelSlice = weightIcSegment.End == weight[1].Length &&
                                                                 kernelColSlice.End == weight[3].End &&
                                                                 kernelRowSlice.End == weight[2].End && i == weightPasses - 1 &&
                                                                 inputPass == inputPasses - 1;
                                                    DataType @float = DataTypes.Float16;
                                                    ACT0_OUTPUT_DEST act0OutputDest = ACT0_OUTPUT_DEST.dm;
                                                    if (!weightGroupOnly)
                                                    {
                                                        int shift = ((TensorConst)_oldLstm[GNNELSTM.XcShiftBits]).Value
                                                            .ToScalar<int>();
                                                        if (isInitH)
                                                        {
                                                            shift = ((!isFirstL)
                                                                ? ((TensorConst)_oldLstm[GNNELSTM.RcShiftBits1]).Value
                                                                .ToScalar<int>()
                                                                : ((TensorConst)_oldLstm[GNNELSTM.RcShiftBits0]).Value
                                                                .ToScalar<int>());
                                                        }

                                                        int setOfmapCcr = 0;
                                                        if (isLastKernelSlice && ocSegment.End == weight[0].End &&
                                                            act0OutputDest == ACT0_OUTPUT_DEST.dm)
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

                                                        List<CcrClr> ccrsToClr2 = new List<CcrClr>();
                                                        int actOffset = 0;
                                                        if (!isInitH)
                                                        {
                                                            actOffset = glb.GlbMap[ItemName.LstmActXc].GlbNByte / 2 * d;
                                                        }

                                                        if (isInitH)
                                                        {
                                                            actOffset = glb.GlbMap[ItemName.LstmActRc].GlbNByte / 2 * d;
                                                            if (!isFirstL)
                                                            {
                                                                actOffset += glb.GlbMap[ItemName.LstmActRc].AllocatedBytes;
                                                            }
                                                        }

                                                        actionUpdater.UpdateR2LPsum(shift, convSegment, convSegment, psum,
                                                            0, l1Pp, act0OutputDest, releaseIf, Math.Max(i, inputPass),
                                                            TcuComputeMode.NormalConv2d, loopStart, isLastKernelSlice, 1, 1,
                                                            weight[0].Length, _inputType, isInitH ? _wRcType : _wXcType,
                                                            @float,
                                                            isInitH
                                                                ? _lActRc0.CheckedDataType
                                                                : _lActXc.CheckedDataType, ofmapCcrsToSet, ccrsToClr2, actOffset,
                                                            glb.GlbMap[ItemName.Ofmap].AllocatedBytes *
                                                            (isInitH ? 1 : 0), null,
                                                            isInitH ? ItemName.LstmActRc : ItemName.LstmActXc);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        ifLives[0] = Math.Min(ifLives[0], 2);
    }

    private List<List<Segment1D>> GetL1McSeg(SegmentND ifmap, SegmentND psum, int mInloop, int cInloop)
    {
        List<Segment1D> ocSegments =
            TileUtilities.GetSegmentStartEndLength(psum[1].Start, mInloop, psum[1].End);
        List<Segment1D> icSegments =
            TileUtilities.GetSegmentStartEndLength(ifmap[1].Start, cInloop, ifmap[1].End);
        return new List<List<Segment1D>> { ocSegments, icSegments };
    }

    private List<int> L1Search(SegmentND weight, SegmentND psum, DataType type)
    {
        int puHeight = Math.Min(GNNEEnv.PuHeight, weight[1].Length);
        int puWidth = Math.Min(GNNEEnv.PuWidth, weight[0].Length);
        int length = weight[2].Length;
        int length2 = weight[3].Length;
        int ofmapRows = 1;
        int ofmapCols = 1;
        int length3 = weight[2].Length;
        int length4 = weight[3].Length;
        int val = 1;
        int val2 = 1;
        int psumPingPangSplit = 1;
        int bytesPerElement = TileUtilities.GetBytesPerElement(type);
        if (!HandleL1Allocate(length, length2, Math.Max(ofmapRows, val), Math.Max(ofmapCols, val2), psumPingPangSplit,
                bytesPerElement))
        {
            throw new NotSupportedException("exceeds L1 size");
        }

        return new List<int>
        {
            puHeight,
            puWidth,
            ofmapRows,
            ofmapCols,
            length3,
            length4
        };
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

    private bool ReshapeIfmapWeightsOfmap(bool isInitH, ref SegmentND ifmap, ref SegmentND ofmap, ref SegmentND weights)
    {
        if (ifmap[2].Length == 1 && ifmap[3].Length == 1 && weights[2].Length == 1 && weights[3].Length == 1 &&
            (weights[1].Length % 24 == 0 || weights[1].Length % 20 == 0 || weights[1].Length % 16 == 0))
        {
            int dim1Size = ((weights[1].Length % 24 == 0) ? 24 : ((weights[1].Length % 20 == 0) ? 20 : 16));
            int groupCount = weights[1].Length / dim1Size;
            uint dim3Size = 4u;
            if (groupCount <= dim3Size)
            {
                dim3Size = (uint)groupCount;
            }
            else
            {
                while (dim3Size != 0 && groupCount % dim3Size != 0L)
                {
                    dim3Size--;
                }
            }

            long remainingGroups = groupCount / dim3Size;
            uint dim2Size = 31u;
            if (remainingGroups <= dim2Size)
            {
                dim2Size = (uint)remainingGroups;
            }
            else
            {
                while (dim2Size != 0 && remainingGroups % dim2Size != 0L)
                {
                    dim2Size--;
                }
            }

            dim1Size = (int)((int)remainingGroups / dim2Size * dim1Size);
            ifmap = new SegmentND(ifmap[0], new Segment1D(..dim1Size, new Padding(0, 0)),
                new Segment1D(..(int)dim2Size, new Padding(0, 0)), new Segment1D(..(int)dim3Size, new Padding(0, 0)));
            weights = new SegmentND(weights[0], new Segment1D(..dim1Size, new Padding(0, 0)),
                new Segment1D(..(int)dim2Size, new Padding(0, 0)), new Segment1D(..(int)dim3Size, new Padding(0, 0)));
            ofmap = new SegmentND(ofmap[0], ofmap[1], ofmap[2], ofmap[3]);
            if (!isInitH)
            {
                _wXcShapeReshape = new GNNEShape(weights[0].Length, dim1Size, (int)dim2Size, (int)dim3Size);
            }
            else
            {
                _wRcShapeReshape = new GNNEShape(weights[0].Length, dim1Size, (int)dim2Size, (int)dim3Size);
            }

            return true;
        }

        if (!isInitH)
        {
            _wXcShapeReshape =
                new GNNEShape(weights[0].Length, weights[1].Length, weights[2].Length, weights[3].Length);
        }
        else
        {
            _wRcShapeReshape =
                new GNNEShape(weights[0].Length, weights[1].Length, weights[2].Length, weights[3].Length);
        }

        return false;
    }

    private GNNEShape ComputeRS(GNNEShape inShape, GNNEShape wShape)
    {
        if (inShape[2] == 1 && inShape[3] == 1 && wShape[2] == 1 && wShape[3] == 1 &&
            (wShape[1] % 24 == 0 || wShape[1] % 20 == 0 || wShape[1] % 16 == 0))
        {
            int dim1Size = ((wShape[1] % 24 == 0) ? 24 : ((wShape[1] % 20 == 0) ? 20 : 16));
            int groupCount = wShape[1] / dim1Size;
            int dim3Size = 4;
            if (groupCount <= dim3Size)
            {
                dim3Size = groupCount;
            }
            else
            {
                while (dim3Size > 0 && groupCount % dim3Size != 0)
                {
                    dim3Size--;
                }
            }

            int remainingGroups = groupCount / dim3Size;
            int dim2Size = 31;
            if (remainingGroups <= dim2Size)
            {
                dim2Size = remainingGroups;
            }
            else
            {
                while (dim2Size > 0 && remainingGroups % dim2Size != 0)
                {
                    dim2Size--;
                }
            }

            dim1Size = remainingGroups / dim2Size * dim1Size;
            return new GNNEShape(wShape[0], dim1Size, dim2Size, dim3Size);
        }

        return wShape;
    }

    private AllocateResult HandleAllocate(int nPingPongSplit, bool isFinal = false)
    {
        List<BoxOnGlb> boxes = new List<BoxOnGlb>();
        Dictionary<ItemName, TensorOnGlb> glbMap = new Dictionary<ItemName, TensorOnGlb>();
        GNNEShape inShape = new GNNEShape(1, _inputShape[3], 1, 1);
        GNNEShape wShape = new GNNEShape(_wXcShape[2], _wXcShape[3], 1, 1);
        GNNEShape wShape2 = new GNNEShape(_wRcShape[2], _wRcShape[3], 1, 1);
        wShape = ComputeRS(inShape, wShape);
        wShape2 = ComputeRS(inShape, wShape2);
        int[] ifmapDims = new int[4] { 1, 0, 1, 1 };
        ifmapDims[1] = _inputShape[3];
        TensorOnGlb ifmapTensor = new TensorOnGlb(ifmapDims, _inputType, 0);
        int value = ifmapTensor.GlbNByte * nPingPongSplit;
        value = TileUtilities.GetAlignedNum(value, GNNEEnv.IfmapBankWidth * GNNEEnv.GlbBankWidth);
        TensorOnGlb wXcTensor =
            new TensorOnGlb(new int[4] { wShape[2], wShape[3], wShape[1], wShape[0] }, _wXcType, 0);
        int wXcBytes = SpaceSearcher.GetWeightSize(wShape[2], wShape[3], wShape[1], wShape[0],
            TileUtilities.GetBytesPerElement(_wXcType)) * _wXcShape[1];
        wXcTensor.AllocatedBytes = wXcBytes / _wXcShape[1];
        wXcBytes = TileUtilities.GetAlignedNum(wXcBytes, GNNEEnv.WBankWidth * GNNEEnv.GlbBankWidth);
        TensorOnGlb actXcTensor = new TensorOnGlb(new int[4] { 1, 1, _actXcShape[2], _actXcShape[3] }, _actXcType, 0);
        int actXcBytes = actXcTensor.GlbNByte * _actXcShape[1];
        actXcBytes = TileUtilities.GetAlignedNum(actXcBytes, GNNEEnv.ActBankWidth * GNNEEnv.GlbBankWidth);
        TensorOnGlb wRcTensor =
            new TensorOnGlb(new int[4] { wShape2[2], wShape2[3], wShape2[1], wShape2[0] }, _wRcType, 0);
        int wRcBytes = SpaceSearcher.GetWeightSize(wShape2[2], wShape2[3], wShape2[1], wShape2[0],
            TileUtilities.GetBytesPerElement(_wRcType)) * _wRcShape[1];
        wRcTensor.AllocatedBytes = wRcBytes / _wRcShape[1];
        wRcBytes = TileUtilities.GetAlignedNum(wRcBytes, GNNEEnv.WBankWidth * GNNEEnv.GlbBankWidth);
        TensorOnGlb actRcTensor = new TensorOnGlb(new int[4] { 1, 1, _actRcShape[2], _actRcShape[3] }, _actRcType, 0);
        int actRcBytes = actRcTensor.GlbNByte * _actRcShape[1];
        actRcBytes = (actRcTensor.AllocatedBytes =
            TileUtilities.GetAlignedNum(actRcBytes, GNNEEnv.ActBankWidth * GNNEEnv.GlbBankWidth));
        actRcBytes *= 2;
        int[] initHDims = new int[4] { 1, 0, 1, 1 };
        initHDims[1] = _initHShape[3];
        TensorOnGlb initHTensor = new TensorOnGlb(initHDims, _outputType, 0);
        int glbNByte = initHTensor.GlbNByte;
        glbNByte = TileUtilities.GetAlignedNum(glbNByte, GNNEEnv.OfmapBankWidth * GNNEEnv.GlbBankWidth);
        int[] initCDims = new int[4] { 1, 0, 1, 1 };
        initCDims[1] = _initCShape[3];
        TensorOnGlb initCTensor = new TensorOnGlb(initCDims, _outputCType, 0);
        int glbNByte2 = initCTensor.GlbNByte;
        glbNByte2 = TileUtilities.GetAlignedNum(glbNByte2, GNNEEnv.OfmapBankWidth * GNNEEnv.GlbBankWidth);
        int[] gateDims = new int[4] { 1, 0, 1, 1 };
        gateDims[1] = _outputShape[3] * 4;
        TensorOnGlb gateTensor = new TensorOnGlb(gateDims, DataTypes.Float16, 0);
        int glbNByte3 = gateTensor.GlbNByte;
        glbNByte3 = (gateTensor.AllocatedBytes =
            TileUtilities.GetAlignedNum(glbNByte3, GNNEEnv.OfmapBankWidth * GNNEEnv.GlbBankWidth));
        glbNByte3 *= 2;
        TensorOnGlb segFittingFtTensor = new TensorOnGlb(new int[4] { 1, 1, 1, 49 }, DataTypes.Float16, 0);
        int glbNByte4 = segFittingFtTensor.GlbNByte;
        glbNByte4 = TileUtilities.GetAlignedNum(glbNByte4, 8 * GNNEEnv.GlbBankWidth);
        TensorOnGlb segFittingGtTensor = new TensorOnGlb(new int[4] { 1, 1, 1, 49 }, DataTypes.Float16, 0);
        int[] binActDims = new int[4] { 1, 1, 0, 7 };
        binActDims[2] = _outputShape[3];
        TensorOnGlb binActTensor = new TensorOnGlb(binActDims, DataTypes.BFloat16, 0);
        int glbNByte5 = binActTensor.GlbNByte;
        glbNByte5 = TileUtilities.GetAlignedNum(glbNByte5, GNNEEnv.GlbWidth * GNNEEnv.GlbBankWidth);
        int[] binQActDims = new int[4] { 1, 1, 0, 7 };
        binQActDims[2] = _outputShape[3];
        TensorOnGlb binQActTensor = new TensorOnGlb(binQActDims, DataTypes.Float16, 0);
        int glbNByte6 = binQActTensor.GlbNByte;
        glbNByte6 = TileUtilities.GetAlignedNum(glbNByte6, GNNEEnv.GlbWidth * GNNEEnv.GlbBankWidth);
        int basementSize = SpaceSearcher.GetBasementSize();
        basementSize = TileUtilities.GetAlignedNum(basementSize, GNNEEnv.GlbWidth * GNNEEnv.GlbBankWidth);
        boxes.Add(new BoxOnGlb(new int[2] { GNNEEnv.GlbWidth, basementSize / GNNEEnv.GlbWidth / GNNEEnv.GlbBankWidth },
            ItemName.Basement));
        boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.IfmapBankWidth, value / GNNEEnv.IfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.Ifmap));
        boxes.Add(new BoxOnGlb(new int[2] { GNNEEnv.WBankWidth, wXcBytes / GNNEEnv.WBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.LstmWXc));
        boxes.Add(new BoxOnGlb(new int[2] { GNNEEnv.ActBankWidth, actXcBytes / GNNEEnv.ActBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.LstmActXc));
        boxes.Add(new BoxOnGlb(new int[2] { GNNEEnv.WBankWidth, wRcBytes / GNNEEnv.WBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.LstmWRc));
        boxes.Add(new BoxOnGlb(new int[2] { GNNEEnv.ActBankWidth, actRcBytes / GNNEEnv.ActBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.LstmActRc));
        boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.OfmapBankWidth, glbNByte / GNNEEnv.OfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.LstmOfH));
        boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.OfmapBankWidth, glbNByte2 / GNNEEnv.OfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.LstmOfC));
        boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.OfmapBankWidth, glbNByte3 / GNNEEnv.OfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.Ofmap));
        boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.OfmapBankWidth, glbNByte4 / GNNEEnv.OfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.LstmSegFittingParamFt));
        boxes.Add(new BoxOnGlb(
            new int[2] { GNNEEnv.OfmapBankWidth, glbNByte4 / GNNEEnv.OfmapBankWidth / GNNEEnv.GlbBankWidth },
            ItemName.LstmSegFittingParamGt));
        boxes.Add(new BoxOnGlb(new int[2] { GNNEEnv.GlbWidth, glbNByte5 / GNNEEnv.GlbWidth / GNNEEnv.GlbBankWidth },
            ItemName.LstmBinAct));
        boxes.Add(new BoxOnGlb(new int[2] { GNNEEnv.GlbWidth, glbNByte6 / GNNEEnv.GlbWidth / GNNEEnv.GlbBankWidth },
            ItemName.LstmBinQAct));
        glbMap.Add(ItemName.Ifmap, ifmapTensor);
        glbMap.Add(ItemName.LstmWXc, wXcTensor);
        glbMap.Add(ItemName.LstmActXc, actXcTensor);
        glbMap.Add(ItemName.LstmWRc, wRcTensor);
        glbMap.Add(ItemName.LstmActRc, actRcTensor);
        glbMap.Add(ItemName.LstmOfH, initHTensor);
        glbMap.Add(ItemName.LstmOfC, initCTensor);
        glbMap.Add(ItemName.Ofmap, gateTensor);
        glbMap.Add(ItemName.LstmSegFittingParamFt, segFittingFtTensor);
        glbMap.Add(ItemName.LstmSegFittingParamGt, segFittingGtTensor);
        glbMap.Add(ItemName.LstmBinAct, binActTensor);
        glbMap.Add(ItemName.LstmBinQAct, binQActTensor);
        if (_wXcType == DataTypes.UInt8 || _wXcType == DataTypes.Int16)
        {
            TensorOnGlb wXcQargTensor = new TensorOnGlb(new int[4] { 0, 0, 0, 0 }, _wXcType, 0);
            int wQargSize = SpaceSearcher.GetWQargSize(_wXcShape[1] * _wXcShape[2],
                TileUtilities.GetBytesPerElement(_lwXcQarg[GNNELoadW.Input].CheckedDataType));
            wQargSize = TileUtilities.GetAlignedNum(wQargSize, GNNEEnv.WQargBankWidth * GNNEEnv.GlbBankWidth);
            boxes.Add(new BoxOnGlb(
                new int[2] { GNNEEnv.WQargBankWidth, wQargSize / GNNEEnv.WQargBankWidth / GNNEEnv.GlbBankWidth },
                ItemName.LstmWXcQarg));
            glbMap.Add(ItemName.LstmWXcQarg, wXcQargTensor);
        }

        if (_wRcType == DataTypes.UInt8 || _wRcType == DataTypes.Int16)
        {
            TensorOnGlb wRcQargTensor = new TensorOnGlb(new int[4] { 0, 0, 0, 0 }, _wRcType, 0);
            int wQargSize2 = SpaceSearcher.GetWQargSize(_wRcShape[1] * _wRcShape[2],
                TileUtilities.GetBytesPerElement(_lwRcQarg[GNNELoadW.Input].CheckedDataType));
            wQargSize2 = TileUtilities.GetAlignedNum(wQargSize2, GNNEEnv.WQargBankWidth * GNNEEnv.GlbBankWidth);
            boxes.Add(new BoxOnGlb(
                new int[2] { GNNEEnv.WQargBankWidth, wQargSize2 / GNNEEnv.WQargBankWidth / GNNEEnv.GlbBankWidth },
                ItemName.LstmWRcQarg));
            glbMap.Add(ItemName.LstmWRcQarg, wRcQargTensor);
        }

        AllocateResult allocateResult = SpaceSearcher.TryAllocate(new BoxPacker(16) { Boxes = boxes });
        if (allocateResult.IsOk)
        {
            foreach (KeyValuePair<ItemName, TensorOnGlb> entry in glbMap)
            {
                if (allocateResult.Items.ContainsKey(entry.Key))
                {
                    entry.Value.Mmu = allocateResult.Items[entry.Key];
                }
            }
        }

        return new AllocateResult { IsOk = allocateResult.IsOk, Items = allocateResult.Items, GlbMap = glbMap };
    }

    private void InitParameters(Call lstmNode, Call ld, Call st, Call stH, Call stC)
    {
        _oldLstm = lstmNode;
        _of = st;
        _ofH = stH;
        _ofC = stC;
        _inputShape = new GNNEShape(_oldLstm[GNNELSTM.Input].CheckedShape[0].FixedValue,
            _oldLstm[GNNELSTM.Input].CheckedShape[1].FixedValue, _oldLstm[GNNELSTM.Input].CheckedShape[2].FixedValue,
            _oldLstm[GNNELSTM.Input].CheckedShape[3].FixedValue);
        _wXcShape = new GNNEShape(_oldLstm[GNNELSTM.WXc].CheckedShape[0].FixedValue,
            _oldLstm[GNNELSTM.WXc].CheckedShape[1].FixedValue, _oldLstm[GNNELSTM.WXc].CheckedShape[2].FixedValue,
            _oldLstm[GNNELSTM.WXc].CheckedShape[3].FixedValue);
        _wXcShapeReshape = new GNNEShape(_oldLstm[GNNELSTM.WXc].CheckedShape[0].FixedValue,
            _oldLstm[GNNELSTM.WXc].CheckedShape[1].FixedValue, _oldLstm[GNNELSTM.WXc].CheckedShape[2].FixedValue,
            _oldLstm[GNNELSTM.WXc].CheckedShape[3].FixedValue);
        _actXcShape = new GNNEShape(_oldLstm[GNNELSTM.ActXc].CheckedShape[0].FixedValue,
            _oldLstm[GNNELSTM.ActXc].CheckedShape[1].FixedValue, _oldLstm[GNNELSTM.ActXc].CheckedShape[2].FixedValue,
            _oldLstm[GNNELSTM.ActXc].CheckedShape[3].FixedValue);
        _wRcShape = new GNNEShape(_oldLstm[GNNELSTM.WRc].CheckedShape[0].FixedValue,
            _oldLstm[GNNELSTM.WRc].CheckedShape[1].FixedValue, _oldLstm[GNNELSTM.WRc].CheckedShape[2].FixedValue,
            _oldLstm[GNNELSTM.WRc].CheckedShape[3].FixedValue);
        _wRcShapeReshape = new GNNEShape(_oldLstm[GNNELSTM.WRc].CheckedShape[0].FixedValue,
            _oldLstm[GNNELSTM.WRc].CheckedShape[1].FixedValue, _oldLstm[GNNELSTM.WRc].CheckedShape[2].FixedValue,
            _oldLstm[GNNELSTM.WRc].CheckedShape[3].FixedValue);
        _actRcShape = new GNNEShape(_oldLstm[GNNELSTM.ActRc0].CheckedShape[0].FixedValue,
            _oldLstm[GNNELSTM.ActRc0].CheckedShape[1].FixedValue, _oldLstm[GNNELSTM.ActRc0].CheckedShape[2].FixedValue,
            _oldLstm[GNNELSTM.ActRc0].CheckedShape[3].FixedValue);
        _initHShape = new GNNEShape(_oldLstm[GNNELSTM.InitialH].CheckedShape[0].FixedValue,
            _oldLstm[GNNELSTM.InitialH].CheckedShape[1].FixedValue,
            _oldLstm[GNNELSTM.InitialH].CheckedShape[2].FixedValue,
            _oldLstm[GNNELSTM.InitialH].CheckedShape[3].FixedValue);
        _initCShape = new GNNEShape(_oldLstm[GNNELSTM.InitialC].CheckedShape[0].FixedValue,
            _oldLstm[GNNELSTM.InitialC].CheckedShape[1].FixedValue,
            _oldLstm[GNNELSTM.InitialC].CheckedShape[2].FixedValue,
            _oldLstm[GNNELSTM.InitialC].CheckedShape[3].FixedValue);
        _outputShape = new GNNEShape(st.CheckedShape[0].FixedValue, st.CheckedShape[1].FixedValue,
            st.CheckedShape[2].FixedValue, st.CheckedShape[3].FixedValue);
        _outputHShape = (((object)_ofH != null)
            ? new GNNEShape(stH.CheckedShape[0].FixedValue, stH.CheckedShape[1].FixedValue,
                stH.CheckedShape[2].FixedValue, stH.CheckedShape[3].FixedValue)
            : new GNNEShape(_initHShape.N, _initHShape.C, _initHShape.H, _initHShape.W));
        _outputCShape = (((object)_ofC != null)
            ? new GNNEShape(stC.CheckedShape[0].FixedValue, stC.CheckedShape[1].FixedValue,
                stC.CheckedShape[2].FixedValue, stC.CheckedShape[3].FixedValue)
            : new GNNEShape(_initCShape.N, _initCShape.C, _initCShape.H, _initCShape.W));
        _lIf = ld;
        _lWXc = (Call)_oldLstm[GNNELSTM.WXc];
        _lActXc = (Call)_oldLstm[GNNELSTM.ActXc];
        _lWRc = (Call)_oldLstm[GNNELSTM.WRc];
        _lActRc0 = (Call)_oldLstm[GNNELSTM.ActRc0];
        _lActRc1 = (Call)_oldLstm[GNNELSTM.ActRc1];
        _lInitH = (Call)_oldLstm[GNNELSTM.InitialH];
        _lInitC = (Call)_oldLstm[GNNELSTM.InitialC];
        _lSegFt = (Call)_oldLstm[GNNELSTM.SegFittingParamFt];
        _lSegGt = (Call)_oldLstm[GNNELSTM.SegFittingParamGt];
        _lwXcQarg = (Call)_oldLstm[GNNELSTM.WXcQarg];
        _lwRcQarg = (Call)_oldLstm[GNNELSTM.WRcQarg];
        _lwBinAct = (Call)_oldLstm[GNNELSTM.ActBin];
        _lwBinQAct = (Call)_oldLstm[GNNELSTM.ActBinQ];
        _of = st;
        _ofH = stH;
        _ofC = stC;
        _inputType = _oldLstm[GNNELSTM.Input].CheckedDataType;
        _wXcType = _oldLstm[GNNELSTM.WXc].CheckedDataType;
        _actXcType = _oldLstm[GNNELSTM.ActXc].CheckedDataType;
        _wRcType = _oldLstm[GNNELSTM.WRc].CheckedDataType;
        _actRcType = _oldLstm[GNNELSTM.ActRc0].CheckedDataType;
        _initHType = _oldLstm[GNNELSTM.InitialH].CheckedDataType;
        _initCType = _oldLstm[GNNELSTM.InitialC].CheckedDataType;
        _outputType = ((TensorType)((TupleType)_oldLstm.CheckedType)[0]).DType;
        _outputHType = (((object)_ofH != null) ? _ofH[GNNEStore.Input].CheckedDataType : _outputType);
        _outputCType = (((object)_ofC != null) ? _ofC[GNNEStore.Input].CheckedDataType : DataTypes.Float16);
        _weightGroups.Add(new WeightGroupHandler(_wXcType, _wXcType));
        _weightGroups.Add(new WeightGroupHandler(_wRcType, _wRcType));
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Expr output = (Expr)__result["output"];
        Call midCall = (Call)__result["midCall"];
        IReadOnlyList<Expr> midCallParams = (IReadOnlyList<Expr>)__result["midCallParams"];
        return GetReplace(output, midCall, midCallParams);
    }
}
