// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.K230;
using Nncase.TIR;

namespace Nncase.Passes.Rules.Tile;

internal sealed class DDrMacCalcVisitor : ExprVisitor<Unit, Unit>
{
    public Fusion EntryFusion => (Fusion)VisitRoot;

    public string FusionName { get; }

    public IDictionary<string, RoofLineInfo> FusionMacsMap { get; }

    public DDrMacCalcVisitor(Dictionary<string, RoofLineInfo> fusionMacsMap, string name)
        : base(false)
    {
        FusionMacsMap = fusionMacsMap;
        FusionName = name;
    }

    protected override Unit DefaultVisitLeaf(Expr expr)
    {
        return default(Unit);
    }

    protected override Unit VisitLeafFusion(Fusion fusion)
    {
        return default(Unit);
    }

    protected override Unit VisitLeafCall(Call call)
    {
        if (!FusionMacsMap.TryGetValue(FusionName, out RoofLineInfo value))
        {
            value = new RoofLineInfo();
            FusionMacsMap.Add(FusionName, value);
        }

        ulong macs = 0uL;
        ulong flops = 0uL;
        long onChipTraffic = 0L;
        long offChipTraffic = 0L;
        ulong loadCount = 0uL;
        ulong storeCount = 0uL;
        long offChipWeightTraffic = 0L;
        long perTileTraffic = 3840L;
        long perKernelTapTraffic = 538L;
        long inputLoadTraffic = 1638400L;
        Expr target = call.Target;
        if (!(target is GNNELoad))
        {
            if (!(target is GNNELoadW))
            {
                if (!(target is GNNEStore))
                {
                    if (!(target is GNNEConv2D))
                    {
                        if (!(target is GNNEConv2DTranspose))
                        {
                            if (!(target is GNNEPdp0DW))
                            {
                                if (!(target is GNNEMatMul))
                                {
                                    if (!(target is GNNEPdp1))
                                    {
                                        if (!(target is GNNEActivation))
                                        {
                                            if (!(target is GNNETranspose))
                                            {
                                                if (!(target is Reshape))
                                                {
                                                    if (!(target is Ai2dResize))
                                                    {
                                                        if (!(target is GNNELSTM))
                                                        {
                                                            if (!(target is GNNEPad))
                                                            {
                                                                if (!(target is GNNEPdp0Reduce))
                                                                {
                                                                    if (!(target is GetItem) && !(target is Concat) &&
                                                                        !(target is Split))
                                                                    {
                                                                        throw new ArgumentException(
                                                                            call.Target.GetType().Name +
                                                                            " is not supported.");
                                                                    }
                                                                }
                                                                else
                                                                {
                                                                    onChipTraffic = (long)CostUtility.GetMemoryAccess(
                                                                        call[GNNEPdp0Reduce.Input].CheckedType,
                                                                        call.CheckedType);
                                                                    value.AddOperator(
                                                                        "GNNEPdp0Reduce",
                                                                        call[GNNEPdp0Reduce.Input].CheckedShape
                                                                            .Select((Dimension x) =>
                                                                                (" ", (ulong)x.FixedValue)).ToArray());
                                                                }
                                                            }
                                                            else
                                                            {
                                                                onChipTraffic = (long)CostUtility.GetMemoryAccess(
                                                                    call[GNNEPad.Input].CheckedType, call.CheckedType);
                                                                value.AddOperator(
                                                                    "GNNEPad",
                                                                    call[GNNEPad.Input].CheckedShape
                                                                        .Select((Dimension x) =>
                                                                            (" ", (ulong)x.FixedValue)).ToArray());
                                                            }
                                                        }
                                                        else
                                                        {
                                                            int[] inputShape = call[GNNELSTM.Input].CheckedShape
                                                                .ToValueArray();
                                                            int[] wXcShape = call[GNNELSTM.WXc].CheckedShape
                                                                .ToValueArray();
                                                            call[GNNELSTM.InitialH].CheckedShape.ToValueArray();
                                                            int[] wRcShape = call[GNNELSTM.WRc].CheckedShape
                                                                .ToValueArray();
                                                            long inputTraffic =
                                                                (long)CostUtility.GetMemoryAccess(call[GNNELSTM.Input]
                                                                    .CheckedType) *
                                                                (long)Math.Ceiling((double)wXcShape[2] /
                                                                    (double)GNNEEnv.PuWidth) * wXcShape[1];
                                                            long wXcTraffic =
                                                                (long)CostUtility.GetMemoryAccess(call[GNNELSTM.WXc]
                                                                    .CheckedType) * inputShape[1] * inputShape[2];
                                                            long initialHTraffic =
                                                                (long)CostUtility.GetMemoryAccess(
                                                                    call[GNNELSTM.InitialH].CheckedType) *
                                                                (long)Math.Ceiling((double)wRcShape[2] /
                                                                    (double)GNNEEnv.PuWidth) * inputShape[1];
                                                            long wRcTraffic =
                                                                (long)CostUtility.GetMemoryAccess(call[GNNELSTM.WRc]
                                                                    .CheckedType) * inputShape[1] * inputShape[2];
                                                            Shape shape = new int[4]
                                                            {
                                                                wXcShape[1], wXcShape[2], inputShape[1], inputShape[2],
                                                            };
                                                            long wXcResultTraffic =
                                                                (long)CostUtility.GetMemoryAccess(
                                                                    new TensorType(DataTypes.Float16, shape));
                                                            Shape shape2 = new int[4]
                                                            {
                                                                wRcShape[1], wRcShape[2], inputShape[1], inputShape[2],
                                                            };
                                                            long wRcResultTraffic =
                                                                (long)CostUtility.GetMemoryAccess(
                                                                    new TensorType(DataTypes.Float16, shape2));
                                                            onChipTraffic = (long)CostUtility.GetMemoryAccess(
                                                                call[GNNELSTM.ActXc].CheckedType,
                                                                call[GNNELSTM.ActRc0].CheckedType,
                                                                call[GNNELSTM.ActRc1].CheckedType,
                                                                call[GNNELSTM.InitialC].CheckedType, call.CheckedType);
                                                            onChipTraffic += inputTraffic + wXcTraffic + initialHTraffic + wRcTraffic + wXcResultTraffic + wRcResultTraffic;
                                                            onChipTraffic += (long)inputShape[1] * (long)inputShape[2] * wXcShape[1] *
                                                                    (long)Math.Ceiling((double)wXcShape[2] /
                                                                        (double)GNNEEnv.PuWidth) * perTileTraffic;
                                                            onChipTraffic += (long)inputShape[1] * (long)inputShape[2] * wXcShape[1] *
                                                                    (long)Math.Ceiling((double)wXcShape[2] /
                                                                        (double)GNNEEnv.PuWidth) * perTileTraffic;
                                                            onChipTraffic += (long)inputShape[1] * (long)inputShape[2] * wRcShape[1] *
                                                                    (long)Math.Ceiling((double)wRcShape[2] /
                                                                        (double)GNNEEnv.PuWidth) * perTileTraffic;
                                                            onChipTraffic += (long)inputShape[1] * (long)inputShape[2] * wRcShape[1] *
                                                                    (long)Math.Ceiling((double)wRcShape[2] /
                                                                        (double)GNNEEnv.PuWidth) * perTileTraffic;
                                                            UInt128 uInt = (UInt128)inputShape[1];
                                                            UInt128 uInt2 = (UInt128)wXcShape[1];
                                                            UInt128 uInt3 = (UInt128)inputShape[2];
                                                            UInt128 uInt4 = (UInt128)wXcShape[3] / (byte)4;
                                                            UInt128 uInt5 = (UInt128)inputShape[^1];
                                                            UInt128 uInt6 = uInt4;
                                                            UInt128 uInt7 = uInt3;
                                                            UInt128 uInt8 = uInt;
                                                            flops = (ulong)(uInt2 * uInt7 * uInt8 *
                                                                           ((byte)4 * uInt6 * uInt5 +
                                                                            (byte)4 * uInt6 * uInt6));
                                                            value.AddOperator(
                                                                "GNNELSTM",
                                                                call[GNNELSTM.Input].CheckedShape
                                                                    .Select((Dimension x) => (" ", (ulong)x.FixedValue))
                                                                    .ToArray());
                                                        }
                                                    }
                                                    else
                                                    {
                                                        onChipTraffic = (long)CostUtility.GetMemoryAccess(
                                                            call[Ai2dResize.Input].CheckedType, call.CheckedType);
                                                        value.AddOperator(
                                                            "Ai2dResize",
                                                            call[Ai2dResize.Input].CheckedShape.Select((Dimension x) =>
                                                                (" ", (ulong)x.FixedValue)).ToArray());
                                                    }
                                                }
                                            }
                                            else
                                            {
                                                onChipTraffic = (long)CostUtility.GetMemoryAccess(
                                                    call[GNNETranspose.Input].CheckedType, call.CheckedType);
                                                value.AddOperator(
                                                    "GNNETranspose",
                                                    call[GNNETranspose.Input].CheckedShape
                                                        .Select((Dimension x) => (" ", (ulong)x.FixedValue)).ToArray());
                                            }
                                        }
                                        else
                                        {
                                            bool isUnary = call[GNNEActivation.InputB] == None.Default;
                                            bool hasLoadOrReshapeInput =
                                                (call[GNNEActivation.InputA] is Call call2 &&
                                                 call2.Target is GNNELoad) ||
                                                (call[GNNEActivation.InputB] is Call call3 &&
                                                 call3.Target is GNNELoad) ||
                                                (call[GNNEActivation.InputA] is Call call4 &&
                                                 call4.Target is Reshape) ||
                                                (call[GNNEActivation.InputB] is Call call5 && call5.Target is Reshape);
                                            onChipTraffic = (long)CostUtility.GetMemoryAccess(
                                                call[GNNEActivation.InputA].CheckedType,
                                                call[GNNEActivation.InputB].CheckedType, call.CheckedType);
                                            onChipTraffic += (hasLoadOrReshapeInput ? inputLoadTraffic : 0) + (isUnary ? (perTileTraffic * 2) : (perTileTraffic * 3));
                                            value.AddOperator(
                                                "GNNEActivation",
                                                call[GNNEActivation.InputA].CheckedShape
                                                    .Select((Dimension x) => (" ", (ulong)x.FixedValue)).ToArray());
                                        }
                                    }
                                    else
                                    {
                                        onChipTraffic = (long)CostUtility.GetMemoryAccess(
                                            call[GNNEPdp1.Input].CheckedType,
                                            call.CheckedType);
                                        value.AddOperator(
                                            "GNNEPdp1",
                                            call[GNNEPdp1.Input].CheckedShape
                                                .Select((Dimension x) => (" ", (ulong)x.FixedValue)).ToArray());
                                    }
                                }
                                else
                                {
                                    ulong outElements = call.CheckedShape.Aggregate(
                                        0uL,
                                        (ulong sum, Dimension x) => sum * (ulong)x.FixedValue);
                                    macs = (ulong)call[GNNEMatMul.InputA].CheckedShape[3].FixedValue * outElements;
                                    ulong[] inputAShape = call[GNNEMatMul.InputA].CheckedShape
                                        .Select((Dimension x) => (ulong)x.FixedValue).ToArray();
                                    ulong[] inputBShape = call[GNNEMatMul.InputB].CheckedShape
                                        .Select((Dimension x) => (ulong)x.FixedValue).ToArray();
                                    ulong[] outputShape = call.CheckedShape.Select((Dimension x) => (ulong)x.FixedValue)
                                        .ToArray();
                                    flops = (ulong)((long)(inputAShape.Aggregate(1uL, (ulong acc, ulong i) => acc * i) *
                                                          outputShape[^1]) *
                                                   (long)call[GNNEMatMul.InputA].CheckedDataType.SizeInBytes *
                                                   call[GNNEMatMul.InputB].CheckedDataType.SizeInBytes);
                                    onChipTraffic = (long)CostUtility.GetMemoryAccess(
                                        call[GNNEMatMul.InputA].CheckedType,
                                        call[GNNEMatMul.InputB].CheckedType, call.CheckedType);
                                    onChipTraffic += (long)Math.Max(inputAShape[0] * inputAShape[1], inputBShape[0] * inputBShape[1]) *
                                            (long)Math.Ceiling((double)inputAShape[3] / (double)GNNEEnv.PuHeight) *
                                            (long)Math.Ceiling((double)inputAShape[2] / (double)GNNEEnv.PuWidth) * perTileTraffic;
                                    onChipTraffic += (long)Math.Ceiling((double)outputShape[1] * (double)outputShape[2] /
                                                               (double)GNNEEnv.PuWidth) * perTileTraffic;
                                    value.AddOperator("GNNEMatMul", (from x in inputAShape.SkipLast(2)
                                        select (" ", x)).Concat(new (string, ulong)[2]
                                    {
                                        ("N", inputAShape[^2]), ("K", inputAShape[^1]),
                                    }).Concat((from x in inputBShape.SkipLast(2)
                                        select (" ", x)).Concat(new (string, ulong)[2]
                                    {
                                        ("K", inputBShape[^2]), ("M", inputBShape[^1]),
                                    })).ToArray());
                                }
                            }
                            else
                            {
                                ulong[] outputShape = call.CheckedShape.Select((Dimension d) => (ulong)d.FixedValue)
                                    .ToArray();
                                macs = (ulong)((long)call[GNNEPdp0DW.Input].CheckedShape[1].FixedValue *
                                              (long)call[GNNEPdp0DW.Weights].CheckedShape[2].FixedValue *
                                              call[GNNEPdp0DW.Weights].CheckedShape[3].FixedValue * (long)outputShape[2]) *
                                      outputShape[3];
                                ulong[] inputShape = call[GNNEPdp0DW.Input].CheckedShape
                                    .Select((Dimension x) => (ulong)x.FixedValue).ToArray();
                                ulong[] weightsShape = call[GNNEPdp0DW.Weights].CheckedShape
                                    .Select((Dimension x) => (ulong)x.FixedValue).ToArray();
                                flops = (ulong)((long)(inputShape[0] * weightsShape[0] * weightsShape[1] * outputShape[2] * outputShape[3] *
                                                      weightsShape[2] * weightsShape[3]) *
                                               (long)call[GNNEPdp0DW.Input].CheckedDataType.SizeInBytes *
                                               call[GNNEPdp0DW.Weights].CheckedDataType.SizeInBytes);
                                onChipTraffic = (long)CostUtility.GetMemoryAccess(
                                    call[GNNEPdp0DW.Input].CheckedType,
                                    call[GNNEPdp0DW.Weights].CheckedType, call[GNNEPdp0DW.Act].CheckedType,
                                    call.CheckedType);
                                onChipTraffic += (long)Math.Ceiling((double)inputShape[1] / (double)GNNEEnv.PuHeight) *
                                        (long)Math.Ceiling((double)inputShape[2] * (double)inputShape[3] /
                                                           (double)(GNNEEnv.PsumL1ElePerChan / 2)) * perTileTraffic;
                                onChipTraffic += (long)Math.Ceiling((double)outputShape[1] / (double)GNNEEnv.PuHeight) *
                                        (long)Math.Ceiling((double)outputShape[2] * (double)outputShape[3] /
                                                           (double)(GNNEEnv.PsumL1ElePerChan / 2)) * perTileTraffic;
                                offChipWeightTraffic = -(long)CostUtility.GetMemoryAccess(
                                    call[GNNEPdp0DW.WeightsQInt8].CheckedType,
                                    call[GNNEPdp0DW.WeightsBiasQint8].CheckedType,
                                    call[GNNEPdp0DW.ActQint8].CheckedType);
                                value.AddOperator("GNNEPdp0DW", ("B", inputShape[0]), ("OC", 1uL), ("IC", weightsShape[0]),
                                    ("OH", outputShape[2]), ("OW", outputShape[3]), ("KH", weightsShape[2]), ("KW", weightsShape[3]));
                            }
                        }
                        else
                        {
                            ulong[] outputShape = call.CheckedShape.Select((Dimension d) => (ulong)d.FixedValue).ToArray();
                            macs = (ulong)((long)call[GNNEConv2DTranspose.Input].CheckedShape[1].FixedValue *
                                          (long)call[GNNEConv2DTranspose.Weights].CheckedShape[2].FixedValue *
                                          call[GNNEConv2DTranspose.Weights].CheckedShape[3].FixedValue *
                                          call.CheckedShape[1].FixedValue *
                                          call[GNNEConv2DTranspose.Input].CheckedShape[2].FixedValue *
                                          call[GNNEConv2DTranspose.Input].CheckedShape[3].FixedValue);
                            ulong[] inputShape = call[GNNEConv2DTranspose.Input].CheckedShape
                                .Select((Dimension x) => (ulong)x.FixedValue).ToArray();
                            ulong[] weightsShape = call[GNNEConv2DTranspose.Weights].CheckedShape
                                .Select((Dimension x) => (ulong)x.FixedValue).ToArray();
                            flops = (ulong)((long)(inputShape[0] * weightsShape[0] * weightsShape[1] * inputShape[2] * inputShape[3] *
                                                  weightsShape[2] * weightsShape[3]) *
                                           (long)call[GNNEConv2DTranspose.Input].CheckedDataType.SizeInBytes *
                                           call[GNNEConv2DTranspose.Weights].CheckedDataType.SizeInBytes);
                            onChipTraffic = (long)CostUtility.GetMemoryAccess(
                                call[GNNEConv2DTranspose.Input].CheckedType,
                                call[GNNEConv2DTranspose.Weights].CheckedType,
                                call[GNNEConv2DTranspose.Act].CheckedType, call.CheckedType);
                            onChipTraffic += (long)inputShape[0] *
                                    (long)Math.Ceiling((double)inputShape[1] / (double)GNNEEnv.PuHeight) *
                                    (long)Math.Ceiling((double)weightsShape[0] / (double)GNNEEnv.PuWidth) *
                                    (long)Math.Ceiling((double)inputShape[2] * (double)inputShape[3] /
                                                       (double)GNNEEnv.PsumL1ElePerChan) * perTileTraffic;
                            onChipTraffic += (long)inputShape[0] *
                                    (long)Math.Ceiling((double)outputShape[1] / (double)GNNEEnv.PuWidth) * perTileTraffic;
                            offChipWeightTraffic = -(long)CostUtility.GetMemoryAccess(
                                call[GNNEConv2DTranspose.WeightsQInt8].CheckedType,
                                call[GNNEConv2DTranspose.WeightsBiasQint8].CheckedType,
                                call[GNNEConv2DTranspose.ActQint8].CheckedType);
                            offChipWeightTraffic += (long)inputShape[0] *
                                    (long)Math.Ceiling((double)inputShape[1] / (double)GNNEEnv.PuHeight) *
                                    (long)Math.Ceiling((double)weightsShape[0] / (double)GNNEEnv.PuWidth) *
                                    (long)Math.Ceiling((double)inputShape[2] * (double)inputShape[3] /
                                                       (double)GNNEEnv.PsumL1ElePerChan) * (long)weightsShape[2] *
                                    (long)weightsShape[3] * perKernelTapTraffic;
                            value.AddOperator("GNNEConv2DTranspose", ("B", inputShape[0]), ("OC", weightsShape[0]),
                                ("IC", weightsShape[1]), ("OH", outputShape[2]), ("OW", outputShape[3]), ("KH", weightsShape[2]),
                                ("KW", weightsShape[3]));
                        }
                    }
                    else
                    {
                        ulong[] inputShape = call[GNNEConv2D.Input].CheckedShape
                            .Select((Dimension d) => (ulong)d.FixedValue).ToArray();
                        ulong[] weightsShape = call[GNNEConv2D.Weights].CheckedShape
                            .Select((Dimension d) => (ulong)d.FixedValue).ToArray();
                        ulong[] outputShape = call.CheckedShape.Select((Dimension d) => (ulong)d.FixedValue).ToArray();
                        int groups = ((TensorConst)call[GNNEConv2D.Groups]).Value.ToScalar<int>();
                        ulong inChannels = inputShape[1];
                        ulong outChannels = outputShape[1];
                        bool isDepthwise = inChannels == outChannels && (int)outChannels == groups && groups != 1;
                        bool feedsPdp0Dw = call.Users.Count == 1 && call.Users.ElementAt(0) is Call call6 &&
                                     call6.Target is GNNEPdp0DW;
                        bool isBetweenLoadAndStore = call[GNNEConv2D.Input] is Call call7 && call7.Target is GNNELoad &&
                                     call.Users.ElementAt(0) is Call call8 && call8.Target is GNNEStore;
                        ulong convMacs = isDepthwise
                            ? (inputShape[1] * weightsShape[2] * weightsShape[3] * outputShape[2] * outputShape[3] +
                               inputShape[1] * outputShape[1] * outputShape[2] * outputShape[3])
                            : (inputShape[1] * weightsShape[2] * weightsShape[3] * outputShape[1] * outputShape[2] * outputShape[3]);
                        int strideH = ((TensorConst)call[GNNEConv2D.Stride]).Value.ToArray<int>()[0];
                        int strideW = ((TensorConst)call[GNNEConv2D.Stride]).Value.ToArray<int>()[1];
                        int dilationH = ((TensorConst)call[GNNEConv2D.Dilation]).Value.ToArray<int>()[0];
                        int dilationW = ((TensorConst)call[GNNEConv2D.Dilation]).Value.ToArray<int>()[1];
                        int[] paddings = ((TensorConst)call[GNNEConv2D.Padding]).Value.ToArray<int>();
                        Padding padding = new Padding(paddings[0], paddings[1]);
                        Padding padding2 = new Padding(paddings[2], paddings[3]);
                        bool isSmallPointwise = false;
                        if (!isDepthwise && groups == 1 && strideH == 1 && strideW == 1 && weightsShape[2] == 1 && weightsShape[3] == 1 &&
                            dilationH == 1 && dilationW == 1 && padding.Sum() == 0 && padding2.Sum() == 0 &&
                            (call[GNNEConv2D.Input].CheckedDataType == DataTypes.Int8 ||
                             call[GNNEConv2D.Input].CheckedDataType == DataTypes.UInt8) &&
                            outputShape[2] * outputShape[3] < 512)
                        {
                            isSmallPointwise = true;
                        }

                        flops = (ulong)((long)(inputShape[0] * weightsShape[0] * weightsShape[1] * outputShape[2] * outputShape[3] *
                                              weightsShape[2] * weightsShape[3]) *
                                       (long)call[GNNEConv2D.Input].CheckedDataType.SizeInBytes *
                                       call[GNNEConv2D.Weights].CheckedDataType.SizeInBytes);
                        macs = convMacs;
                        onChipTraffic = (long)CostUtility.GetMemoryAccess(
                            call[GNNEConv2D.Weights].CheckedType,
                            call[GNNEConv2D.Act].CheckedType, call.CheckedType);
                        onChipTraffic += (long)CostUtility.GetMemoryAccess(call[GNNEConv2D.Input].CheckedType) *
                                (long)Math.Ceiling((double)weightsShape[0] / (double)GNNEEnv.PuWidth);
                        if (isSmallPointwise)
                        {
                            onChipTraffic = (inputShape[0] > 1 && inputShape[2] == 1 && inputShape[3] == 1)
                                ? (onChipTraffic + (long)Math.Ceiling((double)weightsShape[0] / (double)GNNEEnv.PuWidth) *
                                    (long)Math.Ceiling((double)weightsShape[1] / (double)GNNEEnv.PuHeight) * perTileTraffic)
                                : ((!feedsPdp0Dw)
                                    ? (onChipTraffic + (long)inputShape[0] *
                                        (long)Math.Ceiling((double)weightsShape[0] / (double)GNNEEnv.PuWidth) *
                                        (long)Math.Ceiling((double)inputShape[2] * (double)inputShape[3] /
                                                           (double)GNNEEnv.PsumL1ElePerChan) * perTileTraffic)
                                    : (onChipTraffic + (long)inputShape[0] *
                                        (long)Math.Ceiling((double)weightsShape[0] / (double)GNNEEnv.PuWidth) *
                                        (long)Math.Ceiling((double)inputShape[2] * (double)inputShape[3] /
                                            (double)GNNEEnv.PsumL1ElePerChan * 2.0) * perTileTraffic));
                        }
                        else
                        {
                            if (isDepthwise)
                            {
                                onChipTraffic = (!feedsPdp0Dw)
                                    ? (onChipTraffic + (long)inputShape[0] *
                                        (long)Math.Ceiling((double)inputShape[1] / (double)GNNEEnv.PuHeight) *
                                        (long)Math.Ceiling((double)inputShape[2] * (double)inputShape[3] /
                                                           (double)GNNEEnv.PsumL1ElePerChan) * perTileTraffic)
                                    : (onChipTraffic + (long)inputShape[0] *
                                        (long)Math.Ceiling((double)inputShape[1] / (double)GNNEEnv.PuHeight) *
                                        (long)Math.Ceiling((double)inputShape[2] * (double)inputShape[3] /
                                            (double)GNNEEnv.PsumL1ElePerChan * 2.0) * perTileTraffic);
                            }
                            else if (feedsPdp0Dw)
                            {
                                onChipTraffic += (long)inputShape[0] *
                                        (long)Math.Ceiling((double)inputShape[1] / (double)GNNEEnv.PuHeight) *
                                        (long)Math.Ceiling((double)weightsShape[0] / (double)GNNEEnv.PuWidth) *
                                        (long)Math.Ceiling((double)inputShape[2] * (double)inputShape[3] /
                                            (double)GNNEEnv.PsumL1ElePerChan * 2.0) * perTileTraffic;
                            }
                            else
                            {
                                long kernelWPasses = (weightsShape[3] > 31)
                                    ? ((long)Math.Ceiling((double)TileUtilities.GetAlignedNum((int)weightsShape[3], 31) /
                                        (double)GNNEEnv.PuKernelSpad * 2.0))
                                    : 1;
                                onChipTraffic += (long)inputShape[0] *
                                        (long)Math.Ceiling((double)inputShape[1] / (double)GNNEEnv.PuHeight) *
                                        (long)Math.Ceiling((double)weightsShape[0] / (double)GNNEEnv.PuWidth) * kernelWPasses *
                                        (long)Math.Ceiling((double)inputShape[2] * (double)inputShape[3] /
                                                           (double)GNNEEnv.PsumL1ElePerChan) * perTileTraffic;
                            }

                            onChipTraffic *= (!isBetweenLoadAndStore) ? 1 : 12;
                        }

                        onChipTraffic += (long)Math.Ceiling((double)outputShape[1] / (double)GNNEEnv.PuWidth) * perTileTraffic;
                        offChipWeightTraffic = -(long)CostUtility.GetMemoryAccess(
                            call[GNNEConv2D.WeightsQInt8].CheckedType,
                            call[GNNEConv2D.WeightsBiasQint8].CheckedType, call[GNNEConv2D.ActQint8].CheckedType);
                        value.AddOperator("GNNEConv2D", ("B", inputShape[0]), ("OC", weightsShape[0]), ("IC", weightsShape[1]),
                            ("OH", outputShape[2]), ("OW", outputShape[3]), ("KH", weightsShape[2]), ("KW", weightsShape[3]));
                    }
                }
                else
                {
                    storeCount++;
                    onChipTraffic = (long)CostUtility.GetMemoryAccess(call[GNNEStore.Input].CheckedType);
                    offChipTraffic = (long)CostUtility.GetMemoryAccess(call.CheckedType);
                }
            }
            else
            {
                onChipTraffic = (long)CostUtility.GetMemoryAccess(call.CheckedType);
                offChipWeightTraffic = (long)CostUtility.GetMemoryAccess(call[GNNELoadW.Input].CheckedType);
            }
        }
        else
        {
            loadCount++;
            onChipTraffic = (long)CostUtility.GetMemoryAccess(call.CheckedType);
            offChipTraffic = (long)CostUtility.GetMemoryAccess(call[GNNELoad.Input].CheckedType);
        }

        value["Mac"] += macs;
        value["FLOPs"] += flops;
        value["OnChipMemTraffic"] += (ulong)onChipTraffic;
        value["OffChipMemTraffic"] += (ulong)offChipTraffic;
        value["OffChipMemLWTraffic"] += (ulong)offChipWeightTraffic;
        value["OffChipLoadStoreCnt"] += loadCount + storeCount;
        return default(Unit);
    }
}
