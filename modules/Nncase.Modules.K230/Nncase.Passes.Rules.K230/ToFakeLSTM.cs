// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.IR.RNN;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

/// <summary>
/// Rewrites a calibrated (range-of-marker wrapped) ONNX LSTM into a K230 FakeLSTM.
/// </summary>
[RuleGenerator]
public class ToFakeLSTM : RewriteRule<Pattern>
{
    // Hardware activation tables use 15 split points and 16 linear segments.
    private const int SplitPointCount = 15;
    private const int SegmentCount = 16;

    public override Pattern Pattern { get; } = Nncase.PatternMatch.Utility.IsWrappedLSTM(
        Nncase.PatternMatch.F.RNN.IsLSTM(
            "lstm",
            "call",
            (LSTM _) => true,
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "xMarker",
                Nncase.PatternMatch.Utility.IsWildcard("x"),
                Nncase.PatternMatch.Utility.IsConst("xRange")) with
            {
                TypePattern = TypePatternUtility.HasFixedShape(),
            },
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "wMarker",
                Nncase.PatternMatch.Utility.IsTensorConst("w"),
                Nncase.PatternMatch.Utility.IsConst("wRange")) with
            {
                TypePattern = TypePatternUtility.HasFixedShape(),
            },
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "rMarker",
                Nncase.PatternMatch.Utility.IsTensorConst("r"),
                Nncase.PatternMatch.Utility.IsConst("rRange")) with
            {
                TypePattern = TypePatternUtility.HasFixedShape(),
            },
            Nncase.PatternMatch.Utility.IsTensorConst("b"),
            Nncase.PatternMatch.Utility.IsTensorConst(), // sequence_lens
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "initHMarker",
                Nncase.PatternMatch.Utility.IsWildcard("initH"),
                Nncase.PatternMatch.Utility.IsConst("initHRange")) with
            {
                TypePattern = TypePatternUtility.HasFixedShape(),
            },
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "initCMarker",
                Nncase.PatternMatch.Utility.IsWildcard("initC"),
                Nncase.PatternMatch.Utility.IsConst("initCRange")) with
            {
                TypePattern = TypePatternUtility.HasFixedShape(),
            },
            Nncase.PatternMatch.Utility.IsTensorConst(), // p
            Nncase.PatternMatch.Utility.IsTensorConst(), // activation_alpha
            Nncase.PatternMatch.Utility.IsTensorConst(), // activation_beta
            Nncase.PatternMatch.Utility.IsTensorConst(), // clip
            Nncase.PatternMatch.Utility.IsTensorConst(), // hidden_size
            Nncase.PatternMatch.Utility.IsTensorConst(), // input_forget
            Nncase.PatternMatch.Utility.IsTensorConst("outputSize")),
        (Pattern output, int index) =>
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                $"outputMarker_{index}",
                output,
                Nncase.PatternMatch.Utility.IsWildcard()));

    /// <summary>
    /// Replacement body. Parameter names must match the capture names used in <see cref="Pattern"/>,
    /// because the rule generator binds them by name.
    /// </summary>
    private Expr? GetReplace(
        LSTM lstm,
        Call call,
        Expr x,
        TensorConst w,
        TensorConst r,
        Tensor<float> b,
        Expr initH,
        Expr initC,
        int outputSize,
        Marker xMarker,
        Marker wMarker,
        Marker rMarker,
        Marker initHMarker,
        Marker initCMarker,
        Tensor<float> xRange,
        Tensor<float> wRange,
        Tensor<float> rRange,
        Tensor<float> initHRange,
        Tensor<float> initCRange,
        IMatchResult result)
    {
        // Per-channel activation parameters for the two matmul stages (W*x and R*h).
        int channel = b.Shape[0].FixedValue * b.Shape[1].FixedValue / 2;
        var inputBiasAct = new ActParam2(channel, new QuantParam(0, 1f));
        var recurrentBiasAct = new ActParam2(channel, new QuantParam(0, 1f));

        // Shape[1] of the first output (Y) is the number of directions.
        int numDirections = ((TensorType)((TupleType)call.CheckedType)[0]).Shape[1].FixedValue;

        // Bias layout of B per direction: [Wb (4*hidden) | Rb (4*hidden)].
        // Flattened for two directions: fwd Wb, fwd Rb, bwd Wb, bwd Rb.
        int biasSegmentLength = K230Kernels.ComputeSize(b.Shape) / 2 / numDirections;
        float[] biasData = b.ToArray();
        bool isBidirectional = lstm.Direction == LSTMDirection.Bidirectional;

        List<float> inputBias = Slice(biasData, 0, biasSegmentLength);
        List<float> recurrentBias = Slice(biasData, biasSegmentLength, biasSegmentLength);
        if (isBidirectional)
        {
            inputBias.AddRange(Slice(biasData, biasSegmentLength * 2, biasSegmentLength));
            recurrentBias.AddRange(Slice(biasData, biasSegmentLength * 3, biasSegmentLength));
        }

        for (int i = 0; i < inputBias.Count; i++)
        {
            inputBiasAct.Bs[0, i] = inputBias[i];
            inputBiasAct.Bs[1, i] = inputBias[i];
        }

        for (int i = 0; i < recurrentBias.Count; i++)
        {
            recurrentBiasAct.Bs[0, i] = recurrentBias[i];
            recurrentBiasAct.Bs[1, i] = recurrentBias[i];
        }

        // Piecewise-linear activation tables for the gates (sigmoid) and cell/output (tanh).
        var sigmoidParam = new ActParam16(1);
        var tanhParam = new ActParam16(1);

        var sigmoidFunc = new ActFun
        {
            SplitPoint0 = -8f,
            SplitPoint14 = 8f,
            SplitPointCenter = 0f,
            CenterPoint = 7,
            MinParam = new List<float> { 0f, 0f },
            MaxParam = new List<float> { 0f, 1f },
            Func = v => 1f / (MathF.Exp(-v) + 1f),
        };

        var tanhFunc = new ActFun
        {
            SplitPoint0 = -4f,
            SplitPoint14 = 4f,
            SplitPointCenter = 0f,
            CenterPoint = 7,
            MinParam = new List<float> { 0f, -1f },
            MaxParam = new List<float> { 0f, 1f },
            Func = v => MathF.Tanh(v),
        };

        SetSegFittingParamSigmoid(sigmoidParam, sigmoidFunc);
        SetSegFittingParamTanh(tanhParam, tanhFunc);

        // Re-wrap every quantized input as a 4D tensor, keeping the original quant info.
        Marker fakeX = WrapWithRange(
            Nncase.IR.F.Tensors.Reshape(x, PadTo4D(x.CheckedShape)),
            xRange,
            xMarker);

        Marker fakeW = WrapWithRange(
            Tensor.FromBytes(w.CheckedDataType, w.Value.BytesBuffer.ToArray(), PadTo4D(w.CheckedShape)),
            wRange,
            wMarker);

        Marker fakeR = WrapWithRange(
            Tensor.FromBytes(r.CheckedDataType, r.Value.BytesBuffer.ToArray(), PadTo4D(r.CheckedShape)),
            rRange,
            rMarker);

        Expr fakeInitH = WrapInitialState(initH, initHRange, initHMarker);
        Expr fakeInitC = WrapInitialState(initC, initCRange, initCMarker);

        // Activation data for the W*x and R*h stages: [1, 1, num_directions * 4 * hidden, 7].
        Expr inputActivation = new Tensor<float>(
            inputBiasAct.GetAct0Data,
            new[] { 1, 1, numDirections * w.CheckedShape[1].FixedValue, 7 });
        Expr recurrentActivation = new Tensor<float>(
            recurrentBiasAct.GetAct0Data,
            new[] { 1, 1, numDirections * r.CheckedShape[1].FixedValue, 7 });

        Call fakeLstm = Nncase.IR.K230.F.Tensors.FakeLSTM(
            fakeX,
            fakeW,
            inputActivation,
            fakeR,
            recurrentActivation,
            fakeInitH,
            fakeInitC,
            new Tensor<float>(sigmoidParam.GetAct1Data, new[] { 1, 1, 1, 49 }),
            new Tensor<float>(tanhParam.GetAct1Data, new[] { 1, 1, 1, 49 }),
            false,
            lstm.Direction,
            outputSize,
            inputBiasAct,
            recurrentBiasAct);

        Shape[] originalOutputShapes = ((TupleType)call.CheckedType)
            .Select((IRType type) => ((TensorType)type).Shape)
            .ToArray();

        return WrapOutput(fakeLstm, outputSize, originalOutputShapes, result);
    }

    /// <summary>
    /// Takes each tuple field of the fake op, restores the original shape and
    /// puts it back into the matching output marker.
    /// </summary>
    private static Nncase.IR.Tuple WrapOutput(Call fakeLstm, int count, Shape[] originalShapes, IMatchResult result)
    {
        var fields = new Expr[count];
        for (int i = 0; i < count; i++)
        {
            Call item = Nncase.IR.F.Tensors.GetItem(fakeLstm, i);
            fields[i] = ((Marker)result[$"outputMarker_{i}"]).With(
                null,
                Nncase.IR.F.Tensors.Reshape(item, originalShapes[i]));
        }

        return new Nncase.IR.Tuple(fields);
    }

    /// <summary>
    /// Wraps initial hidden/cell state as a 4D range-of-marker. Constants are re-materialized
    /// as 4D tensors, anything else is reshaped.
    /// </summary>
    private static Marker WrapInitialState(Expr state, Tensor<float> range, Marker source)
    {
        int[] shape4D = PadTo4D(state.CheckedShape);

        if (state is TensorConst tensorConst && tensorConst.Value != null)
        {
            Tensor value = tensorConst.Value;
            return WrapWithRange(
                Tensor.FromBytes(value.ElementType, value.BytesBuffer.ToArray(), shape4D),
                range,
                source);
        }

        return WrapWithRange(Nncase.IR.F.Tensors.Reshape(state, shape4D), range, source);
    }

    private static Marker WrapWithRange(Expr data, Tensor<float> range, Marker source)
    {
        return Nncase.IR.F.Math.RangeOfMarker(data, range).With(
            null,
            null,
            null,
            adaQuantInfo: source.AdaQuantInfo,
            mixQuantInfo: source.MixQuantInfo);
    }

    /// <summary>
    /// Pads the dimensions into the trailing three slots of a [1, 1, 1, 1] shape.
    /// </summary>
    private static int[] PadTo4D(Shape shape)
    {
        int[] dims = shape.ToValueArray();
        var padded = new[] { 1, 1, 1, 1 };
        Array.Copy(dims, 0, padded, 1, 3);
        return padded;
    }

    private static List<float> Slice(float[] data, int start, int length)
    {
        return data.Skip(start).Take(length).ToList();
    }

    // Note: the 'f' argument of both SetSegFittingParamSigmoid/Tanh is unused;
    // the segment tables below are precomputed and hardcoded.
    private void SetSegFittingParamSigmoid(ActParam16 actParam, ActFun f)
    {
        float[,] splitPoints = actParam.Xs;
        float[] breakpoints = new float[SplitPointCount]
        {
            -7f, -4.5f, -3.5f, -2.7f, -2.1f, -1.6f, -1f, 0f, 1f, 1.6f, 2.1f, 2.7f, 3.5f, 4.5f, 7f,
        };

        for (int i = 0; i < SplitPointCount; i++)
        {
            splitPoints[i, 0] = breakpoints[i];
        }

        // (slope, intercept) pairs for each of the 16 segments.
        double[] slopeInterceptPairs = new double[SegmentCount * 2]
        {
            0.0005523135475095087, 0.004717946782565874, 0.003582545941984816, 0.024643784893781717,
            0.017628076952972527, 0.08920121475552378, 0.04080084671519202, 0.17057512199171598,
            0.07504241042393778, 0.2641958959068108, 0.11550039574401527, 0.35039917518496155,
            0.16589656476120918, 0.4312276071962273, 0.23123362875892262, 0.4955178069668943,
            0.23398496370676491, 0.5031840614393268, 0.17066333504274322, 0.5625762717242175,
            0.1197647477013204, 0.6417049229131112, 0.07822321089963047, 0.7281582013897308,
            0.04270052410967129, 0.8235195920244173, 0.01849619693381177, 0.9073131477622514,
            0.0037644096557241102, 0.9742926548134362, 0.000567715948649905, 0.9951637114353361,
        };

        for (int segment = 0; segment < SegmentCount; segment++)
        {
            actParam.Ks[segment, 0] = (float)slopeInterceptPairs[segment * 2];
            actParam.Bs[segment, 0] = (float)slopeInterceptPairs[segment * 2 + 1];
        }

        actParam.SetFusedClamp(new ValueRange<float>(0f, 1f));
    }

    private void SetSegFittingParamTanh(ActParam16 actParam, ActFun f)
    {
        float[,] splitPoints = actParam.Xs;
        float[] breakpoints = new float[SplitPointCount]
        {
            -3.1f, -2.28f, -1.76f, -1.438f, -1.122f, -0.82f, -0.51f, 0f, 0.47f, 0.81f, 1.125f, 1.432f, 1.77f, 2.28f,
            3.1f,
        };

        for (int i = 0; i < SplitPointCount; i++)
        {
            splitPoints[i, 0] = breakpoints[i];
        }

        // (slope, intercept) pairs for each of the 16 segments.
        double[] slopeInterceptPairs = new double[SegmentCount * 2]
        {
            0.0009063486449397695, -0.995190713545316, 0.019181480050426303, -0.9381162870143267,
            0.06880845134681457, -0.8250607603958839, 0.1518869708683811, -0.6772576184059398,
            0.27007052179269864, -0.5091301547712312, 0.4374196151711459, -0.3220241963133208,
            0.6540450071930701, -0.14398516081895907, 0.9192436825013623, -0.010453854050918476,
            0.9412592709759997, 0.005250815043192469, 0.6730960939548211, 0.13033188197303736,
            0.43741961517114625, 0.32202419631332024, 0.2700705217926983, 0.5091301547712318,
            0.151886970868381, 0.6772576184059396, 0.0688084513468159, 0.8250607603958814,
            0.01918148005042708, 0.9381162870143249, 0.0009063486449409908, 0.9951907135453116,
        };

        for (int segment = 0; segment < SegmentCount; segment++)
        {
            actParam.Ks[segment, 0] = (float)slopeInterceptPairs[segment * 2];
            actParam.Bs[segment, 0] = (float)slopeInterceptPairs[segment * 2 + 1];
        }

        actParam.SetFusedClamp(new ValueRange<float>(-1f, 1f));
    }

    /// <summary>
    /// Generic segment fitting for an arbitrary activation function (currently unused by this rule).
    /// </summary>
    private void SetSegFittingParam(ActParam16 actParam, ActFun f)
    {
        const int column = 0;
        float[,] splitPoints = actParam.Xs;
        int centerPoint = f.CenterPoint;

        splitPoints[0, column] = f.SplitPoint0;
        splitPoints[14, column] = f.SplitPoint14;
        splitPoints[centerPoint, column] = f.SplitPointCenter;

        // Evenly spaced points between the left end and the center...
        for (int i = 1; i < centerPoint; i++)
        {
            splitPoints[i, column] =
                ((splitPoints[centerPoint, column] - splitPoints[0, column]) / centerPoint * i) + splitPoints[0, column];
        }

        // ...and between the center and the right end.
        for (int i = centerPoint + 1; i < 15; i++)
        {
            splitPoints[i, column] =
                ((splitPoints[14, column] - splitPoints[centerPoint, column]) / (14 - centerPoint) * (i - centerPoint))
                + splitPoints[centerPoint, column];
        }

        // Outer segments are constants (slope/intercept taken from Min/MaxParam).
        actParam.Ks[0, column] = f.MinParam[0];
        actParam.Bs[0, column] = f.MinParam[1];
        actParam.Ks[15, column] = f.MaxParam[0];
        actParam.Bs[15, column] = f.MaxParam[1];

        // Inner segments are secant lines through consecutive split points.
        for (int i = 1; i < 15; i++)
        {
            float slope = (f.Func(splitPoints[i, column]) - f.Func(splitPoints[i - 1, column]))
                          / (splitPoints[i, column] - splitPoints[i - 1, column]);
            float intercept = f.Func(splitPoints[i, column]) - (slope * splitPoints[i, column]);
            actParam.Ks[i, column] = slope;
            actParam.Bs[i, column] = intercept;
        }

        actParam.SetFusedClamp(ValueRange<float>.Full);
    }

    // Generated by [RuleGenerator]: binds captures by name and forwards to the private GetReplace above.
    // Remove this override if you put the file back into a project that runs the generator.
    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        LSTM lstm = (LSTM)__result["lstm"];
        Call call = (Call)__result["call"];
        Expr x = (Expr)__result["x"];
        TensorConst w = (TensorConst)__result["w"];
        TensorConst r = (TensorConst)__result["r"];
        Tensor<float> b = ((TensorConst)__result["b"]).Value.Cast<float>();
        Expr initH = (Expr)__result["initH"];
        Expr initC = (Expr)__result["initC"];
        int outputSize = ((TensorConst)__result["outputSize"]).Value.ToScalar<int>();
        Marker xMarker = (Marker)__result["xMarker"];
        Marker wMarker = (Marker)__result["wMarker"];
        Marker rMarker = (Marker)__result["rMarker"];
        Marker initHMarker = (Marker)__result["initHMarker"];
        Marker initCMarker = (Marker)__result["initCMarker"];
        Tensor<float> xRange = ((TensorConst)__result["xRange"]).Value.Cast<float>();
        Tensor<float> wRange = ((TensorConst)__result["wRange"]).Value.Cast<float>();
        Tensor<float> rRange = ((TensorConst)__result["rRange"]).Value.Cast<float>();
        Tensor<float> initHRange = ((TensorConst)__result["initHRange"]).Value.Cast<float>();
        Tensor<float> initCRange = ((TensorConst)__result["initCRange"]).Value.Cast<float>();
        return GetReplace(
            lstm,
            call,
            x,
            w,
            r,
            b,
            initH,
            initC,
            outputSize,
            xMarker,
            wMarker,
            rMarker,
            initHMarker,
            initCMarker,
            xRange,
            wRange,
            rRange,
            initHRange,
            initCRange,
            __result);
    }
}
