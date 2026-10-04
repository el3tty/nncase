using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using Nncase.Utilities;

namespace Nncase.Passes.Rules.K230;

/// <summary>
/// Lowers a FakeLSTM (float simulation of the hardware op) into the real GNNE LSTM:
/// quantizes weights, builds the fp16 per-channel activation parameter blocks and
/// wraps the op with GNNELoad / GNNELoadW / GNNEStore.
/// </summary>
[RuleGenerator]
public sealed class ToGNNELSTM : IRewriteRule
{
    // Quantizers are created in LowerInit. The names follow the matmul they quantize:
    //   input  = input feature (x) side, hidden = initial h state,
    //   wx     = W*x matmul,             wr     = R*h matmul,
    //   output = output feature side.
    private QuantizeManager? _inputQuantizer;

    private QuantizeManager? _hiddenQuantizer;

    private QuantizeManager? _wxQuantizer;

    private QuantizeManager? _wrQuantizer;

    private QuantizeManager? _outputQuantizer;

    private ConvQuantConf? _conf;

    public IPattern Pattern { get; } = Nncase.PatternMatch.Utility.IsWrappedLSTM(
        Nncase.PatternMatch.F.K230.IsFakeLSTM(
            "fakeLstm",
            "call",
            (FakeLSTM _) => true,
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "inputMarker",
                Nncase.PatternMatch.Utility.IsWildcard("input"),
                Nncase.PatternMatch.Utility.IsConst("inputRange")),
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "wXcMarker",
                Nncase.PatternMatch.Utility.IsTensorConst("wXc"),
                Nncase.PatternMatch.Utility.IsConst("wXcRange")),
            Nncase.PatternMatch.Utility.IsTensorConst("actXc"),
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "wRcMarker",
                Nncase.PatternMatch.Utility.IsTensorConst("wRc"),
                Nncase.PatternMatch.Utility.IsConst("wRcRange")),
            Nncase.PatternMatch.Utility.IsTensorConst("actRc"),
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "initialHMarker",
                Nncase.PatternMatch.Utility.IsWildcard("initialH"),
                Nncase.PatternMatch.Utility.IsConst("initialHRange")),
            Nncase.PatternMatch.Utility.IsRangeOfMarker(
                "initialCMarker",
                Nncase.PatternMatch.Utility.IsWildcard("initialC"),
                Nncase.PatternMatch.Utility.IsConst("initialCRange")),
            Nncase.PatternMatch.Utility.IsTensorConst("segFittingParamFt"),
            Nncase.PatternMatch.Utility.IsTensorConst("segFittingParamGt"),
            Nncase.PatternMatch.Utility.IsTensorConst("hasStatic"),
            Nncase.PatternMatch.Utility.IsTensorConst("outputSize")),
        (Pattern output, int index) => Nncase.PatternMatch.Utility.IsRangeOfMarker(
            $"outputMarker_{index}",
            Nncase.PatternMatch.Utility.IsAlt(
                Nncase.PatternMatch.F.Tensors.IsReshape(
                    output,
                    Nncase.PatternMatch.Utility.IsTensorConst($"shapes_{index}")),
                output),
            Nncase.PatternMatch.Utility.IsTensorConst($"outputRange_{index}")));

    /// <summary>
    /// Replacement body. Parameter names must match the capture names used in <see cref="Pattern"/>,
    /// because the rule generator binds them by name. Several parameters are unused but kept for that reason.
    /// </summary>
    private Expr? GetReplace(
        FakeLSTM fakeLstm,
        Call call,
        Expr input,
        Expr inputMarker,
        Expr initialHMarker,
        Expr initialCMarker,
        Tensor<float> inputRange,
        Tensor wXc,
        Tensor<float> wXcRange,
        Tensor<float> wRcRange,
        Tensor wRc,
        Tensor<float> initialHRange,
        Tensor<float> initialCRange,
        Expr wRcMarker,
        Expr wXcMarker,
        Expr initialH,
        Expr initialC,
        Expr segFittingParamFt,
        Expr segFittingParamGt,
        Tensor<float> outputRange_0,
        Expr hasStatic,
        int outputSize,
        IMatchResult result)
    {
        LowerInit(fakeLstm, inputMarker, inputRange, wXcMarker, initialHRange, initialH, wXc, wRc, outputRange_0);

        ConvQuantConf conf = _conf!;
        QuantizeManager inputQuantizer = _inputQuantizer!;
        QuantizeManager hiddenQuantizer = _hiddenQuantizer!;
        QuantizeManager wxQuantizer = _wxQuantizer!;
        QuantizeManager wrQuantizer = _wrQuantizer!;
        QuantizeManager outputQuantizer = _outputQuantizer!;
        var quantType = (PrimType)conf.QuantType;
        var weightQuantType = (PrimType)conf.WQuantType;

        // ---- Activations: input x ------------------------------------------------------------
        DeQuantizeParam inputDeqParam = inputQuantizer.GetIfDeqQuantParam(conf.QuantType);
        Call loadedInput = Nncase.IR.K230.F.Tensors.GNNELoad(
            quantType,
            Nncase.IR.F.Math.Quantize(
                inputMarker,
                new QuantParam(inputDeqParam.ZeroPoint, inputDeqParam.Scale),
                quantType));

        // ---- Initial hidden state h ----------------------------------------------------------
        // A constant h is quantized like a weight and loaded with the weight type;
        // a dynamic h is quantized like an activation.
        QuantizeParam initialHQuantParam;
        Call loadedInitialH;
        if (initialH is TensorConst initialHConst)
        {
            initialHQuantParam = hiddenQuantizer.GetWeightsQuantParam()[0];
            loadedInitialH = Nncase.IR.K230.F.Tensors.GNNELoad(
                input: QuantizeConstantHidden(hiddenQuantizer, conf.WQuantType, initialHConst.Value.Shape),
                destType: weightQuantType);
        }
        else
        {
            DeQuantizeParam hiddenDeqParam = hiddenQuantizer.GetIfDeqQuantParam(conf.QuantType);
            loadedInitialH = Nncase.IR.K230.F.Tensors.GNNELoad(
                quantType,
                Nncase.IR.F.Math.Quantize(
                    initialH,
                    new QuantParam(hiddenDeqParam.ZeroPoint, hiddenDeqParam.Scale),
                    quantType));
            initialHQuantParam = new QuantizeParam(hiddenDeqParam.ZeroPoint, 1f / hiddenDeqParam.Scale);
        }

        // ---- Initial cell state c (kept in fp16, never quantized) ----------------------------
        Call loadedInitialC;
        if (initialC is TensorConst initialCConst)
        {
            Tensor initialCValue = initialCConst.Value;
            Half[] cellValues = new Half[K230Kernels.ComputeSize(initialCValue.Shape)];
            Array.Copy(initialCValue.ToArray<Half>(), cellValues, cellValues.Length);
            loadedInitialC = Nncase.IR.K230.F.Tensors.GNNELoad(
                DataTypes.Float16,
                Tensor.From(cellValues, initialCValue.Shape));
        }
        else
        {
            loadedInitialC = Nncase.IR.K230.F.Tensors.GNNELoad(DataTypes.Float16, initialC);
        }

        // ---- Quantized weights and their per-channel quantization arguments ------------------
        // Note: the W*x weights and "qarg" bytes come from inputQuantizer (as in the original rule),
        // while the activation parameters for W*x further down come from wxQuantizer.
        Call loadedWx = LoadQuantizedWeights(inputQuantizer, wXc, conf.WQuantType);
        Call wXcQarg = LoadQuantBias(inputQuantizer, conf.WQuantType);
        Call loadedWr = LoadQuantizedWeights(wrQuantizer, wRc, conf.WQuantType);
        Call wRcQarg = LoadQuantBias(wrQuantizer, conf.WQuantType);

        // ---- fp16 per-channel activation parameters ------------------------------------------
        // W*x stage: input scale * per-channel weight scale (+ bias from FakeLSTM).
        var wxAct = new ActParam2(fakeLstm.ActParamXc);
        DeQuantizeParam wxInputDeqParam = wxQuantizer.GetIfDeqQuantParam(conf.QuantType);
        float[] wxWeightsDeqScale = wxQuantizer.GetWeightsDeqScale();
        wxAct.FusedScale(wxInputDeqParam.Scale);
        wxAct.FusedChannelScale(wxWeightsDeqScale);
        Call actXc = Nncase.IR.K230.F.Tensors.GNNELoadW(DataTypes.Float16, wxAct.ToAct0Data());

        // R*h stage: h scale * per-channel weight scale (+ recurrent bias).
        var wrAct = new ActParam2(fakeLstm.ActParamRc);
        QuantizeParam hiddenScaleParam = initialH is TensorConst
            ? hiddenQuantizer.GetWeightsQuantParam()[0]
            : hiddenQuantizer.GetIfQuantParam(conf.QuantType);
        float[] wrWeightsDeqScale = wrQuantizer.GetWeightsDeqScale();
        wrAct.FusedScale(1f / hiddenScaleParam.Scale);
        wrAct.FusedChannelScale(wrWeightsDeqScale);
        Call actRc = Nncase.IR.K230.F.Tensors.GNNELoadW(DataTypes.Float16, wrAct.ToAct0Data());

        // Output side of the R*h stage: output scale * per-channel weight scale.
        var wrOutputAct = new ActParam2(fakeLstm.ActParamRc);
        DeQuantizeParam wrOutputDeqParam = wrQuantizer.GetOfDeqQuantParam(conf.QuantType);
        QuantizeParam wrOutputQuantParam = wrQuantizer.GetOfQuantParam(conf.QuantType);
        float[] wrOutputWeightsDeqScale = wrQuantizer.GetWeightsDeqScale();
        wrOutputAct.FusedScale(wrOutputDeqParam.Scale);
        wrOutputAct.FusedChannelScale(wrOutputWeightsDeqScale);
        Call actRcOut = Nncase.IR.K230.F.Tensors.GNNELoadW(DataTypes.Float16, wrOutputAct.ToAct0Data());

        // Piecewise-linear activation tables (built by ToFakeLSTM), passed through as fp16.
        Call loadedSegFt = Nncase.IR.K230.F.Tensors.GNNELoadW(
            DataTypes.Float16,
            ((TensorConst)segFittingParamFt).Value.Cast<Half>());
        Call loadedSegGt = Nncase.IR.K230.F.Tensors.GNNELoadW(
            DataTypes.Float16,
            ((TensorConst)segFittingParamGt).Value.Cast<Half>());

        // Per-channel parameters of the final stage, sized by the last dim of the first output (Y).
        Shape outputShape = ((TensorType)((TupleType)call.CheckedType)[0]).Shape;
        int outputChannels = outputShape[outputShape.Count - 1].FixedValue;

        var binAct = new ActParam2(outputChannels);
        Call actBin = Nncase.IR.K230.F.Tensors.GNNELoadW(DataTypes.Float16, binAct.ToAct1Data());

        // Final stage that quantizes the fp16 result back to the integer output type.
        var binQuantAct = new ActParam2(outputChannels);
        QuantizeParam finalOutputQuantParam = outputQuantizer.GetOfQuantParam(conf.QuantType);
        binQuantAct.SetFusedClamp(ValueRange<Half>.Full);
        binQuantAct.FusedQuantParam(new QuantParam(finalOutputQuantParam.ZeroPoint, 1f / finalOutputQuantParam.Scale));
        Call actBinQ = Nncase.IR.K230.F.Tensors.GNNELoadW(DataTypes.Float16, binQuantAct.ToAct1Data());

        // ---- The hardware op -----------------------------------------------------------------
        // The run of integer arguments after the activation parameter blocks are zero points
        // (input, h, output) and slots that are always 0; their exact meaning isn't visible here.
        Call gnneLstm = Nncase.IR.K230.F.Tensors.GNNELSTM(
            quantType,
            quantType,
            loadedInput,
            loadedWx,
            actXc,
            loadedWr,
            actRc,
            actRcOut,
            loadedInitialH,
            loadedInitialC,
            loadedSegFt,
            loadedSegGt,
            wXcQarg,
            wRcQarg,
            actBin,
            actBinQ,
            wxAct,
            wrAct,
            wrOutputAct,
            inputDeqParam.ZeroPoint,
            0,
            initialHQuantParam.ZeroPoint,
            wrOutputQuantParam.ZeroPoint,
            0,
            0,
            0,
            0,
            0,
            binAct,
            binQuantAct,
            fakeLstm.Direction,
            hasStatic,
            outputSize);

        // ---- Restore original output shapes --------------------------------------------------
        // Each output was either Reshape(t, const shape) or t itself; the Alt pattern captures
        // "shapes_{i}" only in the first case.
        int[][] outputShapes = new int[outputSize][];
        for (int i = 0; i < outputSize; i++)
        {
            try
            {
                var shapeConst = (TensorConst)result[$"shapes_{i}"];
                outputShapes[i] = shapeConst.Value.ToArray<int>();
            }
            catch (KeyNotFoundException)
            {
                outputShapes[i] = ((Marker)result[$"outputMarker_{i}"]).CheckedShape.ToValueArray();
            }
        }

        return WrapOutput(gnneLstm, outputSize, outputShapes, conf.QuantType, finalOutputQuantParam);
    }

    /// <summary>
    /// Creates one quantizer per hardware stage. Quantization types come from the marker's
    /// MixQuantInfo when bound by the cosine search, otherwise from default compile options.
    /// </summary>
    private void LowerInit(
        FakeLSTM fakeLstm,
        Expr inputMarker,
        Tensor<float> inputRange,
        Expr wXcMarker,
        Tensor<float> initialHRange,
        Expr initialH,
        Tensor wXc,
        Tensor wRc,
        Tensor<float> outputRange)
    {
        int numDirections = fakeLstm.Direction == LSTMDirection.Bidirectional ? 2 : 1;
        Tensor<float> wxRangeByChannel = ReplaceWeightRangeToByChannel(wXc, numDirections);
        Tensor<float> wrRangeByChannel = ReplaceWeightRangeToByChannel(wRc, numDirections);

        var quantizeOptions = new CompileOptions().QuantizeOptions;
        MixQuantInfo? inputMixQuantInfo = ((Marker)inputMarker).MixQuantInfo;
        MixQuantInfo? wxMixQuantInfo = ((Marker)wXcMarker).MixQuantInfo;

        DataType quantType = inputMixQuantInfo?.MarkerQuantType ?? quantizeOptions.QuantType;
        DataType weightQuantType = wxMixQuantInfo?.MarkerQuantType ?? quantizeOptions.WQuantType;

        _conf = new ConvQuantConf
        {
            QuantType = quantType,
            UseMseQuantW = false,
            WQuantType = weightQuantType,
        };

        _inputQuantizer = new QuantizeManager(inputRange, wxRangeByChannel, outputRange, wXc, _conf, is_matmul: true);

        // A constant h is treated as the "weights" of a non-matmul quantizer with a dummy [1, 2] range.
        _hiddenQuantizer = new QuantizeManager(
            initialHRange,
            initialH is TensorConst ? new Tensor<float>(initialHRange.ToArray(), new[] { 1, 2 }) : wrRangeByChannel,
            outputRange,
            initialH is TensorConst initialHConst ? initialHConst.Value : wRc,
            _conf,
            is_matmul: false);

        _wxQuantizer = new QuantizeManager(inputRange, wxRangeByChannel, outputRange, wXc, _conf, is_matmul: true);
        _wrQuantizer = new QuantizeManager(initialHRange, wrRangeByChannel, outputRange, wRc, _conf, is_matmul: true);
        _outputQuantizer = new QuantizeManager(inputRange, wxRangeByChannel, outputRange, wXc, _conf, is_matmul: true);
    }

    /// <summary>
    /// Quantizes a matmul weight tensor with the given quantizer and loads it into the NPU.
    /// </summary>
    private static Call LoadQuantizedWeights(QuantizeManager quantizer, Tensor weights, DataType weightType)
    {
        Tensor quantized;
        if (weightType == DataTypes.UInt8)
        {
            quantized = Tensor.From(quantizer.GetQuantWeights(isMatmul: true).Select(w => w.U8).ToArray(), weights.Shape);
        }
        else if (weightType == DataTypes.Int8)
        {
            quantized = Tensor.From(quantizer.GetQuantWeights(isMatmul: true).Select(w => w.I8).ToArray(), weights.Shape);
        }
        else
        {
            quantized = Tensor.From(quantizer.GetQuantWeightsI16(isMatmul: true).ToArray(), weights.Shape);
        }

        return Nncase.IR.K230.F.Tensors.GNNELoadW(input: quantized, destType: (PrimType)weightType);
    }

    /// <summary>
    /// Loads the per-channel quantization argument bytes of a weight tensor. They are zeroed for int16 weights.
    /// </summary>
    private static Call LoadQuantBias(QuantizeManager quantizer, DataType weightType)
    {
        byte[] biasBytes = quantizer.GetWeightsQuantBiasQint8();
        if (weightType == DataTypes.Int16)
        {
            biasBytes = new byte[biasBytes.Length];
        }

        Tensor<byte> biasTensor = Tensor.From(biasBytes.ToArray(), new[] { 1, 1, 1, biasBytes.Length });
        return Nncase.IR.K230.F.Tensors.GNNELoadW(DataTypes.UInt8, biasTensor);
    }

    /// <summary>
    /// Quantizes a constant initial hidden state the same way weights are quantized.
    /// </summary>
    private static Tensor QuantizeConstantHidden(QuantizeManager quantizer, DataType weightType, Shape shape)
    {
        if (weightType == DataTypes.Int16)
        {
            return Tensor.From(quantizer.GetQuantWeightsI16().ToArray(), shape);
        }

        if (weightType == DataTypes.Int8)
        {
            return Tensor.From(quantizer.GetQuantWeights().Select(w => w.I8).ToArray(), shape);
        }

        return Tensor.From(quantizer.GetQuantWeights().Select(w => w.U8).ToArray(), shape);
    }

    /// <summary>
    /// Stores each tuple field, restores its original shape and dequantizes it back to float32.
    /// The third output (cell state) is stored as float32 and is not quantized.
    /// </summary>
    private static Nncase.IR.Tuple WrapOutput(
        Call gnneLstm,
        int outputSize,
        int[][] outputShapes,
        DataType quantType,
        QuantizeParam outputQuantParam)
    {
        var storeFields = new Expr[outputSize];
        for (int i = 0; i < outputSize; i++)
        {
            Call item = Nncase.IR.F.Tensors.GetItem(gnneLstm, i);
            storeFields[i] = i == 2
                ? Nncase.IR.K230.F.Tensors.GNNEStore(DataTypes.Float32, item)
                : Nncase.IR.K230.F.Tensors.GNNEStore(quantType, item);
        }

        var stores = new Nncase.IR.Tuple(storeFields);

        var fields = new Expr[outputSize];
        for (int i = 0; i < outputSize; i++)
        {
            Call stored = Nncase.IR.F.Tensors.GetItem(stores, i);
            fields[i] = i == 2
                ? Nncase.IR.F.Tensors.Reshape(stored, outputShapes[2])
                : Nncase.IR.F.Math.Dequantize(
                    Nncase.IR.F.Tensors.Reshape(stored, outputShapes[i]),
                    new QuantParam(outputQuantParam.ZeroPoint, 1f / outputQuantParam.Scale),
                    DataTypes.Float32);
        }

        return new Nncase.IR.Tuple(fields);
    }

    /// <summary>
    /// Computes a [channels, 2] (min, max) range per output channel of the weights.
    /// Weights are 4D [1, num_directions, 4 * hidden, input], so channels = num_directions * 4 * hidden.
    /// </summary>
    private Tensor<float> ReplaceWeightRangeToByChannel(Tensor weights, int numDirections)
    {
        float[] values = weights.ToArray<float>();
        int channelsPerDirection = weights.Shape[2].FixedValue;
        int channels = channelsPerDirection * numDirections;
        return new Tensor<float>(
            QuantUtility.GetWeightsRangesByChannel(values, channels).ToArray(),
            new[] { channels, 2 });
    }

    // Generated by [RuleGenerator]: binds captures by name and forwards to the private GetReplace above.
    // Remove this method if you put the file back into a project that runs the generator.
    public Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        FakeLSTM fakeLstm = (FakeLSTM)__result["fakeLstm"];
        Call call = (Call)__result["call"];
        Expr input = (Expr)__result["input"];
        Expr inputMarker = (Expr)__result["inputMarker"];
        Expr initialHMarker = (Expr)__result["initialHMarker"];
        Expr initialCMarker = (Expr)__result["initialCMarker"];
        Tensor<float> inputRange = ((TensorConst)__result["inputRange"]).Value.Cast<float>();
        Tensor wXc = ((TensorConst)__result["wXc"]).Value;
        Tensor<float> wXcRange = ((TensorConst)__result["wXcRange"]).Value.Cast<float>();
        Tensor<float> wRcRange = ((TensorConst)__result["wRcRange"]).Value.Cast<float>();
        Tensor wRc = ((TensorConst)__result["wRc"]).Value;
        Tensor<float> initialHRange = ((TensorConst)__result["initialHRange"]).Value.Cast<float>();
        Tensor<float> initialCRange = ((TensorConst)__result["initialCRange"]).Value.Cast<float>();
        Expr wRcMarker = (Expr)__result["wRcMarker"];
        Expr wXcMarker = (Expr)__result["wXcMarker"];
        Expr initialH = (Expr)__result["initialH"];
        Expr initialC = (Expr)__result["initialC"];
        Expr segFittingParamFt = (Expr)__result["segFittingParamFt"];
        Expr segFittingParamGt = (Expr)__result["segFittingParamGt"];
        Tensor<float> outputRange0 = ((TensorConst)__result["outputRange_0"]).Value.Cast<float>();
        Expr hasStatic = (Expr)__result["hasStatic"];
        int outputSize = ((TensorConst)__result["outputSize"]).Value.ToScalar<int>();
        return GetReplace(
            fakeLstm,
            call,
            input,
            inputMarker,
            initialHMarker,
            initialCMarker,
            inputRange,
            wXc,
            wXcRange,
            wRcRange,
            wRc,
            initialHRange,
            initialCRange,
            wRcMarker,
            wXcMarker,
            initialH,
            initialC,
            segFittingParamFt,
            segFittingParamGt,
            outputRange0,
            hasStatic,
            outputSize,
            __result);
    }
}
