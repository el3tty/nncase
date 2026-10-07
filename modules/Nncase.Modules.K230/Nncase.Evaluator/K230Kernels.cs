#define TRACE
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Imaging;
using Nncase.IR.K230;
using Nncase.Passes.Rules.K230;
using Nncase.PatternMatch;
using OrtKISharp;

namespace Nncase.Evaluator;

/// <summary>
/// Reference (simulator) implementations of the K230 GNNE kernels: activation tables, resize, LSTM and matmul.
/// </summary>
public static class K230Kernels
{
    // ------------------------------------------------------------------------------------------------------
    // Activation parameter tables.
    //
    // Every per-channel activation is a two-piece linear function followed by a clamp:
    //     y = clamp((x < threshold ? x * leftSlope : x * rightSlope) / 2^shift + bias, min, max)
    // and is described by 7 consecutive values per channel. The "Act0" (hardware / Half) and "Act1" (fake
    // float) tables store those 7 values in a different order.
    // ------------------------------------------------------------------------------------------------------

    /// <summary>Number of table entries describing one channel of an Act0/Act1 table.</summary>
    private const int ActParamsPerChannel = 7;

    // Act0 layout: [leftSlope, rightSlope, leftBias, rightBias, clampMin, clampMax, threshold].
    private const int Act0LeftSlope = 0;
    private const int Act0RightSlope = 1;
    private const int Act0LeftBias = 2;
    private const int Act0RightBias = 3;
    private const int Act0ClampMin = 4;
    private const int Act0ClampMax = 5;
    private const int Act0Threshold = 6;

    // Act1 layout: [threshold, leftSlope, rightSlope, leftBias, rightBias, clampMin, clampMax].
    private const int Act1Threshold = 0;
    private const int Act1LeftSlope = 1;
    private const int Act1RightSlope = 2;
    private const int Act1LeftBias = 3;
    private const int Act1RightBias = 4;
    private const int Act1ClampMin = 5;
    private const int Act1ClampMax = 6;

    // Multi-segment Act1 table (16 segments): [15 thresholds | 16 slopes | 16 biases | clampMin, clampMax].
    private const int ActMultiSegmentCount = 16;
    private const int ActMultiSegmentClampMinIndex = 47;

    // LSTM sigmoid/tanh piecewise-linear fit table (Half): [15 thresholds | 16 slopes | 16 biases].
    private const int LstmFitThresholdCount = 15;
    private const int LstmFitSlopeOffset = 15;
    private const int LstmFitBiasOffset = 31;
    private const int LstmFitLastSlopeIndex = LstmFitSlopeOffset + LstmFitThresholdCount;
    private const int LstmFitLastBiasIndex = LstmFitBiasOffset + LstmFitThresholdCount;

    // LSTM gate layout inside the 4 * hidden gate vector (GNNE order is i, o, f, c).
    private const int LstmGateCount = 4;
    private const int LstmGateInput = 0;
    private const int LstmGateOutput = 1;
    private const int LstmGateForget = 2;
    private const int LstmGateCell = 3;

    // Kept from the original implementation: constant added to every bilinear interpolation result.
    private const float BilinearBias = 0f;

    // ------------------------------------------------------------------------------------------------------
    // Per-channel activations.
    // ------------------------------------------------------------------------------------------------------

    /// <summary>Applies the hardware (Half) Act0 activation of <paramref name="channel"/> to <paramref name="value"/>.</summary>
    public static float ApplyAct0(float value, ReadOnlySpan<Half> actData, int channel, sbyte shiftBits)
    {
        int offset = ActParamsPerChannel * channel;
        float activated = ApplyPiecewiseLinear(
            value,
            shiftBits,
            threshold: (float)actData[offset + Act0Threshold],
            leftSlope: (float)actData[offset + Act0LeftSlope],
            leftBias: (float)actData[offset + Act0LeftBias],
            rightSlope: (float)actData[offset + Act0RightSlope],
            rightBias: (float)actData[offset + Act0RightBias]);
        return ClampToRange(activated, (float)actData[offset + Act0ClampMin], (float)actData[offset + Act0ClampMax]);
    }

    /// <summary>Same as <see cref="ApplyAct0"/> but reading a float table.</summary>
    public static float ApplyAct01(float value, ReadOnlySpan<float> actData, int channel, sbyte shiftBits)
    {
        int offset = ActParamsPerChannel * channel;
        float activated = ApplyPiecewiseLinear(
            value,
            shiftBits,
            threshold: actData[offset + Act0Threshold],
            leftSlope: actData[offset + Act0LeftSlope],
            leftBias: actData[offset + Act0LeftBias],
            rightSlope: actData[offset + Act0RightSlope],
            rightBias: actData[offset + Act0RightBias]);
        return ClampToRange(activated, actData[offset + Act0ClampMin], actData[offset + Act0ClampMax]);
    }

    /// <summary>Float (fake-quant) version of the Act0 activation.</summary>
    public static float FakeApplyAct0(float value, float[] actData, int channel, sbyte shiftBits)
    {
        int offset = ActParamsPerChannel * channel;
        float result = value * actData[offset + Act0LeftSlope] / (float)(1L << (int)shiftBits) +
                       ((value.CompareTo(actData[offset + Act0Threshold]) < 0)
                           ? actData[offset + Act0LeftBias]
                           : actData[offset + Act0RightBias]);
        float min = actData[offset + Act0ClampMin];
        float max = actData[offset + Act0ClampMax];

        // NaN compares lower than everything, so it ends up at min.
        if (result.CompareTo(max) > 0)
        {
            return max;
        }

        return (result.CompareTo(min) <= 0) ? min : result;
    }

    /// <summary>
    /// Applies the float Act1 table (<see cref="Act1Threshold"/> layout, or the 16 segment layout) of one channel.
    /// </summary>
    private static float ApplyAct1(float value, ReadOnlySpan<float> actData, int channel, bool is16Segments)
    {
        if (is16Segments)
        {
            float segmented = ApplyMultiSegmentsAct1(ActMultiSegmentCount, value, actData);
            return ClampToRange(
                segmented,
                actData[ActMultiSegmentClampMinIndex % actData.Length],
                actData[(ActMultiSegmentClampMinIndex + 1) % actData.Length]);
        }

        int offset = ActParamsPerChannel * channel;
        float activated = ApplyPiecewiseLinear(
            value,
            0,
            threshold: actData[offset + Act1Threshold],
            leftSlope: actData[offset + Act1LeftSlope],
            leftBias: actData[offset + Act1LeftBias],
            rightSlope: actData[offset + Act1RightSlope],
            rightBias: actData[offset + Act1RightBias]);
        return ClampToRange(activated, actData[offset + Act1ClampMin], actData[offset + Act1ClampMax]);
    }

    /// <summary>Same as the float overload, reading a Half table.</summary>
    private static float ApplyAct1(float value, ReadOnlySpan<Half> actData, int channel, bool is16Segments)
    {
        if (is16Segments)
        {
            float segmented = ApplyMultiSegmentsAct1(ActMultiSegmentCount, value, actData);
            return ClampToRange(
                segmented,
                (float)actData[ActMultiSegmentClampMinIndex],
                (float)actData[ActMultiSegmentClampMinIndex + 1]);
        }

        int offset = ActParamsPerChannel * channel;
        float activated = ApplyPiecewiseLinear(
            value,
            0,
            threshold: (float)actData[offset + Act1Threshold],
            leftSlope: (float)actData[offset + Act1LeftSlope],
            leftBias: (float)actData[offset + Act1LeftBias],
            rightSlope: (float)actData[offset + Act1RightSlope],
            rightBias: (float)actData[offset + Act1RightBias]);
        return ClampToRange(activated, (float)actData[offset + Act1ClampMin], (float)actData[offset + Act1ClampMax]);
    }

    /// <summary>Evaluates an N-segment piecewise-linear function stored as [thresholds | slopes | biases] (Half).</summary>
    public static float ApplyMultiSegmentsAct1(int segmentCount, float value, ReadOnlySpan<Half> actData)
    {
        int slopeBase = segmentCount - 1;
        int biasBase = segmentCount - 1 + segmentCount;
        for (int i = 0; i < segmentCount - 1; i++)
        {
            if (value < (float)actData[i])
            {
                return value * (float)actData[slopeBase + i] + (float)actData[biasBase + i];
            }
        }

        return value * (float)actData[slopeBase + segmentCount - 1] + (float)actData[biasBase + segmentCount - 1];
    }

    /// <summary>Same as the Half overload, reading a float table (in-loop reads wrap around the table length).</summary>
    public static float ApplyMultiSegmentsAct1(int segmentCount, float value, ReadOnlySpan<float> actData)
    {
        int slopeBase = segmentCount - 1;
        int biasBase = segmentCount - 1 + segmentCount;
        for (int i = 0; i < segmentCount - 1; i++)
        {
            if (value < actData[i])
            {
                return value * actData[(slopeBase + i) % actData.Length] + actData[(biasBase + i) % actData.Length];
            }
        }

        return value * actData[slopeBase + segmentCount - 1] + actData[biasBase + segmentCount - 1];
    }

    /// <summary>Clamps <paramref name="value"/> to [clamp[0], clamp[1]] using <see cref="System.Math.Clamp(float, float, float)"/>.</summary>
    public static float ApplyActivation(float value, float[] clamp)
    {
        return System.Math.Clamp(value, clamp[0], clamp[1]);
    }

    /// <summary>
    /// Two-piece linear function: (value * slope / 2^shiftBits) + bias, slope/bias picked by <paramref name="threshold"/>.
    /// </summary>
    private static float ApplyPiecewiseLinear(
        float value,
        sbyte shiftBits,
        float threshold,
        float leftSlope,
        float leftBias,
        float rightSlope,
        float rightBias)
    {
        // CompareTo (not '<') is intentional: NaN sorts below every number.
        if (value.CompareTo(threshold) < 0)
        {
            return value * leftSlope / (float)(1L << (int)shiftBits) + leftBias;
        }

        return value * rightSlope / (float)(1L << (int)shiftBits) + rightBias;
    }

    /// <summary>
    /// Clamps to [min, max] with the hardware tie-breaking: NaN and anything &lt;= min yield min
    /// (also when max &lt;= min).
    /// </summary>
    private static float ClampToRange(float value, float min, float max)
    {
        float capped = (value.CompareTo(max) > 0) ? max : value;
        return (capped.CompareTo(min) <= 0) ? min : capped;
    }

    // ------------------------------------------------------------------------------------------------------
    // Shape helpers.
    // ------------------------------------------------------------------------------------------------------

    /// <summary>Output spatial size of a windowed op (conv / pool).</summary>
    public static int GetWindowedOutputSize(
        int size,
        int filter,
        int stride,
        int dilation,
        bool same,
        bool ceilMode = false)
    {
        int effectiveFilter = (filter - 1) * dilation + 1;
        if (same)
        {
            return (size + stride - 1) / stride;
        }

        if (!ceilMode)
        {
            return (size - effectiveFilter + stride) / stride;
        }

        // NOTE: despite its name, ceil mode truncates the float quotient (kept from the original).
        return (int)((float)(size - effectiveFilter + stride) / (float)stride);
    }

    public static int ComputeSize(Const input)
    {
        return ComputeSize(input.CheckedShape);
    }

    public static int ComputeSize(int[] shape)
    {
        return shape.Aggregate(1, (int product, int dim) => product * dim);
    }

    public static int ComputeSize(Shape shape)
    {
        return shape.Prod().FixedValue;
    }

    /// <summary>Row-major linear index of <paramref name="index"/> in <paramref name="shape"/>.</summary>
    public static int LinearIndex(int[] shape, int[] index)
    {
        int linear = index[0];
        for (int axis = 1; axis < shape.Length; axis++)
        {
            linear = linear * shape[axis] + index[axis];
        }

        return linear;
    }

    /// <summary>Row-major strides where broadcast (size 1) axes get stride 0.</summary>
    public static int[] GetDefaultStrides(int[] shape)
    {
        int[] strides = new int[shape.Length];
        int running = 1;
        for (int axis = shape.Length - 1; axis >= 0; axis--)
        {
            strides[axis] = running;
            running = strides[axis] * shape[axis];
            if (shape[axis] == 1)
            {
                strides[axis] = 0;
            }
        }

        return strides;
    }

    /// <summary>Concatenates <paramref name="chunks"/> into one constant of shape <paramref name="outShape"/>.</summary>
    public static Const ConcatOutput<T>(List<T[]> chunks, int[] outShape)
        where T : unmanaged, IEquatable<T>
    {
        return Const.FromTensor(Tensor.From(chunks.SelectMany(chunk => chunk).ToArray(), outShape));
    }

    /// <summary>All-zero bias of <paramref name="outChannels"/> elements as an ORT tensor.</summary>
    public static OrtKISharp.Tensor ZeroBias(int outChannels)
    {
        return new float[outChannels];
    }

    public static OrtKISharp.Tensor DefaultBias(int oc)
    {
        return Tensor.FromArray(new float[oc]).ToOrtTensor();
    }

    // ------------------------------------------------------------------------------------------------------
    // GNNE activation node (a, b -> act(a op b)).
    // ------------------------------------------------------------------------------------------------------

    /// <summary>Float simulation of the GNNE activation node (no quantization).</summary>
    public static Const FakeGnneActivation(
        bool is16Segments,
        Tensor inputA,
        Tensor inputB,
        bool hasUninitializedInput,
        Shape outputShape,
        ReadOnlySpan<float> actData,
        int outChannels,
        GnneActivationType actType)
    {
        GNNEShape inputAShape = new GNNEShape(inputA.Shape.ToValueArray());
        GNNEShape inputBShape = new GNNEShape(inputB.Shape.ToValueArray());
        float[] inputAData = inputA.ToArray<float>();
        float[] inputBData = inputB.ToArray<float>();
        int outputLength = ComputeSize(outputShape);

        int[] outputDims = new int[outputShape.Count];
        for (int axis = 0; axis < outputShape.Count; axis++)
        {
            outputDims[axis] = outputShape[axis].FixedValue;
        }

        GNNEShape outputGnneShape = new GNNEShape(outputDims);

        // The channel count is always taken from the output shape; the argument is ignored.
        int outputChannels = outputGnneShape[1];

        // Zero point 0 / scale 1 makes the dequantization an exact identity.
        DeQuantizeParam identity = new DeQuantizeParam(0, 1f);
        float[] activated = hasUninitializedInput
            ? ActivateUnary(
                inputAData,
                ToDims(inputAShape),
                outputGnneShape[2],
                outputGnneShape[3],
                outputChannels,
                outputLength,
                actData,
                is16Segments,
                identity)
            : ActivateBinary(
                inputAData,
                ToDims(inputAShape),
                inputBData,
                ToDims(inputBShape),
                outputGnneShape[0],
                outputChannels,
                outputGnneShape[2],
                outputGnneShape[3],
                outputLength,
                actData,
                is16Segments,
                identity,
                identity,
                actType);
        return Const.FromTensor(Tensor.From(activated, outputShape));
    }

    /// <summary>
    /// Integer simulation of the GNNE activation node. Returns null for binary ops whose output type is not one of
    /// int8 / uint8 / float16 / int16.
    /// </summary>
    public static Const GnneActivation(
        bool is16Segments,
        Tensor inputA,
        Tensor inputB,
        bool hasUninitializedInput,
        ReadOnlySpan<float> actData,
        int outChannels,
        DeQuantizeParam deQuantizeParamA,
        DeQuantizeParam deQuantizeParamB,
        GnneActivationType actType,
        TensorType outputType,
        Shape outputShape)
    {
        int[] inputADims = inputA.Shape.ToValueArray();
        float[] inputAData = inputA.ToArray<float>();
        int outputLength = ComputeSize(outputShape);

        if (hasUninitializedInput)
        {
            float[] unaryResult = ActivateUnary(
                inputAData,
                inputADims,
                outputShape[2].FixedValue,
                outputShape[3].FixedValue,
                outChannels,
                outputLength,
                actData,
                is16Segments,
                deQuantizeParamA);
            return ToOutputConst(outputType.DType, unaryResult, inputA.Shape, unknownTypeIsInt16: true);
        }

        DataType outputDType = outputType.DType;
        if (outputDType != DataTypes.Int8 && outputDType != DataTypes.UInt8 && outputDType != DataTypes.Float16 &&
            outputDType != DataTypes.Int16)
        {
            return null;
        }

        // NOTE: the original had a "same shape" fast path guarded by an int[] reference comparison,
        // which could never be true; the broadcast loop below handles every case.
        float[] binaryResult = ActivateBinary(
            inputAData,
            inputADims,
            inputB.ToArray<float>(),
            inputB.Shape.ToValueArray(),
            outputShape[0].FixedValue,
            outChannels,
            outputShape[2].FixedValue,
            outputShape[3].FixedValue,
            outputLength,
            actData,
            is16Segments,
            deQuantizeParamA,
            deQuantizeParamB,
            actType);
        return ToOutputConst(outputDType, binaryResult, inputA.Shape, unknownTypeIsInt16: false);
    }

    /// <summary>
    /// Dequantizes a single NCHW input, nearest-neighbour resamples it to the output H/W and applies the activation.
    /// </summary>
    private static float[] ActivateUnary(
        float[] input,
        int[] inputDims,
        int outputHeight,
        int outputWidth,
        int outChannels,
        int outputLength,
        ReadOnlySpan<float> actData,
        bool is16Segments,
        DeQuantizeParam inputQuant)
    {
        float[] output = new float[outputLength];
        int inputHeight = inputDims[2];
        int inputWidth = inputDims[3];
        int inputPlane = inputHeight * inputWidth;
        int outputPlane = outputHeight * outputWidth;
        int inputBatchStride = inputDims[1] * inputPlane;

        for (int batch = 0; batch < inputDims[0]; batch++)
        {
            int inputBatchOffset = batch * inputBatchStride;
            int outputBatchOffset = batch * outChannels * outputPlane;
            for (int channel = 0; channel < outChannels; channel++)
            {
                int inputChannelOffset = inputBatchOffset + channel * inputPlane;
                int outputChannelOffset = outputBatchOffset + channel * outputPlane;
                for (int y = 0; y < outputHeight; y++)
                {
                    for (int x = 0; x < outputWidth; x++)
                    {
                        int sourceIndex = inputChannelOffset + y * inputHeight / outputHeight * inputWidth +
                                          x * inputWidth / outputWidth;
                        float value = Dequantize(input[sourceIndex], inputQuant);
                        output[outputChannelOffset + y * outputWidth + x] =
                            ApplyAct1(value, actData, channel, is16Segments);
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Dequantizes two NCHW inputs (size-1 axes are broadcast), adds or multiplies them and applies the activation.
    /// </summary>
    private static float[] ActivateBinary(
        float[] inputA,
        int[] inputADims,
        float[] inputB,
        int[] inputBDims,
        int outputBatch,
        int outChannels,
        int outputHeight,
        int outputWidth,
        int outputLength,
        ReadOnlySpan<float> actData,
        bool is16Segments,
        DeQuantizeParam quantA,
        DeQuantizeParam quantB,
        GnneActivationType actType)
    {
        float[] output = new float[outputLength];
        int outputPlane = outputHeight * outputWidth;

        for (int batch = 0; batch < outputBatch; batch++)
        {
            int outputBatchOffset = batch * outChannels * outputPlane;
            for (int channel = 0; channel < outChannels; channel++)
            {
                int outputIndex = outputBatchOffset + channel * outputPlane;
                for (int y = 0; y < outputHeight; y++)
                {
                    for (int x = 0; x < outputWidth; x++)
                    {
                        float a = inputA[BroadcastOffset(inputADims, batch, channel, y, x)];
                        float b = inputB[BroadcastOffset(inputBDims, batch, channel, y, x)];

                        // Float association is deliberately the same as in the original expressions.
                        float value = (actType != GnneActivationType.Mul)
                            ? Dequantize(a, quantA) + Dequantize(b, quantB)
                            : Dequantize(a, quantA) * (b - (float)quantB.ZeroPoint) * quantB.Scale;
                        output[outputIndex] = ApplyAct1(value, actData, channel, is16Segments);
                        outputIndex++;
                    }
                }
            }
        }

        return output;
    }

    private static float Dequantize(float quantized, DeQuantizeParam param)
    {
        return (quantized - (float)param.ZeroPoint) * param.Scale;
    }

    /// <summary>Linear offset of NCHW position (n, c, y, x) in a tensor whose size-1 axes are broadcast.</summary>
    private static int BroadcastOffset(int[] dims, int n, int c, int y, int x)
    {
        int batchOffset = (dims[0] != 1) ? (n * dims[1] * dims[2] * dims[3]) : 0;
        int channelOffset = (dims[1] != 1) ? (c * dims[2] * dims[3]) : 0;
        int rowOffset = (dims[2] != 1) ? (y * dims[3]) : 0;
        int columnOffset = (dims[3] != 1) ? x : 0;
        return batchOffset + channelOffset + rowOffset + columnOffset;
    }

    private static int[] ToDims(GNNEShape shape)
    {
        return new int[4] { shape[0], shape[1], shape[2], shape[3] };
    }

    /// <summary>
    /// Truncating conversion of the activation result to the output type. Unknown types become int16 when
    /// <paramref name="unknownTypeIsInt16"/> is set, otherwise null.
    /// </summary>
    private static Const ToOutputConst(DataType outputType, float[] values, Shape shape, bool unknownTypeIsInt16)
    {
        if (outputType == DataTypes.Int8)
        {
            return CreateConst(values, value => (sbyte)value, shape);
        }

        if (outputType == DataTypes.Float16)
        {
            return CreateConst(values, value => (Half)value, shape);
        }

        if (outputType == DataTypes.UInt8)
        {
            return CreateConst(values, value => (byte)value, shape);
        }

        if (outputType == DataTypes.Int16 || unknownTypeIsInt16)
        {
            return CreateConst(values, value => (short)value, shape);
        }

        return null;
    }

    private static Const CreateConst<T>(float[] values, Func<float, T> convert, Shape shape)
        where T : unmanaged, IEquatable<T>
    {
        return Const.FromTensor(Tensor.From(ConvertAll(values, convert), shape));
    }

    private static T[] ConvertAll<T>(float[] values, Func<float, T> convert)
    {
        T[] converted = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            converted[i] = convert(values[i]);
        }

        return converted;
    }

    // ------------------------------------------------------------------------------------------------------
    // LSTM.
    // ------------------------------------------------------------------------------------------------------

    /// <summary>Tanh evaluated with the piecewise-linear fit table.</summary>
    public static float Tanh(float x, Tensor<Half> segFittingParamGt)
    {
        return EvaluateLstmFit(x, segFittingParamGt.ToArray());
    }

    /// <summary>Sigmoid evaluated with the piecewise-linear fit table.</summary>
    public static float Sigmoid(float x, Tensor<Half> segFittingParamFt)
    {
        return EvaluateLstmFit(x, segFittingParamFt.ToArray());
    }

    /// <summary>
    /// Float simulation of the GNNE LSTM. Returns null when <paramref name="outputSize"/> is not 1, 2 or 3.
    /// </summary>
    public static Tensor<float>[] FakeGnneLstm(
        Tensor<float> input,
        Tensor<float> inputWeights,
        Tensor<Half> inputActivation,
        Tensor<float> recurrentWeights,
        Tensor<Half> recurrentActivation,
        Tensor<float> initialHidden,
        Tensor<float> initialCell,
        Tensor<Half> sigmoidFit,
        Tensor<Half> tanhFit,
        Tensor<float> output,
        Tensor<float> outputH,
        Tensor<float> outputC,
        LSTMDirection direction,
        int outputSize)
    {
        LstmDims dims = new LstmDims(input.Shape, output.Shape);
        int hidden = dims.Hidden;

        float[] inputData = input.ToArray();
        float[] inputWeightData = inputWeights.ToArray();
        float[] recurrentWeightData = recurrentWeights.ToArray();
        Half[] inputActData = inputActivation.ToArray<Half>();
        Half[] recurrentActData = recurrentActivation.ToArray<Half>();
        Half[] sigmoidFitData = sigmoidFit.ToArray();
        Half[] tanhFitData = tanhFit.ToArray();

        float[] outputData = output.ToArray();
        float[] outputHData = outputH.ToArray();
        float[] outputCData = outputC.ToArray();

        // Running hidden / cell state, one [batch, hidden] block per direction.
        float[] hiddenState = CopyOf(initialHidden.ToArray(), ComputeSize(initialHidden.Shape));
        float[] cellState = CopyOf(initialCell.ToArray(), ComputeSize(initialCell.Shape));

        List<int> stepOrder = CreateStepOrder(dims.TimeSteps, direction);
        for (int dir = 0; dir < dims.Directions; dir++)
        {
            if (dir == 1)
            {
                stepOrder.Reverse();
            }

            for (int batch = 0; batch < dims.Batch; batch++)
            {
                for (int stepPos = 0; stepPos < stepOrder.Count; stepPos++)
                {
                    int step = stepOrder[stepPos];
                    float[] inputGates = new float[hidden * LstmGateCount];
                    float[] recurrentGates = new float[hidden * LstmGateCount];

                    float[] stepInput = new float[dims.InputSize];
                    Array.Copy(inputData, dims.InputOffset(batch, step), stepInput, 0, dims.InputSize);

                    // NOTE: sized by the input size but only 'hidden' elements are copied (as in the original).
                    float[] stepHidden = new float[dims.InputSize];
                    Array.Copy(hiddenState, dims.StateReadOffset(dir, batch), stepHidden, 0, hidden);

                    for (int row = 0; row < inputGates.Length; row++)
                    {
                        float[] inputRow = new float[dims.InputSize];
                        Array.Copy(inputWeightData, WeightOffset(inputWeights.Shape, row, dir), inputRow, 0, dims.InputSize);
                        inputGates[row] = Dot(stepInput, inputRow);
                        inputGates[row] = ApplyAct0(inputGates[row], inputActData, ActivationIndex(inputWeights.Shape, row, dir), 0);

                        float[] recurrentRow = new float[dims.InputSize];
                        Array.Copy(recurrentWeightData, WeightOffset(recurrentWeights.Shape, row, dir), recurrentRow, 0, hidden);
                        recurrentGates[row] = Dot(stepHidden, recurrentRow);
                        recurrentGates[row] = ApplyAct0(
                            recurrentGates[row],
                            recurrentActData,
                            ActivationIndex(recurrentWeights.Shape, row, dir),
                            0);
                        inputGates[row] += recurrentGates[row];
                    }

                    float[] gates = inputGates;
                    int forget = hidden * LstmGateForget;
                    int inputGate = hidden * LstmGateInput;
                    int outputGate = hidden * LstmGateOutput;
                    int cell = hidden * LstmGateCell;
                    int stateWrite = dims.StateWriteOffset(dir);
                    int stateRead = dims.StateReadOffset(dir, batch);

                    // f = sigmoid(f); f *= c(t-1)
                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + forget] = EvaluateLstmFit(gates[k + forget], sigmoidFitData);
                    }

                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + forget] *= cellState[k + stateRead];
                    }

                    // i = sigmoid(i); g = tanh(g); i *= g
                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + inputGate] = EvaluateLstmFit(gates[k + inputGate], sigmoidFitData);
                    }

                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + cell] = EvaluateLstmFit(gates[k + cell], tanhFitData);
                    }

                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + inputGate] *= gates[k + cell];
                    }

                    // c(t) = f * c(t-1) + i * g
                    for (int k = 0; k < hidden; k++)
                    {
                        cellState[k + stateWrite] = gates[k + forget] + gates[k + inputGate];
                    }

                    // o = sigmoid(o); h(t) = o * tanh(c(t))
                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + outputGate] = EvaluateLstmFit(gates[k + outputGate], sigmoidFitData);
                    }

                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + cell] = EvaluateLstmFit(cellState[k + stateWrite], tanhFitData);
                    }

                    for (int k = 0; k < hidden; k++)
                    {
                        hiddenState[k + stateWrite] = gates[k + cell] * gates[k + outputGate];
                    }

                    for (int k = 0; k < hidden; k++)
                    {
                        outputData[dims.OutputOffset(dir, batch, step) + k] = hiddenState[stateWrite + k];
                    }

                    if (stepPos == stepOrder.Count - 1)
                    {
                        for (int k = 0; k < hidden; k++)
                        {
                            outputHData[stateRead + k] = hiddenState[stateWrite + k];
                        }

                        for (int k = 0; k < hidden; k++)
                        {
                            outputCData[stateRead + k] = cellState[stateWrite + k];
                        }
                    }
                }
            }
        }

        Tensor<float> outputTensor = Tensor.From(outputData, output.Shape);
        Tensor<float> outputCTensor = Tensor.From(outputCData, outputC.Shape);
        Tensor<float> outputHTensor = Tensor.From(outputHData, outputH.Shape);
        return outputSize switch
        {
            1 => new Tensor<float>[1] { outputTensor },
            2 => new Tensor<float>[2] { outputTensor, outputHTensor },
            3 => new Tensor<float>[3] { outputTensor, outputHTensor, outputCTensor },
            _ => null,
        };
    }

    /// <summary>
    /// Quantized simulation of the GNNE LSTM. Returns null when <paramref name="outputSize"/> is not 1, 2 or 3.
    /// </summary>
    /// <remarks>
    /// The zero-point arguments (<paramref name="inputDeqBias"/>, <paramref name="hiddenDeqBias0"/>,
    /// <paramref name="hiddenDeqBias1"/>, <paramref name="inputWeightsQuantArgs"/>,
    /// <paramref name="recurrentWeightsQuantArgs"/>) are accepted but have no effect: the original implementation
    /// built lazy LINQ queries to subtract them and never enumerated those queries. That behaviour is preserved.
    /// </remarks>
    public static Tensor[] GnneLstmImpl(
        DataType outputDataType,
        DataType outputHDataType,
        Tensor<float> input,
        Tensor<float> inputWeights,
        Tensor<Half> inputActivation,
        Tensor<float> recurrentWeights,
        Tensor<Half> recurrentActivationFirstStep,
        Tensor<Half> recurrentActivationOtherSteps,
        Tensor<float> initialHidden,
        Tensor<float> initialCell,
        Tensor<Half> sigmoidFit,
        Tensor<Half> tanhFit,
        Tensor output,
        Tensor outputH,
        Tensor outputC,
        LSTMDirection direction,
        Tensor<Half> inputWeightsQuantArgs,
        Tensor<Half> recurrentWeightsQuantArgs,
        Tensor<Half> cellBinAct,
        Tensor<Half> hiddenBinQuantAct,
        int inputDeqBias,
        int hiddenDeqBias0,
        int hiddenDeqBias1,
        int inputShiftBits,
        int recurrentShiftBitsFirstStep,
        int recurrentShiftBitsOtherSteps,
        int outputSize)
    {
        LstmDims dims = new LstmDims(input.Shape, output.Shape);
        int hidden = dims.Hidden;

        float[] inputData = input.ToArray();
        float[] inputWeightData = inputWeights.ToArray();
        float[] recurrentWeightData = recurrentWeights.ToArray();
        Half[] inputActData = inputActivation.ToArray<Half>();
        Half[] recurrentActFirstData = recurrentActivationFirstStep.ToArray<Half>();
        Half[] recurrentActOtherData = recurrentActivationOtherSteps.ToArray<Half>();
        Half[] sigmoidFitData = sigmoidFit.ToArray();
        Half[] tanhFitData = tanhFit.ToArray();
        Half[] cellBinActData = cellBinAct.ToArray<Half>();
        Half[] hiddenBinQuantActData = hiddenBinQuantAct.ToArray<Half>();

        Half[] outputCData = outputC.ToArray<Half>();
        LstmTypedBuffers outputBuffers = new LstmTypedBuffers(ComputeSize(output.Shape));
        LstmTypedBuffers outputHBuffers = new LstmTypedBuffers(ComputeSize(outputH.Shape));

        float[] hiddenState = CopyOf(initialHidden.ToArray(), ComputeSize(initialHidden.Shape));
        float[] cellState = CopyOf(initialCell.ToArray(), ComputeSize(initialCell.Shape));

        List<int> stepOrder = CreateStepOrder(dims.TimeSteps, direction);
        for (int dir = 0; dir < dims.Directions; dir++)
        {
            if (dir == 1)
            {
                stepOrder.Reverse();
            }

            for (int batch = 0; batch < dims.Batch; batch++)
            {
                for (int stepPos = 0; stepPos < stepOrder.Count; stepPos++)
                {
                    int step = stepOrder[stepPos];
                    bool isFirstStep = step == stepOrder[0];
                    float[] gates = new float[hidden * LstmGateCount];
                    float[] recurrentGates = new float[hidden * LstmGateCount];

                    float[] stepInput = new float[dims.InputSize];
                    Array.Copy(inputData, dims.InputOffset(batch, step), stepInput, 0, dims.InputSize);

                    // NOTE: sized by the input size but only 'hidden' elements are copied (as in the original).
                    float[] stepHidden = new float[dims.InputSize];
                    Array.Copy(hiddenState, dims.StateReadOffset(dir, batch), stepHidden, 0, hidden);

                    int recurrentShift = isFirstStep ? recurrentShiftBitsFirstStep : recurrentShiftBitsOtherSteps;
                    Half[] recurrentActData = isFirstStep ? recurrentActFirstData : recurrentActOtherData;
                    for (int row = 0; row < gates.Length; row++)
                    {
                        float[] inputRow = new float[dims.InputSize];
                        Array.Copy(inputWeightData, WeightOffset(inputWeights.Shape, row, dir), inputRow, 0, dims.InputSize);
                        gates[row] = Dot(stepInput, inputRow);
                        gates[row] = ApplyAct0(
                            gates[row] / (float)(1 << inputShiftBits),
                            inputActData,
                            ActivationIndex(inputWeights.Shape, row, dir),
                            0);

                        float[] recurrentRow = new float[dims.InputSize];
                        Array.Copy(recurrentWeightData, WeightOffset(recurrentWeights.Shape, row, dir), recurrentRow, 0, hidden);
                        recurrentGates[row] = Dot(stepHidden, recurrentRow);
                        recurrentGates[row] = ApplyAct0(
                            recurrentGates[row] / (float)(1 << recurrentShift),
                            recurrentActData,
                            ActivationIndex(recurrentWeights.Shape, row, dir),
                            0);
                        gates[row] += recurrentGates[row];
                    }

                    int forget = hidden * LstmGateForget;
                    int inputGate = hidden * LstmGateInput;
                    int outputGate = hidden * LstmGateOutput;
                    int cell = hidden * LstmGateCell;
                    int stateWrite = dims.StateWriteOffset(dir);
                    int stateRead = dims.StateReadOffset(dir, batch);

                    // f = sigmoid(f)
                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + forget] = EvaluateLstmFit(gates[k + forget], sigmoidFitData);
                    }

                    // f *= c(t-1), re-quantized by the cell activation table
                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + forget] *= cellState[k + stateRead];
                        gates[k + forget] = ApplyAct1(gates[k + forget], cellBinActData, k, is16Segments: false);
                    }

                    // i = sigmoid(i)
                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + inputGate] = EvaluateLstmFit(gates[k + inputGate], sigmoidFitData);
                    }

                    // g = tanh(g)
                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + cell] = EvaluateLstmFit(gates[k + cell], tanhFitData);
                    }

                    // i *= g, re-quantized by the cell activation table
                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + inputGate] *= gates[k + cell];
                        gates[k + inputGate] = ApplyAct1(gates[k + inputGate], cellBinActData, k, is16Segments: false);
                    }

                    // c(t) = f * c(t-1) + i * g
                    for (int k = 0; k < hidden; k++)
                    {
                        cellState[k + stateWrite] = gates[k + forget] + gates[k + inputGate];
                    }

                    // o = sigmoid(o)
                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + outputGate] = EvaluateLstmFit(gates[k + outputGate], sigmoidFitData);
                    }

                    // tanh(c(t))
                    for (int k = 0; k < hidden; k++)
                    {
                        gates[k + cell] = EvaluateLstmFit(cellState[k + stateWrite], tanhFitData);
                    }

                    // h(t) = o * tanh(c(t)), quantized into the output type
                    int outputBase = dims.OutputOffset(dir, batch, step);
                    for (int k = 0; k < hidden; k++)
                    {
                        hiddenState[k + stateWrite] = gates[k + cell] * gates[k + outputGate];
                        hiddenState[k + stateWrite] = ApplyAct1(
                            hiddenState[k + stateWrite],
                            hiddenBinQuantActData,
                            k,
                            is16Segments: false);
                        int quantized = (int)System.Math.Round(hiddenState[k + stateWrite]);

                        // NOTE: the float16 output stores element 0 of the direction's state for every k
                        // (kept from the original).
                        outputBuffers.Store(outputDataType, outputBase + k, quantized, hiddenState[stateWrite]);
                    }

                    if (stepPos != stepOrder.Count - 1)
                    {
                        continue;
                    }

                    for (int k = 0; k < hidden; k++)
                    {
                        outputHBuffers.CopyFrom(outputHDataType, stateRead + k, outputBuffers, outputBase + k);
                    }

                    for (int k = 0; k < hidden; k++)
                    {
                        outputCData[stateRead + k] = (Half)cellState[k + stateWrite];
                    }
                }
            }
        }

        Tensor outputTensor = outputBuffers.ToTensor(outputDataType, output.Shape);
        Tensor outputHTensor = outputHBuffers.ToTensor(outputHDataType, outputH.Shape);
        Tensor outputCTensor = Tensor.From(outputCData, outputC.Shape);
        return outputSize switch
        {
            1 => new Tensor[1] { outputTensor },
            2 => new Tensor[2] { outputTensor, outputHTensor },
            3 => new Tensor[3] { outputTensor, outputHTensor, outputCTensor },
            _ => null,
        };
    }

    /// <summary>
    /// Piecewise-linear sigmoid / tanh approximation: 15 Half thresholds, 16 slopes, 16 biases.
    /// </summary>
    private static float EvaluateLstmFit(float x, Half[] fit)
    {
        for (int i = 0; i < LstmFitThresholdCount; i++)
        {
            if ((Half)x < fit[i])
            {
                return (float)fit[LstmFitSlopeOffset + i] * x + (float)fit[LstmFitBiasOffset + i];
            }
        }

        return (float)fit[LstmFitLastSlopeIndex] * x + (float)fit[LstmFitLastBiasIndex];
    }

    /// <summary>Time steps in processing order (reversed for reverse LSTMs).</summary>
    private static List<int> CreateStepOrder(int timeSteps, LSTMDirection direction)
    {
        List<int> stepOrder = new List<int>();
        for (int step = 0; step < timeSteps; step++)
        {
            stepOrder.Add(step);
        }

        if (direction == LSTMDirection.Reverse)
        {
            stepOrder.Reverse();
        }

        return stepOrder;
    }

    private static float[] CopyOf(float[] source, int length)
    {
        float[] copy = new float[length];
        for (int i = 0; i < copy.Length; i++)
        {
            copy[i] = source[i];
        }

        return copy;
    }

    private static float Dot(float[] a, float[] b)
    {
        return OrtKI.MatMul(a, b).ToArray<float>()[0];
    }

    /// <summary>Offset of gate row <paramref name="row"/> of direction <paramref name="dir"/> in a [_, dirs, 4H, N] weight tensor.</summary>
    private static int WeightOffset(Shape weightShape, int row, int dir)
    {
        return row * weightShape[3].FixedValue + dir * weightShape[2].FixedValue * weightShape[3].FixedValue;
    }

    /// <summary>Index of gate row <paramref name="row"/> of direction <paramref name="dir"/> in its activation table.</summary>
    private static int ActivationIndex(Shape weightShape, int row, int dir)
    {
        return dir * weightShape[2].FixedValue + row;
    }

    /// <summary>
    /// Dimensions and flat offsets of the GNNE LSTM tensors:
    /// input [_, timeSteps, batch, inputSize], output [timeSteps, directions, batch, hidden].
    /// </summary>
    private readonly struct LstmDims
    {
        public LstmDims(Shape inputShape, Shape outputShape)
        {
            TimeSteps = inputShape[1].FixedValue;
            Batch = inputShape[2].FixedValue;
            InputSize = inputShape[3].FixedValue;
            Directions = outputShape[1].FixedValue;
            OutputBatch = outputShape[2].FixedValue;
            Hidden = outputShape[3].FixedValue;
        }

        public int TimeSteps { get; }

        public int Batch { get; }

        public int InputSize { get; }

        public int Directions { get; }

        public int OutputBatch { get; }

        public int Hidden { get; }

        public int InputOffset(int batch, int step)
        {
            return batch * InputSize + step * Batch * InputSize;
        }

        /// <summary>Offset of this direction's state block when reading [batch, hidden].</summary>
        public int StateReadOffset(int dir, int batch)
        {
            return batch * Hidden + dir * OutputBatch * Hidden;
        }

        /// <summary>Offset of this direction's state block when writing (the batch index is not included).</summary>
        public int StateWriteOffset(int dir)
        {
            return dir * OutputBatch * Hidden;
        }

        public int OutputOffset(int dir, int batch, int step)
        {
            return batch * Hidden + dir * OutputBatch * Hidden + step * Directions * OutputBatch * Hidden;
        }
    }

    /// <summary>One buffer per supported LSTM output element type; only the one chosen by the data type is used.</summary>
    private sealed class LstmTypedBuffers
    {
        private readonly Half[] _float16;
        private readonly sbyte[] _int8;
        private readonly byte[] _uint8;
        private readonly short[] _int16;

        public LstmTypedBuffers(int length)
        {
            _float16 = new Half[length];
            _int8 = new sbyte[length];
            _uint8 = new byte[length];
            _int16 = new short[length];
        }

        /// <summary>Stores a rounded value saturated to the type range; float16 stores <paramref name="halfSource"/>.</summary>
        public void Store(DataType type, int index, int rounded, float halfSource)
        {
            if (type == DataTypes.UInt8)
            {
                _uint8[index] = (byte)System.Math.Min(System.Math.Max(0, rounded), 255);
            }
            else if (type == DataTypes.Int8)
            {
                _int8[index] = (sbyte)System.Math.Min(System.Math.Max(-127, rounded), 127);
            }
            else if (type == DataTypes.Int16)
            {
                _int16[index] = (short)System.Math.Min(System.Math.Max(-2047, rounded), 2047);
            }
            else
            {
                _float16[index] = (Half)halfSource;
            }
        }

        /// <summary>Copies one element of the buffer selected by <paramref name="type"/> from <paramref name="source"/>.</summary>
        public void CopyFrom(DataType type, int index, LstmTypedBuffers source, int sourceIndex)
        {
            if (type == DataTypes.UInt8)
            {
                _uint8[index] = source._uint8[sourceIndex];
            }
            else if (type == DataTypes.Int8)
            {
                _int8[index] = source._int8[sourceIndex];
            }
            else if (type == DataTypes.Int16)
            {
                _int16[index] = source._int16[sourceIndex];
            }
            else
            {
                _float16[index] = source._float16[sourceIndex];
            }
        }

        /// <summary>Float16, int8, int16 or (for any other type) uint8 tensor.</summary>
        public Tensor ToTensor(DataType type, Shape shape)
        {
            if (type == DataTypes.Float16)
            {
                return Tensor.From(_float16, shape);
            }

            if (type == DataTypes.Int8)
            {
                return Tensor.From(_int8, shape);
            }

            if (type == DataTypes.Int16)
            {
                return Tensor.From(_int16, shape);
            }

            return Tensor.From(_uint8, shape);
        }
    }

    // ------------------------------------------------------------------------------------------------------
    // Image resize.
    // ------------------------------------------------------------------------------------------------------

    /// <summary>Float bilinear resize of an NCHW tensor to <paramref name="newSize"/> = [height, width].</summary>
    public static Tensor FakeAi2dResizeBilinear(Tensor input, int[] newSize, bool alignCorners, bool halfPixelCenters)
    {
        float[] resized = ResizeBilinear(input, newSize, alignCorners, halfPixelCenters);
        return new Tensor<float>(resized, ResizedDims(input, newSize));
    }

    /// <summary>Float nearest-neighbour resize of an NCHW tensor to <paramref name="newSize"/> = [height, width].</summary>
    public static Tensor FakeAi2dResizeNearestNeighbor(
        Tensor input,
        int[] newSize,
        bool alignCorners,
        bool halfPixelCenters)
    {
        float[] resized = ResizeNearestNeighbor(input, newSize, alignCorners, halfPixelCenters);
        return new Tensor<float>(resized, ResizedDims(input, newSize));
    }

    /// <summary>Bilinear resize whose result is truncated to <paramref name="dataTypes"/> (float otherwise).</summary>
    public static Tensor Ai2dResizeBilinear(
        PrimType dataTypes,
        Tensor input,
        int[] newSize,
        bool alignCorners,
        bool halfPixelCenters,
        int inDeqBias,
        QuantParam quantParam)
    {
        float[] resized = ResizeBilinear(input, newSize, alignCorners, halfPixelCenters);
        return ConvertResized(dataTypes, resized, ResizedDims(input, newSize));
    }

    /// <summary>Nearest-neighbour resize whose result is truncated to <paramref name="dataTypes"/> (float otherwise).</summary>
    public static Tensor Ai2dResizeNearestNeighbor(
        PrimType dataTypes,
        Tensor input,
        int[] newSize,
        bool alignCorners,
        bool halfPixelCenters,
        int inDeqBias,
        QuantParam quantParam)
    {
        float[] resized = ResizeNearestNeighbor(input, newSize, alignCorners, halfPixelCenters);
        return ConvertResized(dataTypes, resized, ResizedDims(input, newSize));
    }

    /// <summary>Per-axis input/output ratios [height, width].</summary>
    public static float[] GetResizeScales(int[] inShape, int outH, int outW, bool alignCorners)
    {
        float heightScale = (float)inShape[2] / (float)outH;
        float widthScale = (float)inShape[3] / (float)outW;
        if (alignCorners && outH > 1)
        {
            heightScale = (float)(inShape[2] - 1) / (float)(outH - 1);
        }

        if (alignCorners && outW > 1)
        {
            widthScale = (float)(inShape[3] - 1) / (float)(outW - 1);
        }

        return new float[2] { heightScale, widthScale };
    }

    /// <summary>
    /// Source coordinate of destination index <paramref name="value"/> for bilinear resize, as
    /// { scaledValue, lowerIndex, upperIndex }. The last three parameters are ignored (legacy signature).
    /// </summary>
    public static double[] SetResizeBilinear(
        int value,
        double scale,
        bool halfPixelCenters,
        int shapeSize,
        double scaledValue,
        int v0,
        int v1)
    {
        (double scaled, int lower, int upper) = GetBilinearSource(value, scale, halfPixelCenters, shapeSize);
        return new double[3] { scaled, lower, upper };
    }

    /// <summary>Source index of destination index <paramref name="input"/> for nearest-neighbour resize.</summary>
    public static int GetNearestNeighbor(
        float input,
        int shapeSize,
        float scale,
        bool alignCorners,
        bool halfPixelCenters)
    {
        float halfPixelOffset = halfPixelCenters ? 0.5f : 0f;
        float scaled = (input + halfPixelOffset) * scale;
        int nearest = System.Math.Min(
            (int)(alignCorners ? System.Math.Round(scaled) : System.Math.Floor(scaled)),
            shapeSize - 1);
        if (halfPixelCenters)
        {
            nearest = System.Math.Max(0, nearest);
        }

        return nearest;
    }

    private static (double Scaled, int Lower, int Upper) GetBilinearSource(
        int destIndex,
        double scale,
        bool halfPixelCenters,
        int sourceSize)
    {
        double scaled = halfPixelCenters
            ? (((double)destIndex + 0.5) * scale - 0.5)
            : ((double)destIndex * scale);
        int lower = (int)System.Math.Max(System.Math.Floor(scaled), 0.0);
        int upper = (int)System.Math.Min(System.Math.Ceiling(scaled), sourceSize - 1);
        return (scaled, lower, upper);
    }

    private static int[] ResizedDims(Tensor input, int[] newSize)
    {
        int[] inputDims = input.Shape.ToValueArray();
        return new int[4] { inputDims[0], inputDims[1], newSize[0], newSize[1] };
    }

    private static float[] ResizeBilinear(Tensor input, int[] newSize, bool alignCorners, bool halfPixelCenters)
    {
        int[] inputDims = input.Shape.ToValueArray();
        int outputHeight = newSize[0];
        int outputWidth = newSize[1];
        float[] scales = GetResizeScales(inputDims, outputHeight, outputWidth, alignCorners);
        float heightScale = scales[0];
        float widthScale = scales[1];
        int inputHeight = inputDims[2];
        int inputWidth = inputDims[3];
        float[] source = input.ToArray<float>();
        float[] output = new float[inputDims[0] * inputDims[1] * outputHeight * outputWidth];

        for (int batch = 0; batch < inputDims[0]; batch++)
        {
            for (int channel = 0; channel < inputDims[1]; channel++)
            {
                int inputPlaneOffset = batch * inputDims[1] * inputHeight * inputWidth +
                                       channel * inputHeight * inputWidth;
                int outputPlaneOffset = batch * inputDims[1] * outputHeight * outputWidth +
                                        channel * outputHeight * outputWidth;
                for (int y = 0; y < outputHeight; y++)
                {
                    (double scaledYDouble, int yLower, int yUpper) =
                        GetBilinearSource(y, heightScale, halfPixelCenters, inputHeight);
                    float yFraction = (float)scaledYDouble - (float)yLower;
                    for (int x = 0; x < outputWidth; x++)
                    {
                        (double scaledXDouble, int xLower, int xUpper) =
                            GetBilinearSource(x, widthScale, halfPixelCenters, inputWidth);
                        float xFraction = (float)scaledXDouble - (float)xLower;

                        float topLeft = source[inputPlaneOffset + yLower * inputWidth + xLower];
                        float bottomLeft = source[inputPlaneOffset + yUpper * inputWidth + xLower];
                        float topRight = source[inputPlaneOffset + yLower * inputWidth + xUpper];
                        float bottomRight = source[inputPlaneOffset + yUpper * inputWidth + xUpper];

                        float topLeftWeight = (1f - yFraction) * (1f - xFraction);
                        float bottomLeftWeight = yFraction * (1f - xFraction);
                        float topRightWeight = (1f - yFraction) * xFraction;
                        float bottomRightWeight = yFraction * xFraction;

                        output[outputPlaneOffset + y * outputWidth + x] =
                            topLeft * topLeftWeight + bottomLeft * bottomLeftWeight +
                            topRight * topRightWeight + bottomRight * bottomRightWeight + BilinearBias;
                    }
                }
            }
        }

        return output;
    }

    private static float[] ResizeNearestNeighbor(Tensor input, int[] newSize, bool alignCorners, bool halfPixelCenters)
    {
        int[] inputDims = input.Shape.ToValueArray();
        int outputHeight = newSize[0];
        int outputWidth = newSize[1];
        float[] scales = GetResizeScales(inputDims, outputHeight, outputWidth, alignCorners);
        float heightScale = scales[0];
        float widthScale = scales[1];
        int inputHeight = inputDims[2];
        int inputWidth = inputDims[3];
        float[] source = input.ToArray<float>();
        float[] output = new float[inputDims[0] * inputDims[1] * outputHeight * outputWidth];

        for (int batch = 0; batch < inputDims[0]; batch++)
        {
            for (int channel = 0; channel < inputDims[1]; channel++)
            {
                int inputPlaneOffset = batch * inputDims[1] * inputHeight * inputWidth +
                                       channel * inputHeight * inputWidth;
                int outputPlaneOffset = batch * inputDims[1] * outputHeight * outputWidth +
                                        channel * outputHeight * outputWidth;
                for (int y = 0; y < outputHeight; y++)
                {
                    int sourceY = GetNearestNeighbor(y, inputHeight, heightScale, alignCorners, halfPixelCenters);
                    for (int x = 0; x < outputWidth; x++)
                    {
                        int sourceX = GetNearestNeighbor(x, inputWidth, widthScale, alignCorners, halfPixelCenters);
                        output[outputPlaneOffset + y * outputWidth + x] =
                            source[inputPlaneOffset + sourceY * inputWidth + sourceX];
                    }
                }
            }
        }

        return output;
    }

    /// <summary>Truncating conversion of resize results to uint8 / int8 / float16 / int16; anything else stays float.</summary>
    private static Tensor ConvertResized(PrimType dataType, float[] resized, int[] dims)
    {
        if (dataType == DataTypes.UInt8)
        {
            return new Tensor<byte>(ConvertAll(resized, value => (byte)value), dims);
        }

        if (dataType == DataTypes.Int8)
        {
            return new Tensor<sbyte>(ConvertAll(resized, value => (sbyte)value), dims);
        }

        if (dataType == DataTypes.Float16)
        {
            return new Tensor<Half>(ConvertAll(resized, value => (Half)value), dims);
        }

        if (dataType == DataTypes.Int16)
        {
            return new Tensor<short>(ConvertAll(resized, value => (short)value), dims);
        }

        return new Tensor<float>(resized, dims);
    }

    /// <summary>True for the half-pixel transformation modes (HalfPixel, PytorchHalfPixel).</summary>
    public static bool IsAnyHalfPixel(ImageResizeTransformationMode mode)
    {
        return mode == ImageResizeTransformationMode.HalfPixel ||
               mode == ImageResizeTransformationMode.PytorchHalfPixel;
    }

    public static bool CanBeLoweredToCrop(ResizeImage r)
    {
        return CanBeLoweredToCrop(r.ResizeMode, r.NearestMode, r.TransformationMode);
    }

    /// <summary>True when a resize is equivalent to a crop (pure index selection, no interpolation).</summary>
    public static bool CanBeLoweredToCrop(
        ImageResizeMode resizeMode,
        ImageResizeNearestMode nearestMode,
        ImageResizeTransformationMode transformationMode)
    {
        if (transformationMode == ImageResizeTransformationMode.TFCropAndResize)
        {
            return false;
        }

        if (resizeMode == ImageResizeMode.Bilinear && IsAnyHalfPixel(transformationMode))
        {
            return false;
        }

        if (resizeMode == ImageResizeMode.NearestNeighbor)
        {
            switch (nearestMode)
            {
                case ImageResizeNearestMode.Ceil:
                    return false;
                case ImageResizeNearestMode.Floor:
                    return transformationMode == ImageResizeTransformationMode.Asymmetric;
                case ImageResizeNearestMode.RoundPreferCeil:
                    if (transformationMode == ImageResizeTransformationMode.Asymmetric)
                    {
                        return false;
                    }

                    break;
            }
        }

        return true;
    }

    // ------------------------------------------------------------------------------------------------------
    // Dynamic matmul: [aBatch0, aBatch1, aRows, aCols] x [bBatch0, bBatch1, aCols, bCols], size-1 batches broadcast.
    // ------------------------------------------------------------------------------------------------------

    /// <summary>Quantized matmul with per-row activation; output type selects float / Half / rounded integer.</summary>
    public static void DynamicGnneMatmul<TIA, TIB, TO>(
        ReadOnlySpan<TIA> inputA,
        ReadOnlySpan<TIB> inputB,
        Span<TO> output,
        ReadOnlySpan<Half> act,
        ReadOnlySpan<byte> inABias,
        int aBatch0,
        int aBatch1,
        int aRows,
        int aCols,
        int bBatch0,
        int bBatch1,
        int bCols,
        byte inBBias,
        int inAShiftBits,
        int inBShiftBits,
        sbyte shiftBits,
        bool dynamicChannel = true)
        where TIA : unmanaged
        where TIB : unmanaged
        where TO : unmanaged
    {
        ValidateBatchBroadcast(aBatch0, bBatch0, aBatch1, bBatch1, "inputA");

        int[] aStrides = TensorUtilities.GetStrides(new int[4] { aBatch0, aBatch1, aRows, aCols });
        int[] bStrides = TensorUtilities.GetStrides(new int[4] { bBatch0, bBatch1, aCols, bCols });
        int outBatch0 = System.Math.Max(aBatch0, bBatch0);
        int outBatch1 = System.Math.Max(aBatch1, bBatch1);
        int[] outStrides = TensorUtilities.GetStrides(new int[4] { outBatch0, outBatch1, aRows, bCols });

        for (int batch0 = 0; batch0 < outBatch0; batch0++)
        {
            for (int batch1 = 0; batch1 < outBatch1; batch1++)
            {
                for (int row = 0; row < aRows; row++)
                {
                    int outputRowOffset = TensorUtilities.GetIndex(outStrides, new int[4] { batch0, batch1, row, 0 });
                    for (int col = 0; col < bCols; col++)
                    {
                        float sum = 0f;
                        int aOffset = TensorUtilities.GetIndex(
                            aStrides,
                            new int[4]
                            {
                                (batch0 >= aBatch0) ? (aBatch0 - 1) : batch0,
                                (batch1 >= aBatch1) ? (aBatch1 - 1) : batch1,
                                row,
                                0,
                            });
                        int bOffset = TensorUtilities.GetIndex(
                            bStrides,
                            new int[4]
                            {
                                (batch0 >= bBatch0) ? (bBatch0 - 1) : batch0,
                                (batch1 >= bBatch1) ? (bBatch1 - 1) : batch1,
                                0,
                                col,
                            });
                        for (int k = 0; k < aCols; k++)
                        {
                            TIA a = inputA[aOffset + k];
                            TIB b = inputB[bOffset + k * bCols];
                            sum += ((float)Convert.ChangeType(a, typeof(float)) - (float)(int)inABias[row]) *
                                   ((float)Convert.ChangeType(b, typeof(float)) - (float)(int)inBBias);
                        }

                        int activationChannel = dynamicChannel ? 0 : row;
                        int outputIndex = outputRowOffset + col;
                        if (typeof(TO) == typeof(float))
                        {
                            output[outputIndex] = (TO)(object)ApplyAct0(sum, act, activationChannel, shiftBits);
                        }
                        else if (typeof(TO) == typeof(Half))
                        {
                            output[outputIndex] = (TO)(object)(Half)ApplyAct0(sum, act, activationChannel, shiftBits);
                        }
                        else if (typeof(TO) == typeof(byte) || typeof(TO) == typeof(sbyte) ||
                                 typeof(TO) == typeof(short))
                        {
                            output[outputIndex] = (TO)Convert.ChangeType(
                                System.Math.Round(ApplyAct0(sum, act, activationChannel, shiftBits)),
                                typeof(TO));
                        }
                        else
                        {
                            throw new ArgumentOutOfRangeException("inputA");
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Float matmul with the activation applied per row (or per matrix when <paramref name="dynamicChannel"/>).
    /// When the call has a best quant config, marker-bound quant params are applied to the inputs first.
    /// </summary>
    public static void FakeDynamicGnneMatmul(
        IEvaluateContext context,
        Span<float> inputA,
        Span<float> inputB,
        Span<float> output,
        ReadOnlySpan<float> act,
        int aBatch0,
        int aBatch1,
        int aRows,
        int aCols,
        int bBatch0,
        int bBatch1,
        int bCols,
        bool dynamicChannel)
    {
        if (context.CurrentCall.EnodeBestQuantConfigWithCosine != null)
        {
            FakeQuantizeDynamicMatmulInputs(context, inputA, inputB);
        }

        ValidateBatchBroadcast(aBatch0, bBatch0, aBatch1, bBatch1, "context");

        int[] aStrides = TensorUtilities.GetStrides(new int[4] { aBatch0, aBatch1, aRows, aCols });
        int[] bStrides = TensorUtilities.GetStrides(new int[4] { bBatch0, bBatch1, aCols, bCols });
        int outBatch0 = System.Math.Max(aBatch0, bBatch0);
        int outBatch1 = System.Math.Max(aBatch1, bBatch1);
        int[] outStrides = TensorUtilities.GetStrides(new int[4] { outBatch0, outBatch1, aRows, bCols });

        for (int batch0 = 0; batch0 < outBatch0; batch0++)
        {
            for (int batch1 = 0; batch1 < outBatch1; batch1++)
            {
                for (int row = 0; row < aRows; row++)
                {
                    int outputRowOffset = TensorUtilities.GetIndex(outStrides, new int[4] { batch0, batch1, row, 0 });
                    for (int col = 0; col < bCols; col++)
                    {
                        float sum = 0f;
                        int aOffset = TensorUtilities.GetIndex(
                            aStrides,
                            new int[4]
                            {
                                (batch0 >= aBatch0) ? (aBatch0 - 1) : batch0,
                                (batch1 >= aBatch1) ? (aBatch1 - 1) : batch1,
                                row,
                                0,
                            });
                        int bOffset = TensorUtilities.GetIndex(
                            bStrides,
                            new int[4]
                            {
                                (batch0 >= bBatch0) ? (bBatch0 - 1) : batch0,
                                (batch1 >= bBatch1) ? (bBatch1 - 1) : batch1,
                                0,
                                col,
                            });
                        for (int k = 0; k < aCols; k++)
                        {
                            float a = inputA[aOffset + k];
                            float b = inputB[bOffset + k * bCols];
                            sum += a * b;
                        }

                        output[outputRowOffset + col] = ApplyAct01(sum, act, dynamicChannel ? 0 : row, 0);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Replaces the matmul inputs by their quantize-dequantize round trip using the quant params bound to the
    /// marker arguments (input A: one param; input B: one param per equally sized chunk).
    /// </summary>
    private static void FakeQuantizeDynamicMatmulInputs(IEvaluateContext context, Span<float> inputA, Span<float> inputB)
    {
        MarkerPattern markerPattern = Utility.IsRangeOfMarker(Utility.IsWildcard(), Utility.IsWildcard());
        if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[0]))
        {
            MixQuantInfo? mixQuantInfoA = ((Marker)context.CurrentCall.Arguments[0]).MixQuantInfo;
            if (mixQuantInfoA != null && mixQuantInfoA.HasBindedMixQuantInfo)
            {
                List<QuantParam> quantParams = mixQuantInfoA.QuantParameter;
                Trace.Assert(quantParams.Count == 1);
                for (int i = 0; i < inputA.Length; i++)
                {
                    inputA[i] = FakeQuantize(inputA[i], quantParams[0]);
                }
            }
        }

        if (markerPattern.MatchLeaf(context.CurrentCall.Arguments[1]))
        {
            MixQuantInfo? mixQuantInfoB = ((Marker)context.CurrentCall.Arguments[1]).MixQuantInfo;
            if (mixQuantInfoB != null && mixQuantInfoB.HasBindedMixQuantInfo)
            {
                List<QuantParam> quantParams = mixQuantInfoB.QuantParameter;
                int chunkLength = inputB.Length / quantParams.Count;
                for (int i = 0; i < inputB.Length; i++)
                {
                    inputB[i] = FakeQuantize(inputB[i], quantParams[i / chunkLength]);
                }
            }
        }
    }

    /// <summary>Quantize (round unless the param is the identity) and dequantize one value.</summary>
    private static float FakeQuantize(float value, QuantParam quantParam)
    {
        double quantized = (double)value / (double)quantParam.Scale + (double)quantParam.ZeroPoint;
        if (quantParam.Scale != 1f || quantParam.ZeroPoint != 0)
        {
            quantized = System.Math.Round(quantized);
        }

        return (float)((quantized - (double)quantParam.ZeroPoint) * (double)quantParam.Scale);
    }

    /// <summary>Throws when two batch dimensions differ and neither of them is 1.</summary>
    private static void ValidateBatchBroadcast(int aBatch0, int bBatch0, int aBatch1, int bBatch1, string paramName)
    {
        if (aBatch0 != bBatch0 && aBatch0 != 1 && bBatch0 != 1)
        {
            throw new ArgumentOutOfRangeException(paramName);
        }

        if (aBatch1 != bBatch1 && aBatch1 != 1 && bBatch1 != 1)
        {
            throw new ArgumentOutOfRangeException(paramName);
        }
    }
}
