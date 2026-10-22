// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#define TRACE
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Nncase.Utilities;

namespace Nncase.Passes.Rules.K230;

public class QuantizeManager
{
    private Tensor<float> _inRange;

    private Tensor<float> _wRange;

    private Tensor<float> _oRange;

    private ConvQuantConf _conf;

    private QuantizeParam[] _wQuantParam;

    private QuantizeParam[] _wQuantParamQint8;

    private Tensor _oldWeights;

    public ValueRange<float> InputRange => ToRange(_inRange);

    public ValueRange<float> OutRange => ToRange(_oRange);

    public QuantizeManager(Tensor<float> inRange, Tensor<float> wRange, Tensor<float> oRange, Tensor oldWeights,
        ConvQuantConf conf, bool is_matmul, int groupsConvTranspose = 1)
    {
        _inRange = inRange;
        _wRange = wRange;
        _oRange = oRange;
        _oldWeights = oldWeights;
        _conf = conf;
        _wQuantParam = WeightsByChannelQP(is_matmul, groupsConvTranspose);
        _wQuantParamQint8 = WeightsByChannelQPQint8(is_matmul, groupsConvTranspose);
    }

    public static ValueRange<float> ToRange(Tensor<float> t)
    {
        float[] bounds = t.ToArray();
        return new ValueRange<float>(bounds[0], bounds[1]);
    }

    public QuantizeParam GetIfQuantParam(DataType quantType, int inIdx = 0)
    {
        return QuantHelper.GetQuantParamFromDeqParam(GetIfDeqQuantParam(quantType, inIdx));
    }

    public DeQuantizeParam GetIfDeqQuantParam(DataType quantType, int inIdx = 0)
    {
        Trace.Assert(inIdx < 2);
        QuantMode quantMode =
            (!(quantType == DataTypes.UInt8)) ? QuantMode.SignedSymmetricMode : QuantMode.UnsignedMode;
        int bits = (quantType == DataTypes.Int16) ? 12 : 8;
        return QuantHelper.GetDeqParam(QuantUtility.GetQuantParam(InputRange, bits, quantMode));
    }

    public QuantizeParam[] GetWeightsQuantParam()
    {
        return _wQuantParam;
    }

    public QuantizeParam[] GetWeightsQuantParamQint8()
    {
        return _wQuantParamQint8;
    }

    public byte[] GetWeightsQuantBias()
    {
        QuantizeParam[] weightsQuantParam = GetWeightsQuantParam();
        byte[] result = new byte[weightsQuantParam.Length];
        for (int i = 0; i < weightsQuantParam.Length; i++)
        {
            result[i] = (byte)weightsQuantParam[i].ZeroPoint;
        }

        return result;
    }

    public byte[] GetWeightsQuantBiasQint8()
    {
        QuantizeParam[] weightsQuantParamQint = GetWeightsQuantParamQint8();
        byte[] result = new byte[weightsQuantParamQint.Length];
        for (int i = 0; i < weightsQuantParamQint.Length; i++)
        {
            result[i] = (byte)weightsQuantParamQint[i].ZeroPoint;
        }

        return result;
    }

    public float[] GetWeightsDeqScale()
    {
        QuantizeParam[] weightsQuantParam = GetWeightsQuantParam();
        float[] result = new float[weightsQuantParam.Length];
        for (int i = 0; i < weightsQuantParam.Length; i++)
        {
            result[i] = QuantHelper.GetDeqParamFromQuantParam(weightsQuantParam[i]).Scale;
        }

        return result;
    }

    public QuantT[] GetQuantWeights(bool isMatmul = false, int groupsConvTranspose = 1)
    {
        _ = _oldWeights.Shape;
        int channelCount = _oldWeights.Shape[0].FixedValue * groupsConvTranspose;
        if (isMatmul)
        {
            channelCount = _oldWeights.Shape[1].FixedValue * _oldWeights.Shape[2].FixedValue;
        }

        float[] weights = _oldWeights.ToArray<float>();
        int weightsPerChannel = weights.Length / channelCount;
        QuantT[] quantWeights = (from i in Enumerable.Range(0, weights.Length)
            select new QuantT(_conf.WQuantType == DataTypes.UInt8)).ToArray();
        QuantizeParam[] weightsQuantParam = GetWeightsQuantParam();
        QuantizeParam[] weightsQuantParamQint = GetWeightsQuantParamQint8();
        for (int channel = 0; channel < channelCount; channel++)
        {
            for (int index = 0; index < weightsPerChannel; index++)
            {
                int flatIndex = channel * weightsPerChannel + index;
                if (_conf.WQuantType == DataTypes.Int8)
                {
                    quantWeights[flatIndex].I8 =
                        (sbyte)Math.Clamp(
                            Convert.ToInt32(QuantHelper.Quantize<sbyte>(weights[flatIndex], weightsQuantParam[channel])), -127,
                            127);
                    continue;
                }

                if (_conf.WQuantType == DataTypes.UInt8)
                {
                    quantWeights[flatIndex].U8 = QuantHelper.Quantize<byte>(weights[flatIndex], weightsQuantParam[channel]);
                    continue;
                }

                if (_conf.WQuantType == DataTypes.Int16)
                {
                    quantWeights[flatIndex].U8 = QuantHelper.Quantize<byte>(weights[flatIndex], weightsQuantParamQint[channel]);
                    continue;
                }

                throw new InvalidDataException("Invalid w_quant_type");
            }
        }

        return quantWeights;
    }

    public short[] GetQuantWeightsI16(bool isMatmul = false, int groupsConvTranspose = 1)
    {
        _ = _oldWeights.Shape;
        int channelCount = _oldWeights.Shape[0].FixedValue * groupsConvTranspose;
        if (isMatmul)
        {
            channelCount = _oldWeights.Shape[1].FixedValue * _oldWeights.Shape[2].FixedValue;
        }

        float[] weights = _oldWeights.ToArray<float>();
        int weightsPerChannel = weights.Length / channelCount;
        short[] quantWeights = new short[weights.Length];
        QuantizeParam[] weightsQuantParam = GetWeightsQuantParam();
        for (int i = 0; i < channelCount; i++)
        {
            for (int j = 0; j < weightsPerChannel; j++)
            {
                int flatIndex = i * weightsPerChannel + j;
                if (_conf.WQuantType == DataTypes.Int16)
                {
                    quantWeights[flatIndex] =
                        (short)Math.Clamp((int)QuantHelper.Quantize<short>(weights[flatIndex], weightsQuantParam[i]), -2047,
                            2047);
                    continue;
                }

                throw new InvalidDataException("Invalid w_quant_type");
            }
        }

        return quantWeights;
    }

    public byte[] GetQuantWeightsU8(bool isMatmul = false, int groupsConvTranspose = 1)
    {
        _ = _oldWeights.Shape;
        int channelCount = _oldWeights.Shape[0].FixedValue * groupsConvTranspose;
        if (isMatmul)
        {
            channelCount = _oldWeights.Shape[1].FixedValue * _oldWeights.Shape[2].FixedValue;
        }

        float[] weights = _oldWeights.ToArray<float>();
        int weightsPerChannel = weights.Length / channelCount;
        byte[] quantWeights = new byte[weights.Length];
        QuantizeParam[] weightsQuantParamQint = GetWeightsQuantParamQint8();
        for (int i = 0; i < channelCount; i++)
        {
            for (int j = 0; j < weightsPerChannel; j++)
            {
                int flatIndex = i * weightsPerChannel + j;
                quantWeights[flatIndex] = QuantHelper.Quantize<byte>(weights[flatIndex], weightsQuantParamQint[i]);
            }
        }

        return quantWeights;
    }

    public QuantizeParam GetOfQuantParam(DataType quantType, int outIdx = 0)
    {
        return QuantHelper.GetQuantParamFromDeqParam(GetOfDeqQuantParam(quantType, outIdx));
    }

    public DeQuantizeParam GetOfDeqQuantParam(DataType quantType, int outIdx = 0)
    {
        QuantMode quantMode =
            (!(quantType == DataTypes.UInt8)) ? QuantMode.SignedSymmetricMode : QuantMode.UnsignedMode;
        int bits = (quantType == DataTypes.Int16) ? 12 : 8;
        return QuantHelper.GetDeqParam(QuantUtility.GetQuantParam(OutRange, bits, quantMode));
    }

    public QuantizeParam[] WeightsByChannelQP(bool isMatmul = false, int groupConvTranspose = 1)
    {
        int channelCount = _oldWeights.Shape[0].FixedValue * groupConvTranspose;
        if (isMatmul)
        {
            channelCount = (_oldWeights.Shape[1] * _oldWeights.Shape[2]).FixedValue;
        }

        float[] weights = _oldWeights.ToArray<float>();
        QuantizeParam[] channelParams = new QuantizeParam[channelCount];
        int weightsPerChannel = _oldWeights.Shape.Prod().FixedValue / channelCount;
        int bits = (_conf.WQuantType == DataTypes.Int16) ? 12 : 8;
        for (int i = 0; i < channelCount; i++)
        {
            _ = new Span<float>(weights, i * weightsPerChannel, weightsPerChannel);
            bool symmetric = _conf.WQuantType != DataTypes.UInt8;
            QuantMode quantMode = (!(_conf.WQuantType == DataTypes.UInt8))
                ? QuantMode.SignedAsymmetricMode
                : QuantMode.UnsignedMode;
            float min = (_wRange.Rank == 1) ? _wRange[new int[1]] : _wRange[new int[2] { i, 0 }];
            float max = (_wRange.Rank == 1) ? _wRange[new int[1] { 1 }] : _wRange[new int[2] { i, 1 }];
            QuantizeParam quantParam = QuantHelper.GetQuantParam(QuantUtility.GetQuantParam(
                QuantUtility.FixupRange(
                new ValueRange<float> { Min = min, Max = max }, symmetric), bits, quantMode));
            channelParams[i].Scale = quantParam.Scale;
            channelParams[i].ZeroPoint = quantParam.ZeroPoint;
        }

        return channelParams;
    }

    public QuantizeParam[] WeightsByChannelQPQint8(bool isMatmul = false, int groupsConvTranspose = 1)
    {
        int channelCount = _oldWeights.Shape[0].FixedValue * groupsConvTranspose;
        if (isMatmul)
        {
            channelCount = (_oldWeights.Shape[1] * _oldWeights.Shape[2]).FixedValue;
        }

        float[] weights = _oldWeights.ToArray<float>();
        QuantizeParam[] channelParams = new QuantizeParam[channelCount];
        int weightsPerChannel = _oldWeights.Shape.Prod().FixedValue / channelCount;
        if (_conf.UseMseQuantW)
        {
            QuantParam qp = default(QuantParam);
            QuantParam qp2 = default(QuantParam);
            for (int i = 0; i < channelCount; i++)
            {
                Span<float> span = new Span<float>(weights, i * weightsPerChannel, weightsPerChannel);
                float bestMinOffset = 0f;
                float bestMaxOffset = 0f;
                float bestError = float.MaxValue;
                float[] dequantized = Enumerable.Repeat(0f, weightsPerChannel).ToArray();
                byte[] quantized = Enumerable.Repeat((byte)0, weightsPerChannel).ToArray();
                float min = (_wRange.Rank == 1) ? _wRange[new int[1]] : _wRange[new int[2] { i, 0 }];
                float max = (_wRange.Rank == 1) ? _wRange[new int[1] { 1 }] : _wRange[new int[2] { i, 1 }];
                ValueRange<float> valueRange = QuantUtility.FixupRange(new ValueRange<float> { Min = min, Max = max });
                float stepSize = (valueRange.Max - valueRange.Min) / 256f;
                for (uint minStepIndex = 0u; minStepIndex < 32; minStepIndex++)
                {
                    QuantizeParam quantParam = QuantHelper.GetQuantParam(QuantUtility.GetQuantParam(
                        QuantUtility.FixupRange(new ValueRange<float>
                        {
                            Min = valueRange.Min - stepSize * (float)minStepIndex, Max = valueRange.Max,
                        }), 8, QuantMode.UnsignedMode));
                    qp.Scale = 1f / quantParam.Scale;
                    qp.ZeroPoint = quantParam.ZeroPoint;
                    for (int j = 0; j < span.Length; j++)
                    {
                        quantized[j] = QuantHelper.Quantize1<byte>(span[j], qp);
                        dequantized[j] = (float)(quantized[j] - qp.ZeroPoint) * qp.Scale;
                    }

                    float minSearchError = 0f;
                    for (int k = 0; k < weightsPerChannel; k++)
                    {
                        minSearchError += (float)Math.Pow(span[k] - dequantized[k], 2.0);
                    }

                    if (bestError > minSearchError)
                    {
                        bestMinOffset = stepSize * (float)minStepIndex;
                        bestError = minSearchError;
                    }
                }

                for (uint maxStepIndex = 0u; maxStepIndex < 32; maxStepIndex++)
                {
                    QuantizeParam quantParam2 = QuantHelper.GetQuantParam(QuantUtility.GetQuantParam(
                        QuantUtility.FixupRange(new ValueRange<float>
                        {
                            Min = valueRange.Min, Max = valueRange.Max + stepSize * (float)maxStepIndex,
                        }), 8, QuantMode.UnsignedMode));
                    qp2.Scale = 1f / quantParam2.Scale;
                    qp2.ZeroPoint = quantParam2.ZeroPoint;
                    for (int l = 0; l < span.Length; l++)
                    {
                        quantized[l] = QuantHelper.Quantize1<byte>(span[l], qp2);
                        dequantized[l] = (float)(quantized[l] - qp2.ZeroPoint) * qp2.Scale;
                    }

                    float maxSearchError = 0f;
                    for (int m = 0; m < weightsPerChannel; m++)
                    {
                        maxSearchError += (float)Math.Pow(span[m] - dequantized[m], 2.0);
                    }

                    if (bestError > maxSearchError)
                    {
                        bestMaxOffset = stepSize * (float)maxStepIndex;
                        bestError = maxSearchError;
                    }
                }

                ValueRange<float> range = default(ValueRange<float>);
                if (bestMaxOffset > 0f)
                {
                    range.Max = valueRange.Max + bestMaxOffset;
                    range.Min = valueRange.Min;
                }
                else if (bestMinOffset > 0f)
                {
                    range.Max = valueRange.Max;
                    range.Min = valueRange.Min - bestMinOffset;
                }
                else
                {
                    range.Max = valueRange.Max;
                    range.Min = valueRange.Min;
                }

                channelParams[i] = QuantHelper.GetQuantParam(QuantUtility.GetQuantParam(range, 8, QuantMode.UnsignedMode));
            }

            return channelParams;
        }

        int bits = 8;
        for (int n = 0; n < channelCount; n++)
        {
            _ = new Span<float>(weights, n * weightsPerChannel, weightsPerChannel);
            bool symmetric = false;
            QuantMode quantMode = QuantMode.UnsignedMode;
            float min2 = (_wRange.Rank == 1) ? _wRange[new int[1]] : _wRange[new int[2] { n, 0 }];
            float max2 = (_wRange.Rank == 1) ? _wRange[new int[1] { 1 }] : _wRange[new int[2] { n, 1 }];
            QuantizeParam quantParam3 = QuantHelper.GetQuantParam(QuantUtility.GetQuantParam(
                QuantUtility.FixupRange(
                new ValueRange<float> { Min = min2, Max = max2 }, symmetric), bits, quantMode));
            channelParams[n].Scale = quantParam3.Scale;
            channelParams[n].ZeroPoint = quantParam3.ZeroPoint;
        }

        return channelParams;
    }
}
