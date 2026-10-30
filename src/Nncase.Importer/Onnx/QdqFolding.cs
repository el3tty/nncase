// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Quantization;
using Onnx;

namespace Nncase.Importer;

/// <summary>
/// QuantizeLinear/DequantizeLinear pairs at import. A pair on a constant (weights) is folded into its float values with
/// the ONNX semantics, and the folded constant carries the integer grid (<see cref="FakeQuantGrid"/>), since the IR
/// Quantize/Dequantize ops cannot express a per-channel grid. The output of every pair is named after the quantized
/// tensor. The quant scheme is derived from the IR right after import (QdqQuantSchemePass).
/// </summary>
public sealed partial class OnnxImporter
{
    // QuantizeLinear / DequantizeLinear output of a folded weight pair -> fake-quantized float weights and their grid
    private Dictionary<string, (Tensor Weights, FakeQuantGrid Grid)>? _qdqFoldedWeights;

    // DequantizeLinear output -> name of the quantized tensor: the DequantizeLinear call (and whatever replaces it, a
    // folded weight or, after FoldQuantDeQuant, the quantized tensor itself) carries the name the quant scheme uses
    private Dictionary<string, string>? _qdqNames;

    private static (int QMin, int QMax) GetQuantLimits(TensorProto.Types.DataType type) => type switch
    {
        TensorProto.Types.DataType.Uint8 => (byte.MinValue, byte.MaxValue),
        TensorProto.Types.DataType.Int8 => (sbyte.MinValue, sbyte.MaxValue),
        TensorProto.Types.DataType.Uint16 => (ushort.MinValue, ushort.MaxValue),
        TensorProto.Types.DataType.Int16 => (short.MinValue, short.MaxValue),
        _ => throw new NotSupportedException($"Unsupported Q/DQ zero point type {type}."),
    };

    private static DataType GetQuantType(TensorProto.Types.DataType type) => type switch
    {
        TensorProto.Types.DataType.Uint8 => DataTypes.UInt8,
        TensorProto.Types.DataType.Int8 => DataTypes.Int8,
        TensorProto.Types.DataType.Uint16 => DataTypes.UInt16,
        TensorProto.Types.DataType.Int16 => DataTypes.Int16,
        _ => throw new NotSupportedException($"Unsupported Q/DQ zero point type {type}."),
    };

    /// <summary>
    /// DequantizeLinear(QuantizeLinear(w)) with the ONNX semantics (float32 math, round half to even).
    /// </summary>
    private static Tensor FakeQuantize(Tensor weights, QdqPair pair)
    {
        var (qMin, qMax) = GetQuantLimits(pair.ZeroPointType);
        var dims = weights.Dimensions.ToArray();
        var axis = pair.Axis < 0 ? pair.Axis + dims.Length : pair.Axis;
        var channels = pair.Scales.Length;
        var inner = channels == 1 ? 1L : dims.Skip(axis + 1).Aggregate(1L, (a, b) => a * b);
        var values = weights.ToArray<float>();
        var result = new float[values.Length];
        for (long i = 0; i < values.Length; i++)
        {
            var c = channels == 1 ? 0 : (int)(i / inner % channels);
            var s = pair.Scales[c];
            var zp = pair.ZeroPoints[c];
            var q = Math.Clamp(MathF.Round(values[i] / s) + zp, qMin, qMax);
            result[i] = (q - zp) * s;
        }

        return Tensor.From(result, weights.Dimensions);
    }

    private void ProcessQdqPairs()
    {
        var consts = new Dictionary<string, TensorProto>(_constTensors!);
        foreach (var n in _graph.Node.Where(n => n.OpType == "Constant"))
        {
            var value = n.Attribute.FirstOrDefault(a => a.Name == "value")?.T;
            if (value != null)
            {
                consts.TryAdd(n.Output[0], value);
            }
        }

        var pairs = FindQdqPairs(consts);
        if (pairs.Count == 0)
        {
            return;
        }

        _qdqFoldedWeights = new();
        _qdqNames = new();
        foreach (var pair in pairs)
        {
            _qdqNames[pair.DQ.Output[0]] = pair.Input;
            if (consts.ContainsKey(pair.Input))
            {
                var folded = (FakeQuantize(GetTensor(consts[pair.Input]), pair), new FakeQuantGrid(GetQuantType(pair.ZeroPointType), pair.Scales, pair.ZeroPoints, pair.Axis));
                _qdqFoldedWeights[pair.Q.Output[0]] = folded;
                _qdqFoldedWeights[pair.DQ.Output[0]] = folded;
            }
        }
    }

    private List<QdqPair> FindQdqPairs(Dictionary<string, TensorProto> consts)
    {
        var producers = new Dictionary<string, NodeProto>();
        foreach (var n in _graph.Node)
        {
            foreach (var o in n.Output)
            {
                producers[o] = n;
            }
        }

        var pairs = new List<QdqPair>();
        foreach (var dq in _graph.Node.Where(n => n.OpType == "DequantizeLinear"))
        {
            if (!producers.TryGetValue(dq.Input[0], out var q) || q.OpType != "QuantizeLinear")
            {
                continue;
            }

            var qParams = GetQdqParams(q, consts);
            var dqParams = GetQdqParams(dq, consts);
            if (qParams is null || dqParams is null
                || !qParams.Value.Scales.SequenceEqual(dqParams.Value.Scales)
                || !qParams.Value.ZeroPoints.SequenceEqual(dqParams.Value.ZeroPoints)
                || qParams.Value.Type != dqParams.Value.Type
                || (qParams.Value.Scales.Length > 1 && qParams.Value.Axis != dqParams.Value.Axis))
            {
                continue;
            }

            pairs.Add(new(q, dq, q.Input[0], qParams.Value.Scales, qParams.Value.ZeroPoints, (int)qParams.Value.Axis, qParams.Value.Type));
        }

        return pairs;
    }

    private (float[] Scales, int[] ZeroPoints, long Axis, TensorProto.Types.DataType Type)? GetQdqParams(NodeProto n, Dictionary<string, TensorProto> consts)
    {
        if (!consts.TryGetValue(n.Input[1], out var scale))
        {
            return null;
        }

        var scales = GetTensor(scale).ToArray<float>();
        var zpType = TensorProto.Types.DataType.Uint8;
        var zeroPoints = new int[scales.Length];
        if (n.Input.Count > 2 && n.Input[2] != string.Empty)
        {
            if (!consts.TryGetValue(n.Input[2], out var zp))
            {
                return null;
            }

            zpType = (TensorProto.Types.DataType)zp.DataType;
            var zpValues = GetTensor(zp).ToArray<int>();
            zeroPoints = zpValues.Length == 1 ? Enumerable.Repeat(zpValues[0], scales.Length).ToArray() : zpValues;
        }

        return (scales, zeroPoints, GetIntAttribute(n, "axis", 1), zpType);
    }

    private sealed record QdqPair(NodeProto Q, NodeProto DQ, string Input, float[] Scales, int[] ZeroPoints, int Axis, TensorProto.Types.DataType ZeroPointType);
}
