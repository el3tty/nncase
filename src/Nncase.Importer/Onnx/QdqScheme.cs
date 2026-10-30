// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Nncase.Diagnostics;
using Nncase.Quantization;
using Nncase.Utilities;
using Onnx;

namespace Nncase.Importer;

/// <summary>
/// Derives a quant scheme from the QuantizeLinear/DequantizeLinear pairs of a QAT model (same rules as
/// models/model_runner/util/quant_scheme.py), so a QDQ model can be compiled with UsePTQ without a .qs.json.
/// Weight pairs on constants are folded into float weights at import (lossless for every target);
/// activation pairs stay in the IR and are removed by the target (FoldQuantDeQuant) before quantization.
/// </summary>
public sealed partial class OnnxImporter
{
    // nncase widens ranges narrower than this (QuantUtility.FixupRange)
    private const float MinRangeWidth = 0.01f;

    // range-preserving ops: output values are copies/rearrangements of input values
    private static readonly HashSet<string> _rangePreservingOps = new()
    {
        "Reshape", "Transpose", "Slice", "Squeeze", "Unsqueeze", "Flatten", "Identity", "MaxPool", "Resize",
        "Upsample", "Split", "Pad", "Gather", "Expand", "Tile",
    };

    // QuantizeLinear / DequantizeLinear output of a folded weight pair -> fake-quantized float weights
    private Dictionary<string, Tensor>? _qdqFoldedWeights;

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

    private static Output MakeEntry(string name, string mode, IEnumerable<ValueRange<float>> ranges) =>
        new() { Name = name, DataType = "u8", DataRange = ranges.ToArray(), DataRangeMode = mode };

    /// <summary>
    /// Quantizer.FixUpRange (asymmetric), applied to every scheme range before QuantUtility.GetQuantParam.
    /// </summary>
    private static ValueRange<float> QuantizerFixUpRange(ValueRange<float> range)
    {
        range.Max = Math.Max(range.Max, 0);
        range.Min = Math.Min(range.Min, 0);
        var r = range.Max - range.Min;
        if (r < 1e-6f)
        {
            r = 0.1f;
        }

        range.Max = range.Min + r;
        return range;
    }

    private static ValueRange<float> QdqRange(float scale, int zp) =>
        new((float)((byte.MinValue - zp) * (double)scale), (float)((byte.MaxValue - zp) * (double)scale));

    /// <summary>
    /// The range of an exported (scale, zp). A range narrower than nncase's minimum width is widened by increasing the
    /// scale and keeping the zero point, so nncase's own fixup leaves it alone and real 0 keeps its code.
    /// The range is then replayed through nncase's range -> param computation and any difference is reported.
    /// </summary>
    private static ValueRange<float> GetCheckedRange(string name, float scale, int zp, List<string> widened, List<string> mismatched)
    {
        var range = QdqRange(scale, zp);
        if (range.Max - range.Min < MinRangeWidth)
        {
            var newScale = MinRangeWidth / byte.MaxValue;
            while (QdqRange(newScale, zp) is var r && r.Max - r.Min < MinRangeWidth)
            {
                newScale = MathF.BitIncrement(newScale);
            }

            widened.Add(FormattableString.Invariant($"{name}: scale {scale:G9} -> {newScale:G9}, zero point {zp}"));
            scale = newScale;
            range = QdqRange(scale, zp);
        }

        var param = QuantUtility.GetQuantParam(QuantizerFixUpRange(range), 8, QuantMode.UnsignedMode);
        if (param.ZeroPoint != zp || Math.Abs(param.Scale - scale) > 1e-5 * scale)
        {
            mismatched.Add(FormattableString.Invariant($"{name}: exported (scale {scale:G9}, zp {zp}), nncase (scale {param.Scale:G9}, zp {param.ZeroPoint})"));
        }

        return range;
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
        foreach (var pair in pairs.Where(p => consts.ContainsKey(p.Input)))
        {
            var folded = FakeQuantize(GetTensor(consts[pair.Input]), pair);
            _qdqFoldedWeights[pair.Q.Output[0]] = folded;
            _qdqFoldedWeights[pair.DQ.Output[0]] = folded;
            _qdqNames[pair.DQ.Output[0]] = pair.Input;
        }

        var quantizeOptions = CompileSession.CompileOptions.QuantizeOptions;
        if (quantizeOptions.ModelQuantMode == ModelQuantMode.UsePTQ)
        {
            foreach (var pair in pairs)
            {
                _qdqNames[pair.DQ.Output[0]] = pair.Input;
            }

            quantizeOptions.ImportedQuantScheme = BuildQuantScheme(pairs, consts);
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

    private QuantScheme BuildQuantScheme(List<QdqPair> pairs, Dictionary<string, TensorProto> consts)
    {
        var entries = new Dictionary<string, Output>();
        var order = new List<Output>();
        var alias = new Dictionary<string, string>();
        var widened = new List<string>();
        var mismatched = new List<string>();

        void Add(Output entry)
        {
            entries[entry.Name!] = entry;
            order.Add(entry);
        }

        string Resolve(string t) => alias.TryGetValue(t, out var x) ? x : t;

        foreach (var pair in pairs)
        {
            var x = pair.Input;
            alias[pair.DQ.Output[0]] = x;
            // an int8 grid with zero point zp is the uint8 grid with zero point zp + 128: same scale, same values
            var zpOffset = pair.ZeroPointType switch
            {
                TensorProto.Types.DataType.Uint8 => 0,
                TensorProto.Types.DataType.Int8 => 128,
                _ => throw new NotSupportedException($"Q/DQ on {x}: zero point type {pair.ZeroPointType} is not supported, only 8-bit quantization can be compiled from a QDQ model."),
            };

            var isWeight = consts.ContainsKey(x);
            if (!isWeight && pair.Scales.Length != 1)
            {
                throw new NotSupportedException($"Q/DQ on {x}: per-channel activation quantization is not supported.");
            }

            var ranges = pair.Scales.Select((s, c) => GetCheckedRange(pair.Scales.Length == 1 ? x : $"{x}[{c}]", s, pair.ZeroPoints[c] + zpOffset, widened, mismatched));
            // a per-tensor weight range is "by_tensor": the quantizer repeats it for every output channel
            Add(MakeEntry(x, pair.Scales.Length > 1 ? "by_channel" : "by_tensor", ranges));
        }

        // Conv weights that are plain initializers (weights baked by the QAT export)
        foreach (var n in _graph.Node.Where(n => (n.OpType == "Conv" || n.OpType == "ConvTranspose") && n.Input.Count > 1))
        {
            var w = Resolve(n.Input[1]);
            if (consts.TryGetValue(w, out var proto) && !entries.ContainsKey(w) && proto.DataType == (int)TensorProto.Types.DataType.Float)
            {
                var weights = GetTensor(proto);
                var dims = weights.Dimensions.ToArray();
                var values = weights.ToArray<float>();
                var oc = n.OpType == "ConvTranspose" ? dims[1] : dims[0];
                var ranges = Enumerable.Range(0, oc).Select(_ => new ValueRange<float>(float.MaxValue, float.MinValue)).ToArray();
                var inner = dims.Skip(n.OpType == "ConvTranspose" ? 2 : 1).Aggregate(1L, (a, b) => a * b);
                for (long i = 0; i < values.Length; i++)
                {
                    var c = (int)(i / inner % oc);
                    ranges[c] = new(Math.Min(ranges[c].Min, values[i]), Math.Max(ranges[c].Max, values[i]));
                }

                Add(MakeEntry(w, "by_channel", ranges.Select(QuantizerFixUpRange)));
            }
        }

        // graph inputs and range-preserving ops
        ValueRange<float>? GetRange(string t) => entries.TryGetValue(Resolve(t), out var e) ? e.DataRange![0] : null;

        var inputRange = CompileSession.CompileOptions.InputRange;
        foreach (var input in _graph.Input.Where(i => !consts.ContainsKey(i.Name) && !entries.ContainsKey(i.Name)))
        {
            if (inputRange.Length == 2)
            {
                Add(MakeEntry(input.Name, "by_tensor", new ValueRange<float>[] { new(inputRange[0], inputRange[1]) }));
            }
        }

        foreach (var n in _graph.Node)
        {
            if (n.OpType is "QuantizeLinear" or "DequantizeLinear" or "Constant" || n.Input.Count == 0 || entries.ContainsKey(Resolve(n.Output[0])))
            {
                continue;
            }

            ValueRange<float>? src = null;
            if (_rangePreservingOps.Contains(n.OpType))
            {
                src = GetRange(n.Input[0]);
            }
            else if (n.OpType == "Concat")
            {
                var rs = n.Input.Select(GetRange).ToArray();
                if (rs[0] is not null && rs.All(r => r == rs[0]))
                {
                    src = rs[0];
                }
            }

            if (src is ValueRange<float> range)
            {
                foreach (var o in n.Output)
                {
                    Add(MakeEntry(o, "by_tensor", new[] { range }));
                }
            }
        }

        var missing = FindMissingQdq(consts, t => entries.ContainsKey(Resolve(t)));
        var scheme = new QuantScheme { Version = "1.0", Model = null, Outputs = order.ToArray() };
        ReportQuantScheme(scheme, widened, mismatched, missing);
        return scheme;
    }

    /// <summary>
    /// Tensors that the hardware has to quantize but that have no Q/DQ pair (they would silently stay float16).
    /// </summary>
    private List<string> FindMissingQdq(Dictionary<string, TensorProto> consts, Func<string, bool> hasRange)
    {
        var missing = new List<string>();
        void Check(string t, string reason)
        {
            if (!consts.ContainsKey(t) && !hasRange(t) && !missing.Any(m => m.StartsWith(t + ":", StringComparison.Ordinal)))
            {
                missing.Add($"{t}: {reason}");
            }
        }

        foreach (var input in _graph.Input.Where(i => !consts.ContainsKey(i.Name)))
        {
            Check(input.Name, "graph input (no Q/DQ and no --input-range)");
        }

        foreach (var n in _graph.Node)
        {
            switch (n.OpType)
            {
                case "Conv" or "ConvTranspose" or "MatMul" or "Gemm":
                    Check(n.Input[0], $"data input of {n.OpType} {n.Name}");
                    break;
                case "Add" or "Sub" or "Mul" or "Concat":
                    var inputs = n.Input.Where(t => !consts.ContainsKey(t)).ToArray();
                    if (inputs.Any(hasRange))
                    {
                        foreach (var t in inputs)
                        {
                            Check(t, $"input of {n.OpType} {n.Name} whose other inputs are quantized");
                        }
                    }

                    break;
            }
        }

        return missing;
    }

    private void ReportQuantScheme(QuantScheme scheme, List<string> widened, List<string> mismatched, List<string> missing)
    {
        static void Warn(string what, List<string> items)
        {
            if (items.Count != 0)
            {
                Console.WriteLine($"Warning: {items.Count} {what}:");
                foreach (var item in items.Take(5))
                {
                    Console.WriteLine($"    {item}");
                }

                if (items.Count > 5)
                {
                    Console.WriteLine($"    ... (see ImportedQuantSchemeReport.txt in the dump directory)");
                }
            }
        }

        var outputs = scheme.Outputs!;
        Console.WriteLine($"Imported quant scheme from Q/DQ: {outputs.Length} tensors ({outputs.Count(o => o.DataRangeMode == "by_tensor")} by_tensor, {outputs.Count(o => o.DataRangeMode == "by_channel")} by_channel)");
        Warn("Q/DQ ranges narrower than nncase's minimum were widened", widened);
        Warn("Q/DQ params that nncase will not reproduce exactly", mismatched);
        Warn("tensors that should have a Q/DQ pair but have none (they will stay float16)", missing);

        if (Dumpper.IsEnabled(DumpFlags.ImportOps))
        {
            using (var stream = Dumpper.OpenFile("ImportedQuantScheme.json"))
            {
                JsonSerializer.Serialize(stream, scheme, new JsonSerializerOptions { WriteIndented = true });
            }

            using var writer = new StreamWriter(Dumpper.OpenFile("ImportedQuantSchemeReport.txt"));
            foreach (var (title, items) in new[] { ("widened", widened), ("mismatched", mismatched), ("missing", missing) })
            {
                writer.WriteLine($"# {title}: {items.Count}");
                items.ForEach(writer.WriteLine);
            }
        }

        static bool Close(float a, float b) => Math.Abs(a - b) <= 1e-5f * Math.Max(1f, Math.Abs(b));

        var file = CompileSession.CompileOptions.QuantizeOptions.QuantScheme;
        if (file != string.Empty)
        {
            var given = JsonSerializer.Deserialize<QuantScheme>(File.ReadAllText(file), new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            var byName = given?.Outputs?.Where(o => o.Name != null).ToDictionary(o => o.Name!) ?? new();
            var differ = outputs.Where(o => !byName.TryGetValue(o.Name!, out var g) || g.DataType != o.DataType
                || g.DataRange!.Length != o.DataRange!.Length
                || !g.DataRange.Zip(o.DataRange, (a, b) => Close(a.Min, b.Min) && Close(a.Max, b.Max)).All(ok => ok))
                .Select(o => o.Name!).ToList();
            Warn($"Q/DQ ranges differ from or are missing in {file}, which is used instead", differ);
        }
    }

    private sealed record QdqPair(NodeProto Q, NodeProto DQ, string Input, float[] Scales, int[] ZeroPoints, int Axis, TensorProto.Types.DataType ZeroPointType);
}
