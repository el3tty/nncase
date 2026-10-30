// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Google.Protobuf;
using Onnx;

namespace Nncase.Importer;

/// <summary>
/// Rewrites an onnxruntime QOperator model (QLinearConv, com.microsoft QLinearLeakyRelu/QLinearAdd/QLinearConcat,
/// MaxPool/Resize on quantized tensors) into the equivalent QDQ model: every quantized op becomes
/// DequantizeLinear -> float op -> QuantizeLinear. The Q/DQ pairs are then handled like those of a QAT export
/// (QdqScheme.cs): the weight pairs are folded and the activation pairs give the quant scheme.
/// </summary>
public sealed partial class OnnxImporter
{
    private const string QuantizedSuffix = "_quantized";

    private static readonly HashSet<string> _qLinearOps = new()
    {
        "QLinearConv", "QLinearLeakyRelu", "QLinearAdd", "QLinearConcat",
    };

    private static TensorProto MakeFloatTensor(string name, IEnumerable<long> dims, float[] values) => new()
    {
        Name = name,
        DataType = (int)TensorProto.Types.DataType.Float,
        Dims = { dims },
        RawData = ByteString.CopyFrom(MemoryMarshal.AsBytes(values.AsSpan())),
    };

    private void ExpandQLinearOps()
    {
        if (!_graph.Node.Any(n => _qLinearOps.Contains(n.OpType)))
        {
            return;
        }

        var consts = _graph.Initializer.ToDictionary(t => t.Name);
        foreach (var n in _graph.Node.Where(n => n.OpType == "Constant"))
        {
            var value = n.Attribute.FirstOrDefault(a => a.Name == "value")?.T;
            if (value != null)
            {
                consts.TryAdd(n.Output[0], value);
            }
        }

        var names = new HashSet<string>(_graph.Node.SelectMany(n => n.Input.Concat(n.Output)).Concat(consts.Keys).Concat(_graph.Input.Select(i => i.Name)));
        var nodes = new List<NodeProto>();

        // quantized tensor -> its (scale, zero point) names; dequantized copy of a quantized tensor
        var quantized = new Dictionary<string, (string Scale, string ZeroPoint)>();
        var dequantized = new Dictionary<string, string>();

        string NewName(string baseName)
        {
            var name = baseName;
            for (var i = 1; !names.Add(name); i++)
            {
                name = $"{baseName}_{i}";
            }

            return name;
        }

        // the float tensor that a quantized tensor X_quantized is computed from is named X, as in the source model
        string FloatName(string q) => NewName(q.EndsWith(QuantizedSuffix, StringComparison.Ordinal) ? q[..^QuantizedSuffix.Length] : q + "_float");

        void Emit(string opType, IEnumerable<string> inputs, string output, string domain = "", IEnumerable<AttributeProto>? attributes = null, string? name = null)
        {
            var node = new NodeProto { OpType = opType, Domain = domain, Name = name ?? output, Input = { inputs }, Output = { output } };
            if (attributes != null)
            {
                node.Attribute.Add(attributes);
            }

            nodes.Add(node);
        }

        string Dequantize(string q, string scale, string zp)
        {
            if (!quantized.TryGetValue(q, out var qp))
            {
                throw new NotSupportedException($"{q} is used as a quantized tensor but is not produced by a quantized op.");
            }

            if (!SameConst(qp.Scale, scale) || !SameConst(qp.ZeroPoint, zp))
            {
                throw new NotSupportedException($"{q} is dequantized with params ({scale}, {zp}) different from its own ({qp.Scale}, {qp.ZeroPoint}).");
            }

            if (!dequantized.TryGetValue(q, out var f))
            {
                f = NewName(q + "_dequantized");
                Emit("DequantizeLinear", new[] { q, qp.Scale, qp.ZeroPoint }, f);
                dequantized[q] = f;
            }

            return f;
        }

        void Quantize(string f, string q, string scale, string zp)
        {
            Emit("QuantizeLinear", new[] { f, scale, zp }, q);
            quantized[q] = (scale, zp);
        }

        bool SameConst(string a, string b) => a == b
            || (consts.TryGetValue(a, out var ta) && consts.TryGetValue(b, out var tb)
                && GetTensor(ta).ToArray<float>().SequenceEqual(GetTensor(tb).ToArray<float>()));

        float[] Floats(string name) => GetTensor(consts[name]).ToArray<float>();

        foreach (var n in _graph.Node)
        {
            switch (n.OpType)
            {
                case "QuantizeLinear":
                    nodes.Add(n);
                    quantized[n.Output[0]] = (n.Input[1], n.Input[2]);
                    break;

                case "QLinearConv":
                    {
                        // x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [B]
                        var x = Dequantize(n.Input[0], n.Input[1], n.Input[2]);
                        var wq = consts[n.Input[3]];
                        var wScale = Floats(n.Input[4]);
                        var wZp = GetTensor(consts[n.Input[5]]).ToArray<int>();
                        var oc = wq.Dims[0];
                        var inner = wq.Dims.Skip(1).Aggregate(1L, (a, b) => a * b);
                        var wValues = GetTensor(wq).ToArray<int>();
                        var wFloat = new float[wValues.Length];
                        for (long i = 0; i < wValues.Length; i++)
                        {
                            var c = wScale.Length == 1 ? 0 : (int)(i / inner);
                            wFloat[i] = (wValues[i] - wZp[wZp.Length == 1 ? 0 : c]) * wScale[c];
                        }

                        // the weights get their own Q/DQ pair, folded at import, so the scheme carries the exported params
                        var w = FloatName(n.Input[3]);
                        _graph.Initializer.Add(MakeFloatTensor(w, wq.Dims, wFloat));
                        var axis = wScale.Length == 1 ? null : new[] { new AttributeProto { Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = 0 } };
                        var wqName = NewName(w + QuantizedSuffix);
                        var wdqName = NewName(w + "_dequantized");
                        Emit("QuantizeLinear", new[] { w, n.Input[4], n.Input[5] }, wqName, attributes: axis);
                        Emit("DequantizeLinear", new[] { wqName, n.Input[4], n.Input[5] }, wdqName, attributes: axis);

                        var inputs = new List<string> { x, wdqName };
                        if (n.Input.Count > 8 && n.Input[8] != string.Empty)
                        {
                            // int32 bias with scale x_scale * w_scale and zero point 0
                            var xScale = Floats(n.Input[1])[0];
                            var bias = GetTensor(consts[n.Input[8]]).ToArray<int>();
                            var b = FloatName(n.Input[8]);
                            _graph.Initializer.Add(MakeFloatTensor(b, new[] { oc }, bias.Select((v, c) => v * (xScale * wScale[wScale.Length == 1 ? 0 : c])).ToArray()));
                            inputs.Add(b);
                        }

                        var y = FloatName(n.Output[0]);
                        Emit("Conv", inputs, y, attributes: n.Attribute, name: n.Name);
                        Quantize(y, n.Output[0], n.Input[6], n.Input[7]);
                        break;
                    }

                case "QLinearLeakyRelu":
                    {
                        // x, x_scale, x_zp, y_scale, y_zp
                        var y = FloatName(n.Output[0]);
                        Emit("LeakyRelu", new[] { Dequantize(n.Input[0], n.Input[1], n.Input[2]) }, y, attributes: n.Attribute, name: n.Name);
                        Quantize(y, n.Output[0], n.Input[3], n.Input[4]);
                        break;
                    }

                case "QLinearAdd":
                    {
                        // a, a_scale, a_zp, b, b_scale, b_zp, c_scale, c_zp
                        var a = Dequantize(n.Input[0], n.Input[1], n.Input[2]);
                        var b = Dequantize(n.Input[3], n.Input[4], n.Input[5]);
                        var y = FloatName(n.Output[0]);
                        Emit("Add", new[] { a, b }, y, name: n.Name);
                        Quantize(y, n.Output[0], n.Input[6], n.Input[7]);
                        break;
                    }

                case "QLinearConcat":
                    {
                        // y_scale, y_zp, then (x, x_scale, x_zp) per input
                        var inputs = Enumerable.Range(0, (n.Input.Count - 2) / 3)
                            .Select(i => Dequantize(n.Input[2 + (3 * i)], n.Input[3 + (3 * i)], n.Input[4 + (3 * i)])).ToArray();
                        var y = FloatName(n.Output[0]);
                        Emit("Concat", inputs, y, attributes: n.Attribute, name: n.Name);
                        Quantize(y, n.Output[0], n.Input[0], n.Input[1]);
                        break;
                    }

                default:
                    if (n.Domain is not ("" or "ai.onnx") && n.OpType.StartsWith("QLinear", StringComparison.Ordinal))
                    {
                        throw new NotSupportedException($"{n.Domain}.{n.OpType} {n.Name} is not supported.");
                    }

                    if (n.Input.Count > 0 && quantized.TryGetValue(n.Input[0], out var qp) && n.OpType != "DequantizeLinear")
                    {
                        // an op that runs on the quantized values directly: it only moves values, so its output keeps
                        // the input's params
                        if (!_rangePreservingOps.Contains(n.OpType) || n.Output.Count != 1)
                        {
                            throw new NotSupportedException($"{n.OpType} {n.Name} on the quantized tensor {n.Input[0]} is not supported.");
                        }

                        var y = FloatName(n.Output[0]);
                        var op = n.Clone();
                        op.Input[0] = Dequantize(n.Input[0], qp.Scale, qp.ZeroPoint);
                        op.Output[0] = y;
                        nodes.Add(op);
                        Quantize(y, n.Output[0], qp.Scale, qp.ZeroPoint);
                    }
                    else
                    {
                        nodes.Add(n);
                    }

                    break;
            }
        }

        _graph.Node.Clear();
        _graph.Node.Add(nodes);
    }
}
