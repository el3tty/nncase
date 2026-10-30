// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Onnx;
using static Nncase.IR.F.Math;

namespace Nncase.Importer;

/// <summary>
/// onnxruntime QOperator ops (QLinearConv, com.microsoft QLinearLeakyRelu/QLinearAdd/QLinearConcat) and ops that run on
/// their quantized outputs (MaxPool, Resize, ...). Each becomes Dequantize of its inputs -> the float op, built by the
/// float visitor -> Quantize of its output; the quant scheme pass (QdqQuantSchemePass) takes the ranges from the pairs
/// and folds the dequantized weights, as for a QAT export.
/// </summary>
public sealed partial class OnnxImporter
{
    private const string QuantizedSuffix = "_quantized";

    // ops that only move values: run on a quantized tensor, their output keeps its params
    private static readonly HashSet<string> _rangePreservingOps = new()
    {
        "Reshape", "Transpose", "Slice", "Squeeze", "Unsqueeze", "Flatten", "Identity", "MaxPool", "Resize",
        "Upsample", "Pad", "Gather", "Expand", "Tile",
    };

    // quantized tensor (a QuantizeLinear or QOperator output) -> its params
    private readonly Dictionary<string, (QuantParam Param, DataType Type)> _quantizedTensors = new();

    private HashSet<string>? _graphTensorNames;

    // (quantized tensor, params) -> its dequantized copy, shared by the ops that read it
    private readonly Dictionary<(string Tensor, QuantParam Param), string> _dequantizedTensors = new();

    private bool IsOnQuantizedTensor(NodeProto op) =>
        op.Input.Count > 0 && _quantizedTensors.ContainsKey(op.Input[0])
        && !op.OpType.StartsWith("QLinear", StringComparison.Ordinal) && op.OpType is not ("QuantizeLinear" or "DequantizeLinear");

    private Expr VisitQLinearConv(in NodeProto op)
    {
        // x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [B]
        var x = GetQuantParam(op, 1, 2);
        var wScale = ((TensorConst)GetInputExpr(op, 4)).Value.ToArray<float>();
        var wZeroPoint = ((TensorConst)GetInputExpr(op, 5)).Value.ToArray<int>();
        var wq = (TensorConst)GetInputExpr(op, 3);
        var oc = wq.Value.Dimensions[0];
        var weights = FloatName(op.Input[3]);
        if (wScale.Length == 1)
        {
            // the int8 weights stay a Dequantize of the constant, which the quant scheme pass folds and takes the
            // weight range from
            var dq = Dequantize(wq, new QuantParam(wZeroPoint[0], wScale[0]), DataTypes.Float32);
            dq.InferenceType();
            AddTensor(weights, dq);
        }
        else
        {
            // per-channel weights: the IR Quantize/Dequantize take one scale and zero point, so they become baked float
            // weights (exactly the dequantized values), and nncase quantizes them again per output channel from their
            // min/max, as the baked weights of a QAT export
            var codes = wq.Value.ToArray<int>();
            var inner = codes.Length / oc;
            var values = codes.Select((q, i) => (q - wZeroPoint[wZeroPoint.Length == 1 ? 0 : i / inner]) * wScale[i / inner]).ToArray();
            AddTensor(weights, Const.FromTensor(Tensor.From(values, wq.Value.Dimensions)));
        }

        var inputs = new List<string> { Dequantized(op.Input[0], x.Param), weights };
        if (op.Input.Count > 8 && op.Input[8] != string.Empty)
        {
            // int32 bias with scale x_scale * w_scale (per channel when the weights are) and zero point 0
            var bias = ((TensorConst)GetInputExpr(op, 8)).Value.ToArray<int>();
            var b = FloatName(op.Input[8]);
            AddTensor(b, Const.FromTensor(Tensor.From(bias.Select((v, c) => v * (x.Param.Scale * wScale[wScale.Length == 1 ? 0 : c])).ToArray(), new[] { oc })));
            inputs.Add(b);
        }

        return VisitAsFloat(op, "Conv", inputs, op.Attribute, GetQuantParam(op, 6, 7));
    }

    private Expr VisitQLinearLeakyRelu(in NodeProto op)
    {
        // x, x_scale, x_zp, y_scale, y_zp
        var x = GetQuantParam(op, 1, 2);
        return VisitAsFloat(op, "LeakyRelu", new[] { Dequantized(op.Input[0], x.Param) }, op.Attribute, GetQuantParam(op, 3, 4));
    }

    private Expr VisitQLinearAdd(in NodeProto op)
    {
        // a, a_scale, a_zp, b, b_scale, b_zp, c_scale, c_zp
        var a = Dequantized(op.Input[0], GetQuantParam(op, 1, 2).Param);
        var b = Dequantized(op.Input[3], GetQuantParam(op, 4, 5).Param);
        return VisitAsFloat(op, "Add", new[] { a, b }, op.Attribute, GetQuantParam(op, 6, 7));
    }

    private Expr VisitQLinearConcat(in NodeProto op)
    {
        // y_scale, y_zp, then (x, x_scale, x_zp) per input
        var node = op;
        var inputs = Enumerable.Range(0, (op.Input.Count - 2) / 3)
            .Select(i => Dequantized(node.Input[2 + (3 * i)], GetQuantParam(node, 3 + (3 * i), 4 + (3 * i)).Param)).ToArray();
        return VisitAsFloat(op, "Concat", inputs, op.Attribute, GetQuantParam(op, 0, 1));
    }

    /// <summary>
    /// An ordinary op whose data input is quantized (e.g. MaxPool or Resize of a QOperator output): it runs on the
    /// dequantized values and, as it only moves them, its output keeps the input's params.
    /// </summary>
    private Expr VisitOnQuantizedTensor(in NodeProto op)
    {
        if (!_rangePreservingOps.Contains(op.OpType) || op.Output.Count != 1)
        {
            throw new NotSupportedException($"{op.OpType} {op.Name} on the quantized tensor {op.Input[0]} is not supported.");
        }

        var input = _quantizedTensors[op.Input[0]];
        var inputs = op.Input.Select((name, i) => i == 0 ? Dequantized(name, input.Param) : name).ToArray();
        return VisitAsFloat(op, op.OpType, inputs, op.Attribute, input, op.Domain);
    }

    /// <summary>
    /// Builds the float op with the float visitor and quantizes its output: the output of op is the quantized tensor,
    /// the float result is named after it without the "_quantized" suffix.
    /// </summary>
    private Expr VisitAsFloat(NodeProto op, string opType, IEnumerable<string> inputs, IEnumerable<AttributeProto> attributes, (QuantParam Param, DataType Type) output, string domain = "")
    {
        var floatOp = new NodeProto { OpType = opType, Domain = domain, Name = op.Name, Input = { inputs }, Output = { FloatName(op.Output[0]) } };
        floatOp.Attribute.Add(attributes);
        Visit(floatOp);
        _quantizedTensors[op.Output[0]] = output;
        return Quantize(_outputTensors![floatOp.Output[0]], output.Param, output.Type);
    }

    private (QuantParam Param, DataType Type) GetQuantParam(NodeProto op, int scaleIndex, int zeroPointIndex)
    {
        var scale = (TensorConst)GetInputExpr(op, scaleIndex);
        var zeroPoint = (TensorConst)GetInputExpr(op, zeroPointIndex);
        if (scale.Value.Length != 1)
        {
            throw new NotSupportedException($"{op.OpType} {op.Name}: per-axis activation quantization is not supported.");
        }

        return (new QuantParam(zeroPoint.Value.ToScalar<int>(), scale.Value.ToScalar<float>()), zeroPoint.CheckedDataType);
    }

    private string Dequantized(string tensor, QuantParam param)
    {
        if (!_dequantizedTensors.TryGetValue((tensor, param), out var name))
        {
            name = NewTensorName(tensor + "_dequantized");
            AddTensor(name, Dequantize(_outputTensors![tensor], param, DataTypes.Float32));
            _dequantizedTensors[(tensor, param)] = name;
        }

        return name;
    }

    // the float tensor a quantized tensor X_quantized is computed from is named X, as in the source model
    private string FloatName(string quantized) =>
        NewTensorName(quantized.EndsWith(QuantizedSuffix, StringComparison.Ordinal) ? quantized[..^QuantizedSuffix.Length] : quantized + "_float");

    private string NewTensorName(string baseName)
    {
        // a name no tensor of the model has, including the outputs of nodes not visited yet
        _graphTensorNames ??= _graph.Node.SelectMany(n => n.Input.Concat(n.Output)).Concat(_graph.Input.Select(i => i.Name)).ToHashSet();
        var name = baseName;
        for (var i = 1; _outputTensors!.ContainsKey(name) || _constTensors!.ContainsKey(name) || _graphTensorNames.Contains(name); i++)
        {
            name = $"{baseName}_{i}";
        }

        return name;
    }

    private void AddTensor(string name, Expr expr)
    {
        expr.Metadata.OutputNames = new[] { name };
        _outputTensors![name] = expr;
    }
}
