// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Imaging;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Quantization;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Derives a quant scheme from the Quantize/Dequantize pairs of the imported model (same rules as
/// models/model_runner/util/quant_scheme.py), so a QDQ model can be compiled with UsePTQ without a .qs.json.
/// It runs right after import, before anything folds constants or moves the pairs:
/// <list type="bullet">
/// <item>Dequantize(Quantize(x)): the range of the grid, on x (the pair stays, the target removes it).</item>
/// <item>A constant folded by the importer from a Q/DQ pair (<see cref="IRMetadata.FakeQuantGrid"/>) or a
/// Dequantize of an integer constant (folded here): the range of the grid, per channel when the grid is.</item>
/// <item>Conv weights without a grid: per output channel min/max. Graph inputs without a pair: --input-range.</item>
/// <item>Ops that only move values (Reshape, Transpose, MaxPool, ...): the range of their input.</item>
/// </list>
/// The scheme refers to tensors by name, so an unnamed x gets the name of its pair.
/// The result is <see cref="QuantizeOptions.ImportedQuantScheme"/>; a .qs.json given by the user takes precedence.
/// </summary>
public sealed class QdqQuantSchemePass : ModulePass
{
    // nncase widens ranges narrower than this (QuantUtility.FixupRange)
    private const float MinRangeWidth = 0.01f;

    private readonly Dictionary<Expr, ValueRange<float>> _ranges = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<string, Output> _entries = new();
    private readonly List<Output> _order = new();
    private readonly List<string> _widened = new();
    private readonly List<string> _mismatched = new();

    private static string? NameOf(Expr expr) => expr.Metadata.OutputNames?[0] ?? (expr as Var)?.Name;

    // the zero point offset that moves an 8-bit grid onto uint8: int8 with zero point zp is uint8 with zp + 128
    private static int ZeroPointOffset(DataType type, string name) =>
        type == DataTypes.UInt8 ? 0
        : type == DataTypes.Int8 ? 128
        : throw new NotSupportedException($"Q/DQ on {name}: quant type {type} is not supported, only 8-bit quantization can be compiled from a QDQ model.");

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

    private static QuantParam ParamOf(Call call, ParameterInfo parameter) =>
        ((TensorConst)call[parameter]).Value.ToScalar<QuantParam>();

    // the weights of a conv as a constant, looking through the layout Transpose the importer puts on ConvTranspose
    // weights: the constant, and the axis of the output channels in it
    private static (TensorConst Weights, int OcAxis)? ConstWeights(Call conv) => conv.Target switch
    {
        Conv2D when conv[Conv2D.Weights] is TensorConst w => (w, 0),
        Conv2DTranspose when conv[Conv2DTranspose.Weights] is TensorConst w => (w, 0),
        Conv2DTranspose when conv[Conv2DTranspose.Weights] is Call { Target: IR.Tensors.Transpose } t && t[IR.Tensors.Transpose.Input] is TensorConst w => (w, 1),
        _ => null,
    };

    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule module, RunPassContext options)
    {
        var quantizeOptions = CompileSession.CompileOptions.QuantizeOptions;
        if (quantizeOptions.IgnoreQdqRanges || module.Entry is not Function function)
        {
            return Task.FromResult(module);
        }

        var exprs = ExprCollector.Collect(function);
        var found = AddGridConstants(exprs);
        found |= FoldDequantizedConstants(exprs);
        found |= AddActivationPairs(exprs);
        if (!found)
        {
            return Task.FromResult(module);
        }

        // the folds replaced some calls: collect again
        exprs = ExprCollector.Collect(function);
        AddBakedConvWeights(exprs);
        AddInputs(function);
        PropagateRanges(exprs);

        var scheme = new QuantScheme { Version = "1.0", Model = null, Outputs = _order.ToArray() };
        Report(scheme, FindMissing(function, exprs));
        quantizeOptions.ImportedQuantScheme = scheme;
        return Task.FromResult(module);
    }

    private void Add(string name, string mode, IEnumerable<ValueRange<float>> ranges)
    {
        if (!_entries.ContainsKey(name))
        {
            var entry = new Output { Name = name, DataType = "u8", DataRange = ranges.ToArray(), DataRangeMode = mode };
            _entries[name] = entry;
            _order.Add(entry);
        }
    }

    /// <summary>
    /// The range of an exported (scale, zp). A range narrower than nncase's minimum width is widened by increasing the
    /// scale and keeping the zero point, so nncase's own fixup leaves it alone and real 0 keeps its code.
    /// The range is then replayed through nncase's range -> param computation and any difference is reported.
    /// </summary>
    private ValueRange<float> GetCheckedRange(string name, float scale, int zp)
    {
        var range = QdqRange(scale, zp);
        if (range.Max - range.Min < MinRangeWidth)
        {
            var newScale = MinRangeWidth / byte.MaxValue;
            while (QdqRange(newScale, zp) is var r && r.Max - r.Min < MinRangeWidth)
            {
                newScale = MathF.BitIncrement(newScale);
            }

            _widened.Add(FormattableString.Invariant($"{name}: scale {scale:G9} -> {newScale:G9}, zero point {zp}"));
            scale = newScale;
            range = QdqRange(scale, zp);
        }

        var param = QuantUtility.GetQuantParam(QuantizerFixUpRange(range), 8, QuantMode.UnsignedMode);
        if (param.ZeroPoint != zp || Math.Abs(param.Scale - scale) > 1e-5 * scale)
        {
            _mismatched.Add(FormattableString.Invariant($"{name}: exported (scale {scale:G9}, zp {zp}), nncase (scale {param.Scale:G9}, zp {param.ZeroPoint})"));
        }

        return range;
    }

    // a per-tensor weight range is "by_tensor": the quantizer repeats it for every output channel
    private void AddWeights(string name, FakeQuantGrid grid)
    {
        var offset = ZeroPointOffset(grid.QuantType, name);
        var ranges = grid.Scales.Select((s, c) => GetCheckedRange(grid.Scales.Length == 1 ? name : $"{name}[{c}]", s, grid.ZeroPoints[grid.ZeroPoints.Length == 1 ? 0 : c] + offset)).ToArray();
        Add(name, ranges.Length > 1 ? "by_channel" : "by_tensor", ranges);
    }

    /// <summary>
    /// Constants the importer folded from a Q/DQ pair.
    /// </summary>
    private bool AddGridConstants(IReadOnlyList<Expr> exprs)
    {
        var found = false;
        foreach (var c in exprs.OfType<TensorConst>())
        {
            if (c.Metadata.FakeQuantGrid is FakeQuantGrid grid && NameOf(c) is string name)
            {
                AddWeights(name, grid);
                found = true;
            }
        }

        return found;
    }

    /// <summary>
    /// Dequantize of an integer constant (e.g. QLinearConv weights): folded into float constants on the same grid.
    /// </summary>
    private bool FoldDequantizedConstants(IReadOnlyList<Expr> exprs)
    {
        var found = false;
        foreach (var dq in exprs.OfType<Call>().Where(c => c is { Target: Dequantize } && c[Dequantize.Input] is TensorConst).ToArray())
        {
            var input = (TensorConst)dq[Dequantize.Input];
            var param = ParamOf(dq, Dequantize.DequantParam);
            var name = NameOf(dq) ?? NameOf(input) ?? throw new InvalidOperationException("Dequantize of an unnamed constant.");
            var values = input.Value.ToArray<int>().Select(q => (q - param.ZeroPoint) * param.Scale).ToArray();
            var folded = Const.FromTensor(Tensor.From(values, input.Value.Dimensions));
            folded.Metadata.OutputNames = new[] { name };
            var grid = new FakeQuantGrid(input.Value.ElementType, new[] { param.Scale }, new[] { param.ZeroPoint }, 0);

            // an int32 bias is not quantized by the scheme, it only has to become float
            if (input.Value.ElementType == DataTypes.UInt8 || input.Value.ElementType == DataTypes.Int8)
            {
                folded.Metadata.FakeQuantGrid = grid;
                AddWeights(name, grid);
                found = true;
            }

            ReplaceUtility.ReplaceAllUsesWith(dq, folded);
        }

        return found;
    }

    /// <summary>
    /// Dequantize(Quantize(x)) with the same params: the range of the grid, on x.
    /// </summary>
    private bool AddActivationPairs(IReadOnlyList<Expr> exprs)
    {
        var found = false;
        foreach (var dq in exprs.OfType<Call>().Where(c => c is { Target: Dequantize }))
        {
            if (dq[Dequantize.Input] is not Call { Target: Quantize quantize } q)
            {
                continue;
            }

            var qParam = ParamOf(q, Quantize.QuantParam);
            var dqParam = ParamOf(dq, Dequantize.DequantParam);
            if (!Passes.Rules.Neutral.FoldQuantDeQuant.AlmostEqual(qParam, dqParam))
            {
                continue;
            }

            var x = q[Quantize.Input];
            var name = NameOf(x) ?? NameOf(dq) ?? NameOf(q);
            if (name is null)
            {
                continue;
            }

            // the pair disappears in the target (FoldQuantDeQuant) and x takes its place: x carries the name
            if (NameOf(x) is null)
            {
                x.Metadata.OutputNames = new[] { name };
            }

            var range = GetCheckedRange(name, qParam.Scale, qParam.ZeroPoint + ZeroPointOffset(quantize.TargetType, name));
            _ranges[x] = range;
            _ranges[dq] = range;
            Add(name, "by_tensor", new[] { range });
            found = true;
        }

        return found;
    }

    /// <summary>
    /// Conv weights that are plain constants (weights baked by the QAT export): per output channel min/max.
    /// </summary>
    private void AddBakedConvWeights(IReadOnlyList<Expr> exprs)
    {
        foreach (var conv in exprs.OfType<Call>())
        {
            if (ConstWeights(conv) is not var (w, ocAxis) || w.Metadata.FakeQuantGrid != null || NameOf(w) is not string name
                || _entries.ContainsKey(name) || w.Value.ElementType != DataTypes.Float32)
            {
                continue;
            }

            var dims = w.Value.Dimensions.ToArray();
            var values = w.Value.ToArray<float>();
            var oc = dims[ocAxis];
            var ranges = Enumerable.Range(0, oc).Select(_ => new ValueRange<float>(float.MaxValue, float.MinValue)).ToArray();
            var inner = dims.Skip(ocAxis + 1).Aggregate(1L, (a, b) => a * b);
            for (long i = 0; i < values.Length; i++)
            {
                var c = (int)(i / inner % oc);
                ranges[c] = new(Math.Min(ranges[c].Min, values[i]), Math.Max(ranges[c].Max, values[i]));
            }

            Add(name, "by_channel", ranges.Select(QuantizerFixUpRange));
        }
    }

    /// <summary>
    /// Graph inputs without a Q/DQ pair get --input-range.
    /// </summary>
    private void AddInputs(Function function)
    {
        var inputRange = CompileSession.CompileOptions.InputRange;
        if (inputRange.Length != 2)
        {
            return;
        }

        foreach (var input in function.Parameters.ToArray())
        {
            if (!_ranges.ContainsKey(input) && NameOf(input) is string name)
            {
                var range = new ValueRange<float>(inputRange[0], inputRange[1]);
                _ranges[input] = range;
                Add(name, "by_tensor", new[] { range });
            }
        }
    }

    /// <summary>
    /// Ops whose output values are copies or rearrangements of their input values keep the input's range.
    /// </summary>
    private void PropagateRanges(IReadOnlyList<Expr> exprs)
    {
        foreach (var call in exprs.OfType<Call>())
        {
            if (_ranges.ContainsKey(call))
            {
                continue;
            }

            ValueRange<float>? range = call.Target switch
            {
                IR.Tensors.Reshape or IR.Tensors.Transpose or Slice or Squeeze or Unsqueeze or Flatten or Gather
                    or Broadcast or Expand or Tile or Pad or ResizeImage or ReduceWindow2D { ReduceOp: ReduceOp.Max }
                    when _ranges.TryGetValue(call.Arguments[0], out var r) => r,
                Concat when call[Concat.Input] is IR.Tuple t && t.Fields.ToArray() is var fields && fields.Length != 0
                    && fields.All(f => _ranges.TryGetValue(f, out var fr) && fr == _ranges[fields[0]]) => _ranges[fields[0]],
                _ => null,
            };

            if (range is ValueRange<float> value)
            {
                _ranges[call] = value;
                if (NameOf(call) is string name)
                {
                    Add(name, "by_tensor", new[] { value });
                }
            }
        }
    }

    /// <summary>
    /// Tensors that the hardware has to quantize but that have no range (they would silently stay float16).
    /// </summary>
    private List<string> FindMissing(Function function, IReadOnlyList<Expr> exprs)
    {
        var missing = new List<string>();
        void Check(Expr e, string reason)
        {
            if (e is Const || _ranges.ContainsKey(e))
            {
                return;
            }

            var name = NameOf(e) ?? (e is Call c ? $"<unnamed {c.Target.GetType().Name}>" : $"<unnamed {e.GetType().Name}>");
            if (!missing.Any(m => m.StartsWith(name + ":", StringComparison.Ordinal)))
            {
                missing.Add($"{name}: {reason}");
            }
        }

        foreach (var input in function.Parameters.ToArray())
        {
            Check(input, "graph input (no Q/DQ and no --input-range)");
        }

        foreach (var call in exprs.OfType<Call>())
        {
            var name = NameOf(call) ?? string.Empty;
            switch (call.Target)
            {
                case Conv2D or Conv2DTranspose or IR.Math.MatMul:
                    Check(call.Arguments[0], $"data input of {call.Target.GetType().Name} {name}");
                    break;
                case Binary { BinaryOp: BinaryOp.Add or BinaryOp.Sub or BinaryOp.Mul }:
                case Concat:
                    var inputs = (call.Target is Concat && call[Concat.Input] is IR.Tuple t ? t.Fields.ToArray() : call.Arguments.ToArray())
                        .Where(e => e is not Const).ToArray();
                    if (inputs.Any(_ranges.ContainsKey))
                    {
                        foreach (var e in inputs)
                        {
                            Check(e, $"input of {call.Target.GetType().Name} {name} whose other inputs are quantized");
                        }
                    }

                    break;
            }
        }

        return missing;
    }

    private void Report(QuantScheme scheme, List<string> missing)
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
        Warn("Q/DQ ranges narrower than nncase's minimum were widened", _widened);
        Warn("Q/DQ params that nncase will not reproduce exactly", _mismatched);
        Warn("tensors that should have a Q/DQ pair but have none (they will stay float16)", missing);

        if (DumpScope.Current.IsEnabled(DumpFlags.ImportOps))
        {
            using (var stream = DumpScope.Current.OpenFile("ImportedQuantScheme.json"))
            {
                JsonSerializer.Serialize(stream, scheme, new JsonSerializerOptions { WriteIndented = true });
            }

            using var writer = new StreamWriter(DumpScope.Current.OpenFile("ImportedQuantSchemeReport.txt"));
            foreach (var (title, items) in new[] { ("widened", _widened), ("mismatched", _mismatched), ("missing", missing) })
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
}
