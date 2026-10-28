// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Quantization;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;

namespace Nncase.Tests.ImporterTest;

/// <summary>
/// Quant scheme inference from QuantizeLinear/DequantizeLinear pairs (src/Nncase.Importer/Onnx/QdqScheme.cs).
/// The models are written in the ONNX textual syntax and parsed by onnx's own parser (through OrtKI).
/// </summary>
[AutoSetupTestMethod(InitSession = true)]
public class UnitTestQdqScheme : TestClassBase
{
    private const string Header = "<ir_version: 8, opset_import: [\"\" : 13]>\n";

    // x -> Relu -> r -> Q/DQ(0.05, 128) -> Conv(w, baked weights)
    private const string ReluQdqConv = Header + """
        g (float[1,3,4,4] x) => (float[1,2,4,4] y)
        <float s = {0.05}, uint8 zp = {128}, float[2,3,1,1] w = {0.5, -0.25, 0.1, 0.2, 0.3, -0.4}>
        {
            r = Relu(x)
            rq = QuantizeLinear(r, s, zp)
            rdq = DequantizeLinear(rq, s, zp)
            y = Conv(rdq, w)
        }
        """;

    // x -> Conv -> c -> Q/DQ, the graph input has no pair
    private const string ConvQdq = Header + """
        g (float[1,2,4,4] x) => (float[1,2,4,4] y)
        <float s = {0.05}, uint8 zp = {128}, float[2,2,1,1] w = {1, 0, 0, 1}>
        {
            c = Conv(x, w)
            cq = QuantizeLinear(c, s, zp)
            y = DequantizeLinear(cq, s, zp)
        }
        """;

    public UnitTestQdqScheme()
    {
        // the scheme is only inferred for PTQ; the report is written to the Import dump directory
        CompileOptions.QuantizeOptions.ModelQuantMode = ModelQuantMode.UsePTQ;
        CompileOptions.DumpFlags = DumpFlags.ImportOps;
    }

    [Fact]
    public void TestActivationPair()
    {
        var module = Import(ReluQdqConv);

        var r = Entry("r");
        Assert.Equal(("u8", "by_tensor"), (r.DataType, r.DataRangeMode));
        AssertRanges(r, (-128 * 0.05f, 127 * 0.05f));

        // the Q/DQ ops stay in the IR, the target removes them
        Assert.Equal(1, Count<Quantize>(module));
        Assert.Equal(1, Count<Dequantize>(module));
    }

    [Fact]
    public void TestBakedConvWeights()
    {
        Import(ReluQdqConv);

        // per output channel min/max, widened to contain 0
        var w = Entry("w");
        Assert.Equal(("u8", "by_channel"), (w.DataType, w.DataRangeMode));
        AssertRanges(w, (-0.25f, 0.5f), (-0.4f, 0.3f));
    }

    [Fact]
    public void TestBakedConvTransposeWeights()
    {
        // ConvTranspose weights are [ic, oc, kh, kw]: the ranges are per output channel (axis 1)
        Import(Header + """
            g (float[1,2,4,4] x) => (float[1,3,4,4] y)
            <float s = {0.05}, uint8 zp = {128}, float[2,3,1,1] w = {1, 2, 3, -1, 5, 0.5}>
            {
                xq = QuantizeLinear(x, s, zp)
                xdq = DequantizeLinear(xq, s, zp)
                y = ConvTranspose(xdq, w)
            }
            """);

        AssertRanges(Entry("w"), (-1f, 1f), (0f, 5f), (0f, 3f));
    }

    [Fact]
    public void TestWeightPairIsFolded()
    {
        // per-channel weight Q/DQ (axis 0) is folded at import with the ONNX semantics:
        // ch0 (s 0.5, zp 128): 0.25 -> 0.5 rounds to even 0 -> 0; 0.75 -> 1.5 rounds to 2 -> 1
        // ch1 (s 0.25, zp 100): 100 -> 400 clamps to 255 -> 38.75; -30 -> -120 clamps to 0 -> -25
        var module = Import(Header + """
            g (float[1,2,4,4] x) => (float[1,2,4,4] y)
            <float s = {0.05}, uint8 zp = {128}, float[2,2,1,1] w = {0.25, 0.75, 100, -30},
             float[2] ws = {0.5, 0.25}, uint8[2] wzp = {128, 100}>
            {
                xq = QuantizeLinear(x, s, zp)
                xdq = DequantizeLinear(xq, s, zp)
                wq = QuantizeLinear <axis = 0> (w, ws, wzp)
                wdq = DequantizeLinear <axis = 0> (wq, ws, wzp)
                y = Conv(xdq, wdq)
            }
            """);

        var conv = Calls<Conv2D>(module).Single();
        var weights = (TensorConst)conv[Conv2D.Weights];
        Assert.Equal(new[] { 0f, 1f, 38.75f, -25f }, weights.Value.ToArray<float>());
        Assert.Equal("w", weights.Metadata.OutputNames?[0]);

        // only the activation pair is left
        Assert.Equal(1, Count<Quantize>(module));

        // the scheme ranges come from the exported params, not from the folded values
        var w = Entry("w");
        Assert.Equal("by_channel", w.DataRangeMode);
        AssertRanges(w, (-128 * 0.5f, 127 * 0.5f), (-100 * 0.25f, 155 * 0.25f));
    }

    [Fact]
    public void TestRangePropagation()
    {
        Import(Header + """
            g (float[1,2,4,4] x) => (float[1,8] rs, float[1,4,4,4] c1, float[1,4,4,4] c2)
            <float s1 = {0.05}, float s2 = {0.1}, uint8 zp = {0}, int64[2] shp = {1, -1}>
            {
                r = Relu(x)
                rq = QuantizeLinear(r, s1, zp)
                rdq = DequantizeLinear(rq, s1, zp)
                g = Sigmoid(x)
                gq = QuantizeLinear(g, s1, zp)
                gdq = DequantizeLinear(gq, s1, zp)
                t = Abs(x)
                tq = QuantizeLinear(t, s2, zp)
                tdq = DequantizeLinear(tq, s2, zp)
                mp = MaxPool <kernel_shape = [2, 2], strides = [2, 2]> (rdq)
                rs = Reshape(mp, shp)
                c1 = Concat <axis = 1> (rdq, gdq)
                c2 = Concat <axis = 1> (rdq, tdq)
            }
            """);

        var range = (0f, 255 * 0.05f);
        AssertRanges(Entry("mp"), range);
        AssertRanges(Entry("rs"), range);
        AssertRanges(Entry("c1"), range);

        // Concat of different ranges gets no range
        Assert.Null(TryEntry("c2"));
    }

    [Fact]
    public void TestInputRange()
    {
        CompileOptions.InputRange = new[] { 0f, 255f };
        Import(ConvQdq);

        var x = Entry("x");
        Assert.Equal("by_tensor", x.DataRangeMode);
        AssertRanges(x, (0f, 255f));
        Assert.Contains("# missing: 0", Report(), StringComparison.Ordinal);
    }

    [Fact]
    public void TestMissingInputRange()
    {
        Import(ConvQdq);

        Assert.Null(TryEntry("x"));
        Assert.Contains("x: graph input", Report(), StringComparison.Ordinal);
    }

    [Fact]
    public void TestMissingQdq()
    {
        // c1 feeds a Conv without a Q/DQ pair; the Conv output in front of SiLU (c0) is not quantized on purpose
        Import(Header + """
            g (float[1,2,4,4] x) => (float[1,2,4,4] y)
            <float s = {0.05}, uint8 zp = {128}, float[2,2,1,1] w = {1, 0, 0, 1}>
            {
                xq = QuantizeLinear(x, s, zp)
                xdq = DequantizeLinear(xq, s, zp)
                c0 = Conv(xdq, w)
                sg = Sigmoid(c0)
                m = Mul(c0, sg)
                mq = QuantizeLinear(m, s, zp)
                mdq = DequantizeLinear(mq, s, zp)
                c1 = Conv(mdq, w)
                y = Conv(c1, w)
            }
            """);

        var missing = ReportSection("missing");
        Assert.Single(missing);
        Assert.StartsWith("c1: data input of Conv", missing[0], StringComparison.Ordinal);
    }

    [Fact]
    public void TestNarrowRangeIsWidened()
    {
        Import(ReluQdqConv.Replace("s = {0.05}", "s = {0.000001}", StringComparison.Ordinal));

        // the scale grows to the minimal range width, the zero point is kept
        var range = Entry("r").DataRange!.Single();
        Assert.True(range.Max - range.Min >= 0.01f);
        var param = QuantUtility.GetQuantParam(range, 8, QuantMode.UnsignedMode);
        Assert.Equal(128, param.ZeroPoint);
        Assert.Equal(range.Min, -128 * param.Scale, 6);
        Assert.Single(ReportSection("widened"));
        Assert.Empty(ReportSection("mismatched"));
    }

    [Fact]
    public void TestInt8IsRejected()
    {
        var e = Assert.Throws<NotSupportedException>(() => Import(ReluQdqConv.Replace("uint8 zp = {128}", "int8 zp = {0}", StringComparison.Ordinal)));
        Assert.Contains("only uint8", e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TestPerChannelActivationIsRejected()
    {
        Assert.Throws<NotSupportedException>(() => Import(Header + """
            g (float[1,2,4,4] x) => (float[1,2,4,4] y)
            <float[2] s = {0.05, 0.1}, uint8[2] zp = {128, 128}>
            {
                xq = QuantizeLinear <axis = 1> (x, s, zp)
                y = DequantizeLinear <axis = 1> (xq, s, zp)
            }
            """));
    }

    [Fact]
    public void TestMismatchedPairIsIgnored()
    {
        Import(Header + """
            g (float[1,3,4,4] x) => (float[1,2,4,4] y)
            <float s = {0.05}, float s2 = {0.1}, uint8 zp = {128}, float[2,3,1,1] w = {0.5, -0.25, 0.1, 0.2, 0.3, -0.4}>
            {
                xq = QuantizeLinear(x, s, zp)
                xdq = DequantizeLinear(xq, s, zp)
                r = Relu(xdq)
                rq = QuantizeLinear(r, s, zp)
                rdq = DequantizeLinear(rq, s2, zp)
                y = Conv(rdq, w)
            }
            """);

        Assert.NotNull(TryEntry("x"));
        Assert.Null(TryEntry("r"));
    }

    [Fact]
    public void TestNoSchemeWithoutPTQ()
    {
        CompileOptions.QuantizeOptions.ModelQuantMode = ModelQuantMode.NoQuant;
        var module = Import(ReluQdqConv);

        Assert.Null(CompileOptions.QuantizeOptions.ImportedQuantScheme);
        Assert.Equal(1, Count<Quantize>(module));
    }

    [Fact]
    public async Task TestRangesAreBoundAfterFold()
    {
        var module = Import(ReluQdqConv);

        var pmgr = CompileSession.CreatePassManager("Passes");
        pmgr.AddWithName<DataflowPass>("FoldQDQ").Configure(p => p.Add<FoldQuantDeQuant>());
        pmgr.AddWithName<DataflowPass>("AddRangeOfMarker").Configure(p => p.Add<AddRangeOfAndMarker>());
        pmgr.AddWithName<EGraphPassWithQuantize>("AssignRanges");
        module = await pmgr.RunAsync(module);

        Assert.Equal(0, Count<Quantize>(module));
        Assert.Equal(0, Count<Dequantize>(module));

        // the marker on the Conv input carries the exported range of r
        var conv = Calls<Conv2D>(module).Single();
        var marker = Assert.IsType<Marker>(conv[Conv2D.Input]);
        AssertClose(new[] { -128 * 0.05f, 127 * 0.05f }, ((TensorConst)marker.Attribute).Value.ToArray<float>());

        // and the weights marker the per channel ranges
        var weights = Assert.IsType<Marker>(conv[Conv2D.Weights]);
        AssertClose(new[] { -0.25f, 0.5f, -0.4f, 0.3f }, ((TensorConst)weights.Attribute).Value.ToArray<float>());
    }

    private static int Count<T>(IRModule module)
        where T : Op => Calls<T>(module).Count();

    private static System.Collections.Generic.IEnumerable<Call> Calls<T>(IRModule module)
        where T : Op => ExprCollector.Collect(module.Entry!).OfType<Call>().Where(c => c.Target is T);

    // Quantizer.FixUpRange recomputes Max as Min + (Max - Min), which is off by an ulp in float32
    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        foreach (var (e, a) in expected.Zip(actual))
        {
            Assert.Equal(e, a, 5);
        }
    }

    private static void AssertRanges(Output entry, params (float Min, float Max)[] expected)
    {
        Assert.Equal(expected.Length, entry.DataRange!.Length);
        foreach (var (actual, (min, max)) in entry.DataRange.Zip(expected))
        {
            Assert.Equal(min, actual.Min, 5);
            Assert.Equal(max, actual.Max, 5);
        }
    }

    private IRModule Import(string text)
    {
        using var stream = new MemoryStream(OrtKI.ParseOnnxText(text));
        return Importers.ImportOnnx(stream, CompileSession);
    }

    private Output? TryEntry(string name) =>
        CompileOptions.QuantizeOptions.ImportedQuantScheme!.Outputs!.SingleOrDefault(o => o.Name == name);

    private Output Entry(string name) => TryEntry(name) ?? throw new InvalidOperationException($"no scheme entry for {name}");

    private string Report() => File.ReadAllText(Path.Join(CompileOptions.DumpDir, "Import", "ImportedQuantSchemeReport.txt"));

    private string[] ReportSection(string title) =>
        Report().Split('#', StringSplitOptions.RemoveEmptyEntries)
            .Single(s => s.TrimStart().StartsWith(title + ":", StringComparison.Ordinal))
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Skip(1)
            .ToArray();
}
