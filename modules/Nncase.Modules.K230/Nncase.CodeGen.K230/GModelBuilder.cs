// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Runtime;
using Nncase.Simulator;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.CodeGen.K230;

public sealed class GModelBuilder : IRTModule
{
    private List<IRTFunction> _functions;

    private PrimFunction _entry;

    private Dictionary<Nncase.TIR.Buffer, (uint Start, uint SegOffset, uint Size)> _allocation =
        new Dictionary<Nncase.TIR.Buffer, (uint, uint, uint)>(ReferenceEqualityComparer.Instance);

    public ModuleType ModuleType => ModuleType.Create("k230");

    public byte[] Source
    {
        get
        {
            Serialize();
            return File.ReadAllBytes(BinFilePath);
        }
    }

    public bool IsSerialized { get; private set; }

    public IReadOnlyList<IRTFunction> Functions => _functions;

    public int StackSize { get; init; }

    public bool EmbeddedParams { get; init; }

    public bool OnlyInstructions { get; init; }

    public AssemblyParser AssemblyParser { get; init; }

    public string Backend { get; set; }

    public string BinFilePath { get; set; }

    private string DdrCtrlPath { get; set; }

    private string ArgFilePath { get; set; }

    private string DescFilePath { get; set; }

    private DdrRegionStartInfo StartInfo { get; init; }

    public ValueTask InitializeAsync()
    {
        throw new NotImplementedException();
    }

    public ValueTask UninitializeAsync()
    {
        throw new NotImplementedException();
    }

    public GModelBuilder(PrimFunction? entry = null, AssemblyParser? assemblyParser = null,
        bool embedded_params = false)
    {
        IsSerialized = false;
        StackSize = assemblyParser?.ParserStackSize() ?? 262144;
        OnlyInstructions = assemblyParser != null && assemblyParser.Params == null;
        AssemblyParser = assemblyParser;
        EmbeddedParams = embedded_params;
        _entry = ((object)entry == null) ? AssemblyParser.ParserFunction() : entry;
        BinFilePath = Nncase.CodeGen.CodeGenUtil.GetTempFileName(".bin");
        DdrCtrlPath = Nncase.CodeGen.CodeGenUtil.GetTempFileName(".pc_addr_ctrl");
        ArgFilePath = Nncase.CodeGen.CodeGenUtil.GetTempFileName(".invoke_command");
        DescFilePath = Nncase.CodeGen.CodeGenUtil.GetTempFileName(".desc");
        StartInfo = new DdrRegionStartInfo();
        _functions = new List<IRTFunction>();
        Backend = "cmodel";
    }

    public string Dump(string name, string dumpDirPath)
    {
        if (!Directory.Exists(dumpDirPath))
        {
            Directory.CreateDirectory(dumpDirPath);
        }

        if (!IsSerialized)
        {
            BinFilePath = Path.Join(dumpDirPath, name + Path.GetExtension(BinFilePath));
            DdrCtrlPath = Path.Join(dumpDirPath, name + Path.GetExtension(DdrCtrlPath));
            ArgFilePath = Path.Join(dumpDirPath, name + Path.GetExtension(ArgFilePath));
            DescFilePath = Path.Join(dumpDirPath, name + Path.GetExtension(DescFilePath));
            Serialize();
        }

        if (Path.GetDirectoryName(BinFilePath) != dumpDirPath)
        {
            Func<string, bool, string> func = (old_path, copy) =>
            {
                string dumpPath = Path.Join(dumpDirPath, name + Path.GetExtension(old_path));
                if (File.Exists(dumpPath))
                {
                    File.Delete(dumpPath);
                }

                if (copy)
                {
                    File.Copy(old_path, dumpPath);
                }

                return dumpPath;
            };
            BinFilePath = func(BinFilePath, arg2: true);
            if (!OnlyInstructions)
            {
                DdrCtrlPath = func(DdrCtrlPath, arg2: false);
                ArgFilePath = func(ArgFilePath, arg2: true);
                WriteInvokeArgs();
                WriteVdutInvokeArgs();
                if (!EmbeddedParams)
                {
                    DescFilePath = func(DescFilePath, arg2: true);
                }
            }
        }

        return BinFilePath;
    }

    private void WriteBin()
    {
        using BinaryWriter binaryWriter = new BinaryWriter(File.OpenWrite(BinFilePath));
        uint dataSize = 0u;
        StartInfo.Input = 0u;
        binaryWriter.Position(StartInfo.Input + _entry.Parameters.InputBufferOf().ToArray().Aggregate(
            0L,
            (long acc, Nncase.TIR.Buffer bf) => acc + ((TensorConst)bf.MemSpan.Size).Value.ToScalar<int>()));
        binaryWriter.AlignPosition(8L);
        StartInfo.Rdata = (uint)binaryWriter.Position();
        binaryWriter.Write(Array.Empty<byte>());
        binaryWriter.AlignPosition(8L);
        StartInfo.Text = (uint)binaryWriter.Position();
        WriteInstText(binaryWriter);
        StartInfo.Data = (uint)binaryWriter.Position();
        StartInfo.Output = StartInfo.Data + dataSize;
    }

    private void WriteInstText(BinaryWriter gw)
    {
        new InstSerializeVisitor(gw).Visit(_entry);
    }

    private void AllocateBuffer()
    {
        List<(uint, uint, uint)> list = new List<(uint, uint, uint)>();
        uint baseOffset = 0u;
        uint inputOffset = 0u;
        WhereEnumerator<Nncase.TIR.Buffer, FunctionWrapper<Nncase.TIR.Buffer, bool>> enumerator =
            _entry.Parameters.InputBufferOf().GetEnumerator();
        while (enumerator.MoveNext())
        {
            Nncase.TIR.Buffer current = enumerator.Current;
            list.Add((inputOffset - baseOffset + StartInfo.Input, inputOffset, (uint)current.Size()));
            _allocation.Add(current, list.Last());
            inputOffset += (uint)current.Size();
        }

        uint outputOffset = 0u;
        enumerator = _entry.Parameters.OutputOf().GetEnumerator();
        while (enumerator.MoveNext())
        {
            Nncase.TIR.Buffer current2 = enumerator.Current;
            list.Add((outputOffset - baseOffset + StartInfo.Output, outputOffset, (uint)current2.Size()));
            _allocation.Add(current2, list.Last());
            outputOffset += (uint)current2.Size();
        }

        list.Add((StartInfo.Rdata, 0u, 0u));
        list.Add((StartInfo.Data, 0u, 0u));
    }

    private void WriteGLBCtrl(Tensor[] inputs)
    {
        using FileStream stream = File.OpenWrite(DdrCtrlPath);
        using StreamWriter streamWriter = new StreamWriter(stream);
        Dictionary<Nncase.TIR.Buffer, int> dictionary =
            new Dictionary<Nncase.TIR.Buffer, int>(ReferenceEqualityComparer.Instance);
        foreach (var item in inputs.Zip(_entry.Parameters.ToArray()))
        {
            if (item.Second.Dimensions.Length == 0)
            {
                if (item.First.BytesBuffer.Length != 4)
                {
                    throw new InvalidDataException("Const Scalar Data Length Must = 4");
                }

                dictionary[item.Second] = BitConverter.ToInt32(item.First.BytesBuffer.ToArray(), 0);
            }
        }

        Dictionary<Nncase.TIR.Buffer, (uint, uint, uint)> dictionary2 =
            new Dictionary<Nncase.TIR.Buffer, (uint, uint, uint)>(_allocation, ReferenceEqualityComparer.Instance);
        foreach (Nncase.TIR.Buffer buffer in dictionary2.Keys.Where((Nncase.TIR.Buffer b) =>
                     b.MemSpan.Location == MemoryLocation.Output))
        {
            (uint, uint, uint) tuple = dictionary2[buffer];
            dictionary2[buffer] = (tuple.Item1 - StartInfo.Output, tuple.Item2, tuple.Item3);
        }

        List<int> source = CodeGenUtil.ToStackArgs(_entry.Parameters.ToArray(), dictionary2, dictionary);
        int stackAddress = StackSize - 16;
        foreach (string[] chunk in source.Select((int a) => $"{a:X8}").Chunk(4))
        {
            string line = string.Empty;
            for (int wordIndex = 0; wordIndex < 4; wordIndex++)
            {
                string word = (wordIndex < chunk.Length) ? chunk[wordIndex] : $"{0:X8}";
                line += word;
            }

            streamWriter.Write($"{stackAddress:X8} ");
            streamWriter.WriteLine(line);
            stackAddress -= 16;
        }

        streamWriter.WriteLine($"00000000 {0:X8}{StartInfo.Output:X8}{0:X8}{0:X8}");
        streamWriter.WriteLine($"00400100 {StartInfo.Text:X32}");
        streamWriter.WriteLine("00400120 00000001000000010000000000000000");
    }

    private void WriteDesc()
    {
        using StreamWriter streamWriter = new StreamWriter(File.OpenWrite(DescFilePath));
        streamWriter.WriteLine($"{_entry.Parameters.InputBufferOf().Count()} {_entry.Parameters.OutputOf().Count()}");
        WhereEnumerator<Nncase.TIR.Buffer, FunctionWrapper<Nncase.TIR.Buffer, bool>> enumerator =
            _entry.Parameters.InputBufferOf().GetEnumerator();
        while (enumerator.MoveNext())
        {
            Nncase.TIR.Buffer current = enumerator.Current;
            int[] inputShape = current.Shape();
            streamWriter.WriteLine(
                $"{current.ElemType.GetDisplayName()} ({inputShape[0]}, {inputShape[1]}, {inputShape[2]}, {inputShape[3]})");
        }

        enumerator = _entry.Parameters.OutputOf().GetEnumerator();
        while (enumerator.MoveNext())
        {
            Nncase.TIR.Buffer current2 = enumerator.Current;
            int[] outputShape = current2.Shape();
            streamWriter.WriteLine(
                $"{current2.ElemType.GetDisplayName()} ({outputShape[0]}, {outputShape[1]}, {outputShape[2]}, {outputShape[3]})");
        }

        enumerator = _entry.Parameters.InputBufferOf().GetEnumerator();
        while (enumerator.MoveNext())
        {
            Nncase.TIR.Buffer current3 = enumerator.Current;
            streamWriter.WriteLine(
                $"{_allocation[current3].Start} {_allocation[current3].SegOffset} {_allocation[current3].Size}");
        }

        enumerator = _entry.Parameters.OutputOf().GetEnumerator();
        while (enumerator.MoveNext())
        {
            Nncase.TIR.Buffer current4 = enumerator.Current;
            streamWriter.WriteLine(
                $"{_allocation[current4].Start} {_allocation[current4].SegOffset} {_allocation[current4].Size}");
        }

        streamWriter.WriteLine($"{StartInfo.Rdata} {0} {StartInfo.Data - StartInfo.Rdata}");
        streamWriter.WriteLine($"{StartInfo.Data} {0} {0}");
        streamWriter.WriteLine($"{StartInfo.Text} {0} {StartInfo.Data - StartInfo.Text}");
    }

    private void WriteInvokeArgs()
    {
        using var streamWriter = new StreamWriter(
            File.Open(ArgFilePath, FileMode.Create, FileAccess.Write, FileShare.Read));

        var outputs = _entry.Parameters.OutputOf().ToArray();

        streamWriter.WriteLine(BinFilePath);
        streamWriter.WriteLine(StartInfo.Text);

        streamWriter.WriteLine(JoinOrZero(outputs.Select(b => _allocation[b].Start)));
        streamWriter.WriteLine(JoinOrZero(outputs.Select(b => b.Size() / b.ElemType.SizeInBytes)));
        streamWriter.WriteLine(Path.GetDirectoryName(BinFilePath) + "/");
        streamWriter.WriteLine(JoinOrZero(outputs.Select(b => b.ElemType.ToCModelTypeCode())));

        streamWriter.WriteLine("wait-key");
    }

    private static string JoinOrZero<T>(IEnumerable<T> values)
    {
        var joined = string.Join(",", values);
        return string.IsNullOrEmpty(joined) ? "0" : joined;
    }

    private void WriteVdutGLBCtrl()
    {
        using StreamWriter streamWriter = new StreamWriter(File.OpenWrite(DdrCtrlPath));
        streamWriter.WriteLine("00000000 00000000000000000000000000000000");
        streamWriter.WriteLine("00000010 00000000000000000000000000000000");
        streamWriter.WriteLine($"00400100 {StartInfo.Text:X32}");
        streamWriter.WriteLine("00400120 00000001000000010000000000000000");
    }

    private void WriteVdutInvokeArgs()
    {
        var outputs = _entry.Parameters.OutputOf().ToArray();

        if (outputs.Length != 1)
        {
            return;
        }

        var output = outputs[0];

        string simRoot = Path.GetFullPath(Path.Combine(
            GetThisFilePath(
                @"C:\GitLab-Runner\builds\sVHyYdAc\0\software\k80\nncase\nncase-k80\modules\Nncase.Modules.K230\CodeGen\GModelBuilder.cs"),
            "../../../../../maix3-arch-sim/"));

        using var streamWriter = new StreamWriter(
            File.Open(ArgFilePath + ".sc", FileMode.Create, FileAccess.Write, FileShare.Read));

        streamWriter.WriteLine(simRoot + "ic_env/ld-linux-x86-64.so.2");

        streamWriter.WriteLine(
            $"--library-path {simRoot}ic_env/ " +
            $"{simRoot}ic_env/Vkpu_top " +
            $"{BinFilePath} " +
            $"{StartInfo.Text} " +
            $"{StartInfo.Output} " +
            $"{output.Size() / output.ElemType.SizeInBytes} " +
            $"{Path.GetDirectoryName(BinFilePath)}/ " +
            $"{output.ElemType.ToCModelTypeCode()} " +
            "+perf +ckp +hang_10000");
    }

    public void Serialize()
    {
        if (IsSerialized)
        {
            return;
        }

        if (OnlyInstructions)
        {
            using BinaryWriter gw = new BinaryWriter(File.OpenWrite(BinFilePath));
            WriteInstText(gw);
        }
        else
        {
            WriteBin();
            AllocateBuffer();
            if (!EmbeddedParams)
            {
                WriteDesc();
            }
            else
            {
                AssemblyParser.OverrideGlbParams(CodeGenUtil.ToStackArgs(_entry.Parameters.ToArray(), _allocation));
                _entry = AssemblyParser.ParserFunction();
                using (BinaryWriter binaryWriter = new BinaryWriter(File.OpenWrite(BinFilePath)))
                {
                    binaryWriter.Position(StartInfo.Text);
                    new InstSerializeVisitor(binaryWriter).Visit(_entry);
                }

                WriteVdutGLBCtrl();
            }

            WriteVdutInvokeArgs();
            WriteInvokeArgs();
        }

        IsSerialized = true;
    }

    public void FillInputs(params Tensor[] inputs)
    {
        if (inputs.Length != _entry.Parameters.InputOf().Count())
        {
            throw new InvalidOperationException("Input Argument Number Error!");
        }

        using BinaryWriter binaryWriter = new BinaryWriter(File.OpenWrite(BinFilePath));
        foreach (var pair in from p in inputs.Zip(_entry.Parameters.ToArray())
                 where p.First.Rank != 0
                 select p)
        {
            Tensor input = pair.First;
            Nncase.TIR.Buffer buffer = pair.Second;
            Tensor tensor = input;
            binaryWriter.Position(_allocation[buffer].Start);
            if (tensor.BytesBuffer.Length != buffer.Size())
            {
                throw new InvalidOperationException("The Input Tensor Is Not Equal The Input Buffer!");
            }

            binaryWriter.Write(tensor.BytesBuffer);
        }
    }

    private static string GetThisFilePath([CallerFilePath] string path = "")
    {
        return path;
    }

    public Tensor[] Invoke(params Tensor[] inputs)
    {
        if (OnlyInstructions)
        {
            throw new InvalidOperationException("The Only Instructions Mode Can't Be Invoke!");
        }

        FillInputs(inputs);
        WriteGLBCtrl(inputs);
        string backend = Backend;
        string text;
        if (!(backend == "cmodel"))
        {
            if (!(backend == "sc_model"))
            {
                throw new ArgumentOutOfRangeException();
            }

            text = Path.GetFullPath(Path.Combine(
                GetThisFilePath(
                    "C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\CodeGen\\GModelBuilder.cs"),
                "../../../../../maix3-arch-sim/")) + "ic_env/ld-linux-x86-64.so.2";
        }
        else
        {
            text = "k230_cmodel_cli";
        }

        string exe = text;
        text = Backend;
        if (!(text == "cmodel"))
        {
            if (!(text == "sc_model"))
            {
                throw new ArgumentOutOfRangeException();
            }

            backend = File.ReadLines(ArgFilePath + ".sc").Skip(1).First();
        }
        else
        {
            backend = ArgFilePath;
        }

        string arguments = backend;
        new GModelEngine(exe, Path.GetDirectoryName(ArgFilePath) ?? string.Empty).Run(arguments);
        text = Path.GetDirectoryName(BinFilePath);
        string backend2 = Backend;
        if (!(backend2 == "cmodel"))
        {
            if (!(backend2 == "sc_model"))
            {
                throw new ArgumentOutOfRangeException();
            }

            backend = "nncase_result_0_sc.bin";
        }
        else
        {
            backend = "nncase_result_0.bin";
        }

        BinaryReader result_bin = new BinaryReader(File.OpenRead(Path.Combine(text, backend)));
        return _entry.Parameters.OutputOf()
            .Select<Tensor>((Func<Nncase.TIR.Buffer, Tensor>)((Nncase.TIR.Buffer o_buf) =>
                Tensor.FromBytes(o_buf.ElemType, result_bin.ReadBytes(o_buf.Size()), o_buf.Shape()))).ToArray();
    }
}
