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

	private Dictionary<Nncase.TIR.Buffer, (uint start, uint seg_offset, uint size)> _allocation = new Dictionary<Nncase.TIR.Buffer, (uint, uint, uint)>(ReferenceEqualityComparer.Instance);

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

	private string _ddrCtrlPath { get; set; }

	private string _argFilePath { get; set; }

	private string _descFilePath { get; set; }

	private DdrRegionStartInfo _startInfo { get; init; }

	public ValueTask InitializeAsync()
	{
		throw new NotImplementedException();
	}

	public ValueTask UninitializeAsync()
	{
		throw new NotImplementedException();
	}

	public GModelBuilder(PrimFunction? entry = null, AssemblyParser? assemblyParser = null, bool embedded_params = false)
	{
		IsSerialized = false;
		StackSize = assemblyParser?.ParserStackSize() ?? 262144;
		OnlyInstructions = assemblyParser != null && assemblyParser.Params == null;
		AssemblyParser = assemblyParser;
		EmbeddedParams = embedded_params;
		_entry = (((object)entry == null) ? AssemblyParser.ParserFunction() : entry);
		BinFilePath = Nncase.CodeGen.CodeGenUtil.GetTempFileName(".bin");
		_ddrCtrlPath = Nncase.CodeGen.CodeGenUtil.GetTempFileName(".pc_addr_ctrl");
		_argFilePath = Nncase.CodeGen.CodeGenUtil.GetTempFileName(".invoke_command");
		_descFilePath = Nncase.CodeGen.CodeGenUtil.GetTempFileName(".desc");
		_startInfo = new DdrRegionStartInfo();
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
			_ddrCtrlPath = Path.Join(dumpDirPath, name + Path.GetExtension(_ddrCtrlPath));
			_argFilePath = Path.Join(dumpDirPath, name + Path.GetExtension(_argFilePath));
			_descFilePath = Path.Join(dumpDirPath, name + Path.GetExtension(_descFilePath));
			Serialize();
		}
		if (Path.GetDirectoryName(BinFilePath) != dumpDirPath)
		{
			Func<string, bool, string> func = delegate(string old_path, bool copy)
			{
				string text = Path.Join(dumpDirPath, name + Path.GetExtension(old_path));
				if (File.Exists(text))
				{
					File.Delete(text);
				}
				if (copy)
				{
					File.Copy(old_path, text);
				}
				return text;
			};
			BinFilePath = func(BinFilePath, arg2: true);
			if (!OnlyInstructions)
			{
				_ddrCtrlPath = func(_ddrCtrlPath, arg2: false);
				_argFilePath = func(_argFilePath, arg2: true);
				writeInvokeArgs();
				writeVdutInvokeArgs();
				if (!EmbeddedParams)
				{
					_descFilePath = func(_descFilePath, arg2: true);
				}
			}
		}
		return BinFilePath;
	}

	private void writeBin()
	{
		//IL_0037: Unknown result type (might be due to invalid IL or missing references)
		//IL_003c: Unknown result type (might be due to invalid IL or missing references)
		using BinaryWriter binaryWriter = new BinaryWriter(File.OpenWrite(BinFilePath));
		uint num = 0u;
		_startInfo.input = 0u;
		binaryWriter.Position(_startInfo.input + _entry.Parameters.InputBufferOf().ToArray().Aggregate(0L, (long acc, Nncase.TIR.Buffer bf) => acc + ((TensorConst)bf.MemSpan.Size).Value.ToScalar<int>()));
		binaryWriter.AlignPosition(8L);
		_startInfo.rdata = (uint)binaryWriter.Position();
		binaryWriter.Write(Array.Empty<byte>());
		binaryWriter.AlignPosition(8L);
		_startInfo.text = (uint)binaryWriter.Position();
		WriteInstText(binaryWriter);
		_startInfo.data = (uint)binaryWriter.Position();
		_startInfo.output = _startInfo.data + num;
	}

	private void WriteInstText(BinaryWriter gw)
	{
		new InstSerializeVisitor(gw).Visit(_entry);
	}

	private void AllocateBuffer()
	{
		//IL_0015: Unknown result type (might be due to invalid IL or missing references)
		//IL_001a: Unknown result type (might be due to invalid IL or missing references)
		//IL_001e: Unknown result type (might be due to invalid IL or missing references)
		//IL_0023: Unknown result type (might be due to invalid IL or missing references)
		//IL_0085: Unknown result type (might be due to invalid IL or missing references)
		//IL_008a: Unknown result type (might be due to invalid IL or missing references)
		//IL_008e: Unknown result type (might be due to invalid IL or missing references)
		//IL_0093: Unknown result type (might be due to invalid IL or missing references)
		List<(uint, uint, uint)> list = new List<(uint, uint, uint)>();
		uint num = 0u;
		uint num2 = 0u;
		WhereEnumerator<Nncase.TIR.Buffer, FunctionWrapper<Nncase.TIR.Buffer, bool>> enumerator = _entry.Parameters.InputBufferOf().GetEnumerator();
		while (enumerator.MoveNext())
		{
			Nncase.TIR.Buffer current = enumerator.Current;
			list.Add((num2 - num + _startInfo.input, num2, (uint)current.Size()));
			_allocation.Add(current, list.Last());
			num2 += (uint)current.Size();
		}
		uint num3 = 0u;
		enumerator = _entry.Parameters.OutputOf().GetEnumerator();
		while (enumerator.MoveNext())
		{
			Nncase.TIR.Buffer current2 = enumerator.Current;
			list.Add((num3 - num + _startInfo.output, num3, (uint)current2.Size()));
			_allocation.Add(current2, list.Last());
			num3 += (uint)current2.Size();
		}
		list.Add((_startInfo.rdata, 0u, 0u));
		list.Add((_startInfo.data, 0u, 0u));
	}

	private void writeGLBCtrl(Tensor[] inputs)
	{
		using FileStream stream = File.OpenWrite(_ddrCtrlPath);
		using StreamWriter streamWriter = new StreamWriter(stream);
		Dictionary<Nncase.TIR.Buffer, int> dictionary = new Dictionary<Nncase.TIR.Buffer, int>(ReferenceEqualityComparer.Instance);
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
		Dictionary<Nncase.TIR.Buffer, (uint, uint, uint)> dictionary2 = new Dictionary<Nncase.TIR.Buffer, (uint, uint, uint)>(_allocation, ReferenceEqualityComparer.Instance);
		foreach (Nncase.TIR.Buffer item2 in dictionary2.Keys.Where((Nncase.TIR.Buffer b) => b.MemSpan.Location == MemoryLocation.Output))
		{
			(uint, uint, uint) tuple = dictionary2[item2];
			dictionary2[item2] = (tuple.Item1 - _startInfo.output, tuple.Item2, tuple.Item3);
		}
		List<int> source = CodeGenUtil.ToStackArgs(_entry.Parameters.ToArray(), dictionary2, dictionary);
		int num = StackSize - 16;
		foreach (string[] item3 in source.Select((int a) => $"{a:X8}").Chunk(4))
		{
			string text = string.Empty;
			for (int num2 = 0; num2 < 4; num2++)
			{
				string text2 = ((num2 < item3.Length) ? item3[num2] : $"{0:X8}");
				text += text2;
			}
			streamWriter.Write($"{num:X8} ");
			streamWriter.WriteLine(text);
			num -= 16;
		}
		streamWriter.WriteLine($"00000000 {0:X8}{_startInfo.output:X8}{0:X8}{0:X8}");
		streamWriter.WriteLine($"00400100 {_startInfo.text:X32}");
		streamWriter.WriteLine("00400120 00000001000000010000000000000000");
	}

	private void writeDesc()
	{
		//IL_0027: Unknown result type (might be due to invalid IL or missing references)
		//IL_002c: Unknown result type (might be due to invalid IL or missing references)
		//IL_0052: Unknown result type (might be due to invalid IL or missing references)
		//IL_0057: Unknown result type (might be due to invalid IL or missing references)
		//IL_007b: Unknown result type (might be due to invalid IL or missing references)
		//IL_0080: Unknown result type (might be due to invalid IL or missing references)
		//IL_0083: Unknown result type (might be due to invalid IL or missing references)
		//IL_0088: Unknown result type (might be due to invalid IL or missing references)
		//IL_0148: Unknown result type (might be due to invalid IL or missing references)
		//IL_014d: Unknown result type (might be due to invalid IL or missing references)
		//IL_0150: Unknown result type (might be due to invalid IL or missing references)
		//IL_0155: Unknown result type (might be due to invalid IL or missing references)
		//IL_0215: Unknown result type (might be due to invalid IL or missing references)
		//IL_021a: Unknown result type (might be due to invalid IL or missing references)
		//IL_021d: Unknown result type (might be due to invalid IL or missing references)
		//IL_0222: Unknown result type (might be due to invalid IL or missing references)
		//IL_02c0: Unknown result type (might be due to invalid IL or missing references)
		//IL_02c5: Unknown result type (might be due to invalid IL or missing references)
		//IL_02c8: Unknown result type (might be due to invalid IL or missing references)
		//IL_02cd: Unknown result type (might be due to invalid IL or missing references)
		using StreamWriter streamWriter = new StreamWriter(File.OpenWrite(_descFilePath));
		streamWriter.WriteLine($"{_entry.Parameters.InputBufferOf().Count()} {_entry.Parameters.OutputOf().Count()}");
		WhereEnumerator<Nncase.TIR.Buffer, FunctionWrapper<Nncase.TIR.Buffer, bool>> enumerator = _entry.Parameters.InputBufferOf().GetEnumerator();
		while (enumerator.MoveNext())
		{
			Nncase.TIR.Buffer current = enumerator.Current;
			int[] array = current.Shape();
			streamWriter.WriteLine($"{current.ElemType.GetDisplayName()} ({array[0]}, {array[1]}, {array[2]}, {array[3]})");
		}
		enumerator = _entry.Parameters.OutputOf().GetEnumerator();
		while (enumerator.MoveNext())
		{
			Nncase.TIR.Buffer current2 = enumerator.Current;
			int[] array2 = current2.Shape();
			streamWriter.WriteLine($"{current2.ElemType.GetDisplayName()} ({array2[0]}, {array2[1]}, {array2[2]}, {array2[3]})");
		}
		enumerator = _entry.Parameters.InputBufferOf().GetEnumerator();
		while (enumerator.MoveNext())
		{
			Nncase.TIR.Buffer current3 = enumerator.Current;
			streamWriter.WriteLine($"{_allocation[current3].start} {_allocation[current3].seg_offset} {_allocation[current3].size}");
		}
		enumerator = _entry.Parameters.OutputOf().GetEnumerator();
		while (enumerator.MoveNext())
		{
			Nncase.TIR.Buffer current4 = enumerator.Current;
			streamWriter.WriteLine($"{_allocation[current4].start} {_allocation[current4].seg_offset} {_allocation[current4].size}");
		}
		streamWriter.WriteLine($"{_startInfo.rdata} {0} {_startInfo.data - _startInfo.rdata}");
		streamWriter.WriteLine($"{_startInfo.data} {0} {0}");
		streamWriter.WriteLine($"{_startInfo.text} {0} {_startInfo.data - _startInfo.text}");
	}

    private void writeInvokeArgs()
    {
        using var streamWriter = new StreamWriter(
            File.Open(_argFilePath, FileMode.Create, FileAccess.Write, FileShare.Read));

        var outputs = _entry.Parameters.OutputOf().ToArray();

        streamWriter.WriteLine(BinFilePath);
        streamWriter.WriteLine(_startInfo.text);

        streamWriter.WriteLine(JoinOrZero(outputs.Select(b => _allocation[b].start)));
        streamWriter.WriteLine(JoinOrZero(outputs.Select(b => b.Size() / b.ElemType.SizeInBytes)));
        streamWriter.WriteLine(Path.GetDirectoryName(BinFilePath) + "/");
        streamWriter.WriteLine(JoinOrZero(outputs.Select(b => b.ElemType.ToCModelTypeCode())));

        streamWriter.WriteLine("wait-key");
    }

    private static string JoinOrZero<T>(IEnumerable<T> values)
    {
        var text = string.Join(",", values);
        return string.IsNullOrEmpty(text) ? "0" : text;
    }

	private void writeVdutGLBCtrl()
	{
		using StreamWriter streamWriter = new StreamWriter(File.OpenWrite(_ddrCtrlPath));
		streamWriter.WriteLine("00000000 00000000000000000000000000000000");
		streamWriter.WriteLine("00000010 00000000000000000000000000000000");
		streamWriter.WriteLine($"00400100 {_startInfo.text:X32}");
		streamWriter.WriteLine("00400120 00000001000000010000000000000000");
	}

    private void writeVdutInvokeArgs()
    {
        var outputs = _entry.Parameters.OutputOf().ToArray();

        if (outputs.Length != 1)
            return;

        var output = outputs[0];

        string simRoot = Path.GetFullPath(Path.Combine(
            GetThisFilePath(@"C:\GitLab-Runner\builds\sVHyYdAc\0\software\k80\nncase\nncase-k80\modules\Nncase.Modules.K230\CodeGen\GModelBuilder.cs"),
            "../../../../../maix3-arch-sim/"));

        using var streamWriter = new StreamWriter(
            File.Open(_argFilePath + ".sc", FileMode.Create, FileAccess.Write, FileShare.Read));

        streamWriter.WriteLine(simRoot + "ic_env/ld-linux-x86-64.so.2");

        streamWriter.WriteLine(
            $"--library-path {simRoot}ic_env/ " +
            $"{simRoot}ic_env/Vkpu_top " +
            $"{BinFilePath} " +
            $"{_startInfo.text} " +
            $"{_startInfo.output} " +
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
			writeBin();
			AllocateBuffer();
			if (!EmbeddedParams)
			{
				writeDesc();
			}
			else
			{
				AssemblyParser.OverrideGlbParams(CodeGenUtil.ToStackArgs(_entry.Parameters.ToArray(), _allocation));
				_entry = AssemblyParser.ParserFunction();
				using (BinaryWriter binaryWriter = new BinaryWriter(File.OpenWrite(BinFilePath)))
				{
					binaryWriter.Position(_startInfo.text);
					new InstSerializeVisitor(binaryWriter).Visit(_entry);
				}
				writeVdutGLBCtrl();
			}
			writeVdutInvokeArgs();
			writeInvokeArgs();
		}
		IsSerialized = true;
	}

	public void FillInputs(params Tensor[] inputs)
	{
		//IL_000e: Unknown result type (might be due to invalid IL or missing references)
		//IL_0013: Unknown result type (might be due to invalid IL or missing references)
		if (inputs.Length != _entry.Parameters.InputOf().Count())
		{
			throw new InvalidOperationException("Input Argument Number Error!");
		}
		using BinaryWriter binaryWriter = new BinaryWriter(File.OpenWrite(BinFilePath));
		foreach (var item3 in from p in inputs.Zip(_entry.Parameters.ToArray())
			where p.First.Rank != 0
			select p)
		{
			Tensor item = item3.First;
			Nncase.TIR.Buffer item2 = item3.Second;
			Tensor tensor = item;
			binaryWriter.Position(_allocation[item2].start);
			if (tensor.BytesBuffer.Length != item2.Size())
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
		//IL_016a: Unknown result type (might be due to invalid IL or missing references)
		//IL_016f: Unknown result type (might be due to invalid IL or missing references)
		//IL_017f: Unknown result type (might be due to invalid IL or missing references)
		//IL_0184: Unknown result type (might be due to invalid IL or missing references)
		if (OnlyInstructions)
		{
			throw new InvalidOperationException("The Only Instructions Mode Can't Be Invoke!");
		}
		FillInputs(inputs);
		writeGLBCtrl(inputs);
		string backend = Backend;
		string text;
		if (!(backend == "cmodel"))
		{
			if (!(backend == "sc_model"))
			{
				throw new ArgumentOutOfRangeException();
			}
			text = Path.GetFullPath(Path.Combine(GetThisFilePath("C:\\GitLab-Runner\\builds\\sVHyYdAc\\0\\software\\k80\\nncase\\nncase-k80\\modules\\Nncase.Modules.K230\\CodeGen\\GModelBuilder.cs"), "../../../../../maix3-arch-sim/")) + "ic_env/ld-linux-x86-64.so.2";
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
			backend = File.ReadLines(_argFilePath + ".sc").Skip(1).First();
		}
		else
		{
			backend = _argFilePath;
		}
		string arguments = backend;
		new GModelEngine(exe, Path.GetDirectoryName(_argFilePath) ?? "").Run(arguments);
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
		return _entry.Parameters.OutputOf().Select<Tensor>((Func<Nncase.TIR.Buffer, Tensor>)((Nncase.TIR.Buffer o_buf) => Tensor.FromBytes(o_buf.ElemType, result_bin.ReadBytes(o_buf.Size()), o_buf.Shape()))).ToArray();
	}
}
