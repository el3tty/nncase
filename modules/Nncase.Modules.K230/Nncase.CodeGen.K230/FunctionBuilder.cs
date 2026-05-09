using System;
using System.Collections.Generic;
using System.IO;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.CodeGen.K230;

internal class FunctionBuilder : IDisposable
{
    private struct MemoryRange
    {
        public uint Start;

        public uint Size;
    }

    private struct DescHeader
    {
        public uint InputPoolSize;

        public uint OutputPoolSize;

        public uint Inputs;

        public uint Outputs;
    }

    private readonly uint _id;

    private readonly SectionManager _textSectionManager;

    private readonly BinaryWriter _textWriter;

    private readonly BinaryWriter _rdataWriter;

    public FunctionBuilder(uint id, BinaryWriter rdataWriter)
    {
        _id = id;
        _textSectionManager = new SectionManager();
        _textWriter = _textSectionManager.GetWriter(".text");
        _rdataWriter = rdataWriter;
    }

    public unsafe LinkableFunction Build(PrimFunction function)
    {
        //IL_006e: Unknown result type (might be due to invalid IL or missing references)
        //IL_0073: Unknown result type (might be due to invalid IL or missing references)
        //IL_0096: Unknown result type (might be due to invalid IL or missing references)
        //IL_009b: Unknown result type (might be due to invalid IL or missing references)
        //IL_009f: Unknown result type (might be due to invalid IL or missing references)
        //IL_00a4: Unknown result type (might be due to invalid IL or missing references)
        //IL_011f: Unknown result type (might be due to invalid IL or missing references)
        //IL_0124: Unknown result type (might be due to invalid IL or missing references)
        //IL_0147: Unknown result type (might be due to invalid IL or missing references)
        //IL_014c: Unknown result type (might be due to invalid IL or missing references)
        //IL_0150: Unknown result type (might be due to invalid IL or missing references)
        //IL_0155: Unknown result type (might be due to invalid IL or missing references)
        new InstSerializeVisitor(_textWriter).Visit(function.Body);
        SectionManager sectionManager = new SectionManager();
        using (BinaryWriter writer = sectionManager.GetWriter(".desc"))
        {
            DescHeader value = new DescHeader { InputPoolSize = 0u, OutputPoolSize = 0u, Inputs = 0u, Outputs = 0u };
            long pos = writer.Position();
            writer.Skip((ulong)sizeof(DescHeader));
            WhereEnumerator<Nncase.TIR.Buffer, FunctionWrapper<Nncase.TIR.Buffer, bool>> enumerator =
                ArrayExtensions.AsValueEnumerable<Nncase.TIR.Buffer>(function.Parameters)
                    .Where((Func<Nncase.TIR.Buffer, bool>)((Nncase.TIR.Buffer buf) =>
                        buf.MemSpan.Location == MemoryLocation.Input)).GetEnumerator();
            while (enumerator.MoveNext())
            {
                Nncase.TIR.Buffer current = enumerator.Current;
                value.Inputs++;
                MemoryRange value2 = checked(new MemoryRange
                {
                    Start = (uint)current.Start(), Size = (uint)current.Size()
                });
                writer.Write(ref value2);
                value.InputPoolSize = Math.Max(value.InputPoolSize, value2.Start + value2.Size);
            }

            enumerator = ArrayExtensions.AsValueEnumerable<Nncase.TIR.Buffer>(function.Parameters)
                .Where((Func<Nncase.TIR.Buffer, bool>)((Nncase.TIR.Buffer buf) =>
                    buf.MemSpan.Location == MemoryLocation.Output)).GetEnumerator();
            while (enumerator.MoveNext())
            {
                Nncase.TIR.Buffer current2 = enumerator.Current;
                value.Outputs++;
                MemoryRange value3 = checked(new MemoryRange
                {
                    Start = (uint)current2.Start(), Size = (uint)current2.Size()
                });
                writer.Write(ref value3);
                value.OutputPoolSize = Math.Max(value.OutputPoolSize, value3.Start + value3.Size);
            }

            writer.Position(pos);
            writer.Write(ref value);
        }

        foreach (KeyValuePair<Const, ValueRange<long>> rdata in function.SchedResult.Rdatas)
        {
            rdata.Deconstruct(out var key, out var value4);
            Const obj = key;
            ValueRange<long> valueRange = value4;
            Span<byte> bytesBuffer = ((TensorConst)obj).Value.BytesBuffer;
            if ((uint)bytesBuffer.Length != valueRange.Max - valueRange.Min)
            {
                throw new InvalidDataException("The Buffer Szie Not Equal!");
            }

            _rdataWriter.Position((uint)valueRange.Min);
            _rdataWriter.Write(bytesBuffer);
        }

        return new LinkableFunction(_id, function, _textSectionManager.GetContent(".text"),
            sectionManager.GetContent(".desc"));
    }

    public void Dispose()
    {
    }
}
