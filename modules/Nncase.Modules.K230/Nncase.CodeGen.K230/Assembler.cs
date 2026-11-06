// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace Nncase.CodeGen.K230;

internal sealed class Assembler
{
    private abstract class AsmLine
    {
        public string Line { get; set; }

        public string Update { get; set; }

        public int Pc { get; set; }

        public AsmLine(string line, int pc)
        {
            Line = line;
            Pc = pc;
            Update = string.Empty;
        }
    }

    private sealed class AsmInst : AsmLine
    {
        public string InstName { get; set; }

        public AsmInst(string inst_name, string line, int pc)
            : base(line, pc)
        {
            InstName = inst_name;
        }
    }

    private sealed class AsmSymobl : AsmLine
    {
        public string SymbolName { get; set; }

        public AsmSymobl(string symbol_name, string line, int pc)
            : base(line, pc)
        {
            SymbolName = symbol_name;
        }
    }

    private sealed class AsmLabel : AsmLine
    {
        public string LabelName { get; set; }

        public AsmLabel(string label_name, string line, int pc)
            : base(line, pc)
        {
            LabelName = label_name;
        }
    }

    private sealed class AsmOther : AsmLine
    {
        public AsmOther(string line, int pc)
            : base(line, pc)
        {
        }
    }

    public string AsmPath { get; set; }

    private readonly List<AsmLine> _asmLines;

    private readonly Dictionary<string, int> _symobelMap;

    private readonly Dictionary<string, int> _labelMap;

    public Assembler(string asm_path)
    {
        AsmPath = asm_path;
        _asmLines = new List<AsmLine>();
        _symobelMap = new Dictionary<string, int>();
        _labelMap = new Dictionary<string, int>();
    }

    public void Assemble()
    {
        Collect(AsmPath);
        int dumpIndex = 0;
        bool needsRelocate = false;
        do
        {
            if (needsRelocate)
            {
                Relocate();
                DumpRelocate(Path.Combine(
                    Path.GetDirectoryName(AsmPath) ?? string.Empty,
                    Path.GetFileNameWithoutExtension(AsmPath) + "_" + dumpIndex++ + ".s"));
                needsRelocate = false;
            }

            foreach (var (inst, index) in from t in _asmLines.Select((AsmLine l, int i) => (l: l, i: i))
                     where t.l is AsmInst
                     select ((AsmInst)t.l, i: t.i))
            {
                if (ReplaceSymobl(inst, index))
                {
                    needsRelocate = true;
                    break;
                }

                if (ReplaceLabel(inst, index))
                {
                    needsRelocate = true;
                    break;
                }
            }
        } while (needsRelocate);
    }

    private void Collect(string asm_path)
    {
        using StreamReader streamReader = new StreamReader(File.OpenRead(asm_path));
        int pc = 0;
        while (true)
        {
            string rawLine = streamReader.ReadLine();
            if (rawLine == null)
            {
                break;
            }

            string line = rawLine;
            if (line.StartsWith(".L."))
            {
                string nameLine = line;
                AsmLabel asmLabel = new AsmLabel(nameLine.Substring(0, nameLine.Length - 1), line, pc);
                _asmLines.Add(asmLabel);
                _labelMap.Add(asmLabel.LabelName, pc);
                continue;
            }

            string symbolLine = line;
            if (!symbolLine.StartsWith("  ") && symbolLine.EndsWith(":"))
            {
                string nameLine = symbolLine;
                AsmSymobl asmSymobl = new AsmSymobl(nameLine.Substring(0, nameLine.Length - 1), symbolLine, pc);
                _asmLines.Add(asmSymobl);
                _symobelMap.Add(asmSymobl.SymbolName, pc);
                continue;
            }

            string instLine = line;
            if (instLine.StartsWith("  I."))
            {
                AsmInst asmInst = new AsmInst(instLine.Substring(4, instLine.IndexOf("(") - 4), instLine, pc);
                _asmLines.Add(asmInst);
                pc += CodeGenUtil.InstLength(asmInst.InstName);
                continue;
            }

            if (line.StartsWith("  ."))
            {
                _asmLines.Add(new AsmOther(rawLine, pc));
                continue;
            }

            throw new ArgumentOutOfRangeException(rawLine);
        }
    }

    private bool IsOverflow(int imm, int bits = 12)
    {
        if (imm < 0)
        {
            int highBitsMask = ~((1 << bits - 1) - 1);
            if (highBitsMask == (imm & highBitsMask))
            {
                return false;
            }

            return true;
        }

        return imm > 1 << bits - 1;
    }

    private (uint High, uint Low) Split(int imm)
    {
        int high = ((imm >> 12) + ((imm >> 11) & 1)) & 0xFFFFF;
        uint low = (uint)(imm & 0xFFF);
        return (High: (uint)high, Low: low);
    }

    private void Relocate()
    {
        _symobelMap.Clear();
        _labelMap.Clear();
        int pc = 0;
        foreach (AsmLine asmLine in _asmLines)
        {
            if (!(asmLine is AsmLabel asmLabel))
            {
                if (!(asmLine is AsmSymobl asmSymobl))
                {
                    if (asmLine is AsmInst asmInst)
                    {
                        asmInst.Pc = pc;
                        pc += CodeGenUtil.InstLength(asmInst.InstName);
                    }
                    else
                    {
                        asmLine.Pc = pc;
                    }
                }
                else
                {
                    asmSymobl.Pc = pc;
                    _symobelMap[asmSymobl.SymbolName] = pc;
                }
            }
            else
            {
                asmLabel.Pc = pc;
                _labelMap[asmLabel.LabelName] = pc;
            }
        }
    }

    private bool ReplaceLabel(AsmInst inst, int index)
    {
        Match match = new Regex("  I.(\\w+)\\(.*(\\.L\\..*)\\)").Match(inst.Line);
        if (match == Match.Empty)
        {
            return false;
        }

        string value = match.Groups[2].Value;
        int offset;
        if (value.EndsWith(".high"))
        {
            Dictionary<string, int> labelMap = _labelMap;
            string labelRef = value;
            offset = labelMap[labelRef.Substring(0, labelRef.Length - 5)] - inst.Pc;
            inst.Update = inst.Line.Replace(value, Split(offset).High.ToString(), StringComparison.Ordinal);
        }
        else if (value.EndsWith(".low"))
        {
            Dictionary<string, int> labelMap2 = _labelMap;
            string labelRef = value;
            offset = labelMap2[labelRef.Substring(0, labelRef.Length - 4)] - inst.Pc;
            inst.Update = inst.Line.Replace(value, Split(offset + 4).Low.ToString(), StringComparison.Ordinal);
        }
        else
        {
            offset = _labelMap[value] - inst.Pc;
            int bits = 12;
            if (inst.InstName == "JAL")
            {
                bits = 20;
            }

            if (IsOverflow(offset, bits))
            {
                if (inst.InstName == "JAL")
                {
                    throw new NotSupportedException("overflow imm!");
                }

                inst.Line = inst.Line.Replace(value, "8", StringComparison.Ordinal);
                _asmLines.InsertRange(
                    index + 1,
                    new AsmInst[4]
                    {
                        new AsmInst("JAL", "  I.JAL(R.r0,16)", 0),
                        new AsmInst("AUIPC", "  I.AUIPC(R.r10," + value + ".high)", 0),
                        new AsmInst("ADDI", "  I.ADDI(R.r10,R.r10," + value + ".low)", 0),
                        new AsmInst("JALR", "  I.JALR(R.r0,R.r10,0) ", 0),
                    });
                return true;
            }

            inst.Update = inst.Line.Replace(value, offset.ToString(), StringComparison.Ordinal);
        }

        inst.Update += $" # {value} = {inst.Pc} + {offset}";
        return false;
    }

    private bool ReplaceSymobl(AsmInst inst, int index)
    {
        Match match = new Regex("\\(.*,?.*(@[a-zA-Z\\d_]+\\.(\\w+)).*,?.*\\)").Match(inst.Line);
        if (match == Match.Empty)
        {
            return false;
        }

        string symbolRef = match.Groups[1].Value.Trim();
        Dictionary<string, int> symobelMap = _symobelMap;
        string symbolExpr = symbolRef;
        int suffixLength = match.Groups[2].Length + 1;
        int offset = symobelMap[symbolExpr.Substring(1, symbolExpr.Length - suffixLength - 1)] - inst.Pc;
        uint immediate;
        if (match.Groups[2].Value == "high")
        {
            immediate = Split(offset).High;
        }
        else
        {
            if (!(match.Groups[2].Value == "low"))
            {
                throw new ArgumentOutOfRangeException();
            }

            immediate = Split(offset + 4).Low;
        }

        inst.Update = inst.Line.Replace(symbolRef, immediate.ToString(), StringComparison.Ordinal);
        inst.Update += $" # {symbolRef} = {inst.Pc} + {offset}";
        return false;
    }

    public void DumpRelocate(string output_path)
    {
        using StreamWriter streamWriter = new StreamWriter(File.Open(output_path, FileMode.Create));
        foreach (AsmLine asmLine in _asmLines)
        {
            streamWriter.WriteLine(asmLine.Line);
        }
    }

    public void Dump(string output_path)
    {
        using StreamWriter streamWriter = new StreamWriter(File.Open(output_path, FileMode.Create));
        foreach (AsmLine asmLine in _asmLines)
        {
            if (asmLine is AsmInst asmInst)
            {
                streamWriter.Write($"{asmInst.Pc,8:D8}");
            }

            streamWriter.WriteLine((asmLine.Update == string.Empty) ? asmLine.Line : asmLine.Update);
        }
    }
}
