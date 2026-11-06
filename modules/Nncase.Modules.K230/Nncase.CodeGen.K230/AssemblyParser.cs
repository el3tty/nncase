// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Text.RegularExpressions;
using Nncase.IR;
using Nncase.TIR;
using Nncase.TIR.Builders;

namespace Nncase.CodeGen.K230;

public sealed class AssemblyParser
{
    public string AsmOutPath { get; set; }

    public Nncase.TIR.Buffer[]? Params { get; set; }

    public Regex InstFinder { get; } = new Regex("^(\\d{8})\\s\\s(I\\.(\\w+)\\((.*?)\\))", RegexOptions.Multiline);

    public AssemblyParser(string asm_out_path, CallableType? signature)
    {
        AsmOutPath = asm_out_path;
        Params = ((object)signature == null) ? null : CodeGenUtil.ToPrimFuncParameters(signature).ToArray();
    }

    public void OverrideGlbParams(IReadOnlyList<int> args)
    {
        string[] lines = File.ReadAllLines(AsmOutPath);
        int glbStartLine = 0;
        for (int i = 0; i < lines.Length; i++)
        {
            if (lines[i].StartsWith("gnne_start:"))
            {
                glbStartLine = i;
                break;
            }
        }

        glbStartLine += 6;
        int lineOffset = 1;
        Regex regex = new Regex("(^\\d{8}\\s\\sI.ADDI\\(R.rax,R.r0,)(0)(\\))");
        foreach (byte[] argBytes in args.Select(BitConverter.GetBytes))
        {
            for (int byteIndex = 3; byteIndex >= 0; byteIndex--)
            {
                byte value = argBytes[byteIndex];
                lines[glbStartLine + lineOffset] = regex.Replace(lines[glbStartLine + lineOffset], $"${{1}}{value}$3");
                lineOffset += 2;
            }
        }

        File.WriteAllLines(AsmOutPath, lines);
    }

    public IEnumerable<Expr> ParserInst()
    {
        MatchCollection source = InstFinder.Matches(File.ReadAllText(AsmOutPath));
        _ = new Dictionary<string, MethodCallExpression>();
        return source.OfType<Match>().Select(delegate(Match m)
        {
            string value = m.Groups[3].Value;
            string[] array = (from s in m.Groups[4].Value.Split(',')
                where s != string.Empty
                select s).ToArray();
            MethodInfo method = typeof(I).GetMethod(value);
            IEnumerable<Expression> arguments = array.Zip(method.GetParameters())
                .Select<(string, System.Reflection.ParameterInfo), Expression>(
                    delegate((string, System.Reflection.ParameterInfo) t)
                    {
                        string operand = t.Item1.Trim();
                        if (operand.StartsWith("R."))
                        {
                            Type? typeFromHandle = typeof(R);
                            string operandText = operand;
                            return Expression.Field(
                                null,
                                typeFromHandle.GetField(operandText.Substring(2, operandText.Length - 2)));
                        }

                        string value2 = operand;
                        if (t.Item2.ParameterType == typeof(Expr))
                        {
                            return Expression.Convert(
                                Expression.Constant(Convert.ToInt32(value2), typeof(int)),
                                typeof(Expr));
                        }

                        string value3 = operand;
                        if (!t.Item2.ParameterType.IsEnum)
                        {
                            throw new ArgumentOutOfRangeException();
                        }

                        return Expression.Constant(Enum.Parse(t.Item2.ParameterType, value3), t.Item2.ParameterType);
                    }).Concat(from p in method.GetParameters().Skip(array.Length)
                    select (p.DefaultValue is DBNull)
                        ? Expression.Constant(null, p.ParameterType)
                        : Expression.Constant(p.DefaultValue, p.ParameterType));
            return Expression.Lambda<Func<Expr>>(Expression.Call(method, arguments), Array.Empty<ParameterExpression>())
                .Compile()();
        });
    }

    public PrimFunction ParserFunction()
    {
        ISequentialBuilder<PrimFunction> sequentialBuilder =
            T.PrimFunc("main", "k230", Params ?? Array.Empty<Nncase.TIR.Buffer>());
        object[] exprOrBuilders = ParserInst().ToArray();
        return sequentialBuilder.Body(exprOrBuilders).Build();
    }

    public int ParserStackSize()
    {
        using StreamReader streamReader = new StreamReader(File.OpenRead(AsmOutPath));
        return Convert.ToInt32(Regex.Match(streamReader.ReadLine(), "^\\s\\s.stack_size\\s(\\d+)").Groups[1].Value);
    }
}
