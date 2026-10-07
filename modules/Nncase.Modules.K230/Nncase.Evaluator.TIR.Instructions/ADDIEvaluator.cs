// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.Instructions;

namespace Nncase.Evaluator.TIR.Instructions;

internal sealed class ADDIEvaluator : ITypeInferencer<ADDI>, ITypeInferencer, IOpPrinter<ADDI>, IOpPrinter
{
    public IRType Visit(ITypeInferenceContext context, ADDI target)
    {
        return TupleType.Void;
    }

    public string Visit(IIRPrinterContext context, ADDI target, bool ILmode)
    {
        return
            $"I.ADDI(rd: {target.rd}, rs: {target.rs}, imm: {context.GetArgument(target, ADDI.imm)}, funct5: {target.funct5})";
    }
}
