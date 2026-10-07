// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.Instructions;

namespace Nncase.Evaluator.TIR.Instructions;

internal sealed class SHEvaluator : ITypeInferencer<SH>, ITypeInferencer, IOpPrinter<SH>, IOpPrinter
{
    public IRType Visit(ITypeInferenceContext context, SH target)
    {
        return TupleType.Void;
    }

    public string Visit(IIRPrinterContext context, SH target, bool ILmode)
    {
        return
            $"I.SH(rd: {target.rd}, rs: {target.rs}, offset: {context.GetArgument(target, SH.offset)}, funct3: {target.funct3})";
    }
}
