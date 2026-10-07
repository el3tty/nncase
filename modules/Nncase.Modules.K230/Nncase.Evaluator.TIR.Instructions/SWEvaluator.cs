// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.Instructions;

namespace Nncase.Evaluator.TIR.Instructions;

internal sealed class SWEvaluator : ITypeInferencer<SW>, ITypeInferencer, IOpPrinter<SW>, IOpPrinter
{
    public IRType Visit(ITypeInferenceContext context, SW target)
    {
        return TupleType.Void;
    }

    public string Visit(IIRPrinterContext context, SW target, bool ILmode)
    {
        return
            $"I.SW(rd: {target.rd}, rs: {target.rs}, offset: {context.GetArgument(target, SW.offset)}, funct3: {target.funct3})";
    }
}
