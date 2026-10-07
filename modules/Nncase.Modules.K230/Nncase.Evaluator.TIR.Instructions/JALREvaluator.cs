// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.Instructions;

namespace Nncase.Evaluator.TIR.Instructions;

internal sealed class JALREvaluator : ITypeInferencer<JALR>, ITypeInferencer, IOpPrinter<JALR>, IOpPrinter
{
    public IRType Visit(ITypeInferenceContext context, JALR target)
    {
        return TupleType.Void;
    }

    public string Visit(IIRPrinterContext context, JALR target, bool ILmode)
    {
        return $"I.JALR(rd: {target.rd}, rs: {target.rs}, offset: {context.GetArgument(target, JALR.offset)})";
    }
}
