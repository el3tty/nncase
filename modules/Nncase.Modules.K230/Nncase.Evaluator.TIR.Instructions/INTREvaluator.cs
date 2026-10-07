// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.Instructions;

namespace Nncase.Evaluator.TIR.Instructions;

internal sealed class INTREvaluator : ITypeInferencer<INTR>, ITypeInferencer, IOpPrinter<INTR>, IOpPrinter
{
    public IRType Visit(ITypeInferenceContext context, INTR target)
    {
        return TupleType.Void;
    }

    public string Visit(IIRPrinterContext context, INTR target, bool ILmode)
    {
        return $"I.INTR(rs: {target.rs})";
    }
}
