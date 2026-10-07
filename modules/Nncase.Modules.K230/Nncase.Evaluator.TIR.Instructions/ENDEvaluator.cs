// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.Instructions;

namespace Nncase.Evaluator.TIR.Instructions;

internal sealed class ENDEvaluator : ITypeInferencer<END>, ITypeInferencer, IOpPrinter<END>, IOpPrinter
{
    public IRType Visit(ITypeInferenceContext context, END target)
    {
        return TupleType.Void;
    }

    public string Visit(IIRPrinterContext context, END target, bool ILmode)
    {
        return $"I.END(rs: {target.rs})";
    }
}
