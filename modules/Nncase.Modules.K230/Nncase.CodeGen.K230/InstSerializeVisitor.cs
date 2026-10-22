// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using Nncase.IR;
using Nncase.TIR.Instructions;

namespace Nncase.CodeGen.K230;

internal sealed class InstSerializeVisitor : ExprVisitor<bool, bool>
{
    private readonly BinaryWriter _writer;

    public InstSerializeVisitor(BinaryWriter binaryWriter)
        : base(false)
    {
        _writer = binaryWriter;
    }

    protected override bool VisitLeafCall(Call expr)
    {
        if (expr.Target is ISerializeInst serializeInst)
        {
            serializeInst.Serialize(_writer, expr);
            return true;
        }

        throw new InvalidOperationException("The " + expr.Target.GetType().Name + " is invalid in here!");
    }

    protected override bool DefaultVisitLeaf(Expr expr)
    {
        return true;
    }
}
