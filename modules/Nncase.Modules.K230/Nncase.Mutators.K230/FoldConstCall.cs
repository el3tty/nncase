// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.Mutators.K230;

internal sealed class FoldConstCall : ExprRewriter
{
    protected override Expr RewriteLeafTuple(Nncase.IR.Tuple expr)
    {
        if (IsAllConst(expr.Fields))
        {
            return new TupleConst(new TupleValue(ArrayExtensions.AsValueEnumerable<Expr>(expr.Fields)
                .Select<IValue>((Func<Expr, IValue>)((Expr x) => Value.FromConst((Const)x))).ToArray()));
        }

        return expr;
    }

    protected override Expr RewriteLeafCall(Call expr)
    {
        Expr target = expr.Target;
        bool canFold = (target is Op op && op.CanFoldConstCall) || target is Function;
        if (canFold)
        {
            if (!IsAllConst(expr.Arguments))
            {
                return expr;
            }

            return Const.FromValue(expr.Evaluate());
        }

        return expr;
    }

    private bool IsAllConst(ReadOnlySpan<Expr> parameters)
    {
        return ArrayExtensions.AsValueEnumerable<Expr>(parameters).All((Func<Expr, bool>)((Expr e) => e is Const));
    }
}
