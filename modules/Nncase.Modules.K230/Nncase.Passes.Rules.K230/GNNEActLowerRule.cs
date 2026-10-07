// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;

namespace Nncase.Passes.Rules.K230;

public abstract class GNNEActLowerRule : GNNEDIFQuantRule
{
    public Func<Expr, Expr> WithTmpFloat(Func<Expr, Expr> inputCtor)
    {
        return Utility.WithTmpType(inputCtor, DataTypes.Float32);
    }
}
