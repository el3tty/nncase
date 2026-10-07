// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Rules.K230;

public class GnneActionCcrDecl : GnneAction
{
    public Gpr Num { get; }

    public GnneActionCcrDecl(Gpr num)
        : base(GnneActionName.CcrDecl)
    {
        Num = num;
    }
}
