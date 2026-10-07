// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Rules.K230;

public class GnneActionIntr : GnneAction
{
    public Gpr IntrNum { get; }

    public GnneActionIntr(Gpr intrNum)
        : base(GnneActionName.Intr)
    {
        IntrNum = intrNum;
    }
}
