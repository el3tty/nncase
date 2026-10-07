// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Rules.K230;

public class GnneActionCcrClr : GnneAction
{
    public int Ccr { get; }

    public GnneActionCcrClr(int ccr)
        : base(GnneActionName.CcrClr)
    {
        Ccr = ccr;
    }
}
