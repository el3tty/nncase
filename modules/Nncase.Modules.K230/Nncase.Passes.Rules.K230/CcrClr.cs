// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Rules.K230;

public class CcrClr
{
    public int Ccr { get; set; }

    public int Ccrclr { get; set; }

    public CcrClr(int ccr)
    {
        Ccr = ccr;
        Ccrclr = 1;
    }
}
