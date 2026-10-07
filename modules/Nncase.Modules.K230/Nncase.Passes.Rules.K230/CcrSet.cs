// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Rules.K230;

public class CcrSet
{
    public int Ccr { get; set; }

    public int Value { get; set; }

    public CcrSet(int ccr, int value)
    {
        Ccr = ccr;
        Value = value;
    }
}
