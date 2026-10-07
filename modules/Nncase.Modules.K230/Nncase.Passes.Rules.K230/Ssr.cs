// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Rules.K230;

public struct Ssr(int index, long value, bool needRenewal)
{
    public int Index { get; set; } = index;

    public long Value { get; set; } = value;

    public bool NeedRenewal { get; set; } = needRenewal;
}
