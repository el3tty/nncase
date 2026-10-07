// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;

namespace Nncase.Runtime.K230;

public class K230RtModule : RTModule
{
    public static readonly string Kind = "k230";

    public static readonly uint Version = 1u;

    public K230RtModule(IReadOnlyList<IRTFunction> functions)
        : base(functions)
    {
    }
}
