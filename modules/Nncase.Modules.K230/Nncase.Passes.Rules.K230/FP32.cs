// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;

namespace Nncase.Passes.Rules.K230;

[StructLayout(LayoutKind.Explicit)]
public struct FP32
{
    [FieldOffset(0)] public uint U;

    [FieldOffset(0)] public float F;
}
