// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;

namespace Nncase.Passes.Rules.K230;

[StructLayout(LayoutKind.Sequential, Size = 1)]
public struct Ai2dSram
{
    public static int SramLen { get; } = 256;

    public static int SramSize { get; } = SramLen * SramLen;
}
