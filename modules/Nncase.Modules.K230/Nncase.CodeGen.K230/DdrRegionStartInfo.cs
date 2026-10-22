// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CodeGen.K230;

internal class DdrRegionStartInfo
{
    public uint Input { get; set; }

    public uint Output { get; set; }

    public uint Rdata { get; set; }

    public uint Data { get; set; }

    public uint Text { get; set; }
}
