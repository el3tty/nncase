// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.K230;

namespace Nncase.Passes.Rules.K230;

public sealed class Pdp0DwFusion : GNNESingleInputFusion<GNNEPdp0DW>
{
    public override string Name { get; } = "TilePdp0DwCase";
}
