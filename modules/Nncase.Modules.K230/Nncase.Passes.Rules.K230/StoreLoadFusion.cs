// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.K230;
using Nncase.Passes.Rules.Neutral;

namespace Nncase.Passes.Rules.K230;

public sealed class StoreLoadFusion : DataTransferFusion<GNNELoad, GNNEStore>
{
    public override string ModuleKind { get; } = "k230";
}
