// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.K230;
using Nncase.Passes.Rules.Neutral;

namespace Nncase.Passes.Rules.K230;

public class GNNEDoubleInputFusion<T> : DoubleInputFusion<T, GNNELoad, GNNEStore> where T : Op
{
    public override string ModuleKind { get; } = "k230";
}
