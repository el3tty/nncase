// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Passes.Mutators;

namespace Nncase.Passes.Rules.Tile;

internal sealed class GNNEShortCutFusionMergeRuleLeft : ShortCutFusionMergeRuleLeft
{
    public override string ModuleKind => "k230";
}
