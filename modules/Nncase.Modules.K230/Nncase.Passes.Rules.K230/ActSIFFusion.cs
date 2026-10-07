// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.K230;
using Nncase.PatternMatch;

namespace Nncase.Passes.Rules.K230;

public sealed class ActSIFFusion : GNNESingleInputFusion<GNNEActivation>
{
    public override string Name { get; } = "TileAct1Case";

    public override Pattern Pattern { get; } = Nncase.PatternMatch.Utility.IsCallWildcard("endCall",
        Nncase.PatternMatch.Utility.IsOp<GNNEStore>("endOp"),
        Nncase.PatternMatch.Utility.IsCallWildcard("midCall", Nncase.PatternMatch.Utility.IsOp<GNNEActivation>("midOp"),
            Nncase.PatternMatch.Utility.IsCallWildcard("beginCall",
                Nncase.PatternMatch.Utility.IsOp<GNNELoad>("beginOp"), Nncase.PatternMatch.Utility.IsWildcard("input")),
            Nncase.PatternMatch.Utility.IsNone()));
}
