// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;

namespace Nncase.Passes.Rules.K230;

internal class TileSplitGlb : TiledGlb
{
    private readonly TensorOnGlb _ifGlb;

    private readonly TensorOnGlb _ofGlb;

    public TileSplitGlb(Dictionary<ItemName, TensorOnGlb> glbMap, Dictionary<ItemName, MmuItem> items,
        int[] lastOutShape, int nPingPongSplit, TensorOnGlb ifGlb, TensorOnGlb ofGlb)
        : base(glbMap, items, lastOutShape, nPingPongSplit)
    {
        _ifGlb = ifGlb;
        _ofGlb = ofGlb;
    }
}
