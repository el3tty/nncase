// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Rules.Tile;

public sealed record TileOptions(int[] TargetTileSize, bool ForceFence = false);
