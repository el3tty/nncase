// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.Rules.Tile;

internal interface IFusionChecker
{
    bool Check(Fusion fusion, RunPassContext passOptions);

    PrimFunction Convert();
}
