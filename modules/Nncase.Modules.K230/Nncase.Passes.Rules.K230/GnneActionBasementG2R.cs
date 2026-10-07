// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Rules.K230;

public class GnneActionBasementG2R : GnneAction
{
    private int _rbasement;

    public int Rbasement => _rbasement;

    public GnneActionBasementG2R(int rbasement)
        : base(GnneActionName.BasementG2R)
    {
        _rbasement = rbasement;
    }
}
