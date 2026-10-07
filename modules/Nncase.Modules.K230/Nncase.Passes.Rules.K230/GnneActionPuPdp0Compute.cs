// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.Rules.K230;

public class GnneActionPuPdp0Compute : GnneAction
{
    public int TcuId { get; }

    public Gpr AddrS { get; }

    public GnneActionPuPdp0Compute(int tcuId, Gpr addrS)
        : base(GnneActionName.PuPdp0Compute)
    {
        TcuId = tcuId;
        AddrS = addrS;
    }
}
