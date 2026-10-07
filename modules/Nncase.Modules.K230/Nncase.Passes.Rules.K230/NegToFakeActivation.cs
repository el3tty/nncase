// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.K230;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public class NegToFakeActivation : UnaryToFakeActivation
{
    public override UnaryOp Op => UnaryOp.Neg;

    public override void ProcessActParam(ActParam2 actParam, int i)
    {
        actParam.Ks[0, i] = (actParam.Ks[1, i] = -1f);
    }
}
