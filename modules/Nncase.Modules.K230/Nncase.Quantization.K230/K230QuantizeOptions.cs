// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Quantization.K230;

public sealed class K230QuantizeOptions : QuantizeOptions
{
    private PrimType _quantType;

    private PrimType _wQuantType;

    public K230QuantizeOptions(PrimType quantType, PrimType wQuantType)
    {
        _quantType = quantType;
        _wQuantType = wQuantType;
    }
}
