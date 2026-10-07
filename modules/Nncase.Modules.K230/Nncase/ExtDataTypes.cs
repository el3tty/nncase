// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase;

public static class ExtDataTypes
{
    public static readonly ValueType QuantParam = new QuantizeParamType();

    public static readonly ValueType DeQuantParam = new DeQuantizeParamType();

    public static readonly ValueType CropBBox = new CropBBoxType();
}
