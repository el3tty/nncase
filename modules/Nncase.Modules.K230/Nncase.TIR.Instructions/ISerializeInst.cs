// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.IR;

namespace Nncase.TIR.Instructions;

public interface ISerializeInst
{
    void Serialize(BinaryWriter writer, Call call);
}
