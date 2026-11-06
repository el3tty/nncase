// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using Nncase.IR;

namespace Nncase.TIR.Instructions;

public sealed class LoadImm : InstructionOp, ISerializeInst, IEquatable<LoadImm?>
{
    public static readonly ParameterInfo Value =
        new ParameterInfo(typeof(LoadImm), 0, "value", TypePatternUtility.IsIntegralScalar());

    public GP_REGISTER Rd { get; }

    public void Serialize(BinaryWriter writer, Call call)
    {
        uint imm = ((TensorConst)call[Value]).Value.ToScalar<uint>();
        if (Rd != GP_REGISTER.x0 && Rd != GP_REGISTER.invalid)
        {
            if (imm < 2048 && imm != 0)
            {
                Call call2 = I.ADDI(Rd, R.r0, imm & 0x7FF);
                ((ADDI)call2.Target).Serialize(writer, call2);
                return;
            }

            uint high = ((imm >> 12) + ((imm >> 11) & 1)) & 0xFFFFF;
            uint low = imm & 0xFFF;
            Call call3 = I.LUI(Rd, high);
            Call call4 = I.ADDI(Rd, Rd, low);
            ((LUI)call3.Target).Serialize(writer, call3);
            ((ADDI)call4.Target).Serialize(writer, call4);
        }
    }

    public LoadImm(GP_REGISTER rd)
    {
        Rd = rd;
    }

    public LoadImm With(GP_REGISTER? rd = null)
    {
        return new LoadImm(rd ?? Rd);
    }

    public override bool Equals(object? obj)
    {
        return Equals(obj as LoadImm);
    }

    public bool Equals(LoadImm? other)
    {
        if ((object)this == other)
        {
            return true;
        }

        if ((object)other != null && Equals((InstructionOp?)other))
        {
            return Rd.Equals(other.Rd);
        }

        return false;
    }

    protected override int GetHashCodeCore()
    {
        return HashCode.Combine(base.GetHashCodeCore(), HashCode.Combine(Rd));
    }
}
