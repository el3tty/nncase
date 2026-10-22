// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class Expand1x1Conv : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = Nncase.PatternMatch.F.K230.IsGNNEConv2D("conv", "convCall",
        (GNNEConv2D _) => true, Nncase.PatternMatch.F.K230.IsGNNELoad("ld", "ldCall", (GNNELoad _) => true),
        Nncase.PatternMatch.F.K230.IsGNNELoadW("lw", "lwCall", (GNNELoadW _) => true),
        Nncase.PatternMatch.F.K230.IsGNNELoadW("lwBias", "lwBiasCall", (GNNELoadW _) => true),
        Nncase.PatternMatch.Utility.IsWildcard("weightsBiasQint8"), Nncase.PatternMatch.Utility.IsWildcard("act"),
        Nncase.PatternMatch.Utility.IsWildcard("actQint8"), Nncase.PatternMatch.Utility.IsWildcard("deqBias"),
        Nncase.PatternMatch.Utility.IsWildcard("shiftBits"), Nncase.PatternMatch.Utility.IsWildcard("shiftBitsQint8"),
        Nncase.PatternMatch.Utility.IsWildcard("qint8Qp"), Nncase.PatternMatch.Utility.IsWildcard("padding"),
        Nncase.PatternMatch.Utility.IsWildcard("stride"), Nncase.PatternMatch.Utility.IsWildcard("dilation"),
        Nncase.PatternMatch.Utility.IsTensorConst("groups"), Nncase.PatternMatch.Utility.IsWildcard("is16Quant"),
        Nncase.PatternMatch.Utility.IsWildcard("padValue"), Nncase.PatternMatch.Utility.IsWildcard("weightsQInt8"));

    private Expr? GetReplace(GNNEConv2D conv, Call ldCall, Call convCall, Call lwCall, Call lwBiasCall,
        Expr weightsBiasQint8, Expr act, Expr actQint8, Expr deqBias, Expr shiftBits, Expr shiftBitsQint8, Expr qint8Qp,
        Tensor<int> padding, int[] stride, int[] dilation, int groups, Expr is16Quant, Expr padValue, Expr weightsQInt8)
    {
        if (!GNNETypePatternUtility.IsDepthWise(ldCall, lwCall, groups) &&
            (ldCall.CheckedDataType == DataTypes.UInt8 || ldCall.CheckedDataType == DataTypes.Int8) &&
            (lwCall.CheckedDataType == DataTypes.UInt8 || lwCall.CheckedDataType == DataTypes.Int8) &&
            (ldCall.CheckedShape[0].FixedValue == 1 || ldCall.CheckedShape[2].FixedValue != 1 ||
             ldCall.CheckedShape[3].FixedValue != 1) && lwCall.CheckedShape[2].FixedValue == 1 &&
            lwCall.CheckedShape[3].FixedValue == 1 && stride[0] == 1 && stride[1] == 1 && dilation[0] == 1 &&
            dilation[1] == 1 && padding.Sum() == 0 && groups == 1 &&
            ldCall.CheckedShape[2].FixedValue * ldCall.CheckedShape[3].FixedValue <= 512 &&
            lwCall.CheckedShape[1].FixedValue > 24 && lwCall.CheckedShape[1].FixedValue % 24 != 0)
        {
            byte[] weightBytes = ((TensorConst)lwCall[GNNELoadW.Input]).Value.BytesBuffer.ToArray();
            byte[] bias = ((TensorConst)lwBiasCall[GNNELoadW.Input]).Value.BytesBuffer.ToArray();
            int padChannels = (24 - lwCall.CheckedShape[1].FixedValue % 24) % 24;
            int elementScale = (!(lwCall.CheckedDataType == DataTypes.Float16)) ? 1 : 2;
            int bytesPerOutChannel = weightBytes.Length / lwCall.CheckedShape[0].FixedValue;
            int paddedBytesPerOutChannel = bytesPerOutChannel / lwCall.CheckedShape[1].FixedValue * (lwCall.CheckedShape[1].FixedValue + padChannels);
            byte[] newWeights = new byte[paddedBytesPerOutChannel * lwCall.CheckedShape[0].FixedValue];
            for (int i = 0; i < lwCall.CheckedShape[0].FixedValue; i++)
            {
                byte[] biasPadding = Enumerable.Repeat(bias[i], padChannels * elementScale).ToArray();
                byte[] channelBytes = new byte[paddedBytesPerOutChannel];
                Array.Copy(weightBytes, i * bytesPerOutChannel, channelBytes, 0, bytesPerOutChannel);
                Array.Copy(biasPadding, 0, channelBytes, bytesPerOutChannel, biasPadding.Length);
                Array.Copy(channelBytes, 0, newWeights, i * paddedBytesPerOutChannel, channelBytes.Length);
            }

            int[] newShape = lwCall.CheckedShape.ToValueArray();
            newShape[1] += padChannels;
            Call weights = Nncase.IR.K230.F.Tensors.GNNELoadW(
                (PrimType)lwCall.CheckedDataType,
                Tensor.FromBytes(new TensorType(lwCall.CheckedDataType, newShape), newWeights));
            return Nncase.IR.K230.F.Tensors.GNNEConv2D((PrimType)convCall.CheckedDataType, ldCall, weights,
                weightsQInt8, lwBiasCall, weightsBiasQint8, act, actQint8, deqBias, shiftBits, shiftBitsQint8, qint8Qp,
                padding, stride, dilation, groups, is16Quant, padValue, conv.ActParam, conv.ActParamQInt8);
        }

        return null;
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        GNNEConv2D conv = (GNNEConv2D)result["conv"];
        Call ldCall = (Call)result["ldCall"];
        Call convCall = (Call)result["convCall"];
        Call lwCall = (Call)result["lwCall"];
        Call lwBiasCall = (Call)result["lwBiasCall"];
        Expr weightsBiasQint = (Expr)result["weightsBiasQint8"];
        Expr act = (Expr)result["act"];
        Expr actQint = (Expr)result["actQint8"];
        Expr deqBias = (Expr)result["deqBias"];
        Expr shiftBits = (Expr)result["shiftBits"];
        Expr shiftBitsQint = (Expr)result["shiftBitsQint8"];
        Expr qint8Qp = (Expr)result["qint8Qp"];
        Tensor<int> padding = ((TensorConst)result["padding"]).Value.Cast<int>();
        int[] stride = ((TensorConst)result["stride"]).Value.ToArray<int>();
        int[] dilation = ((TensorConst)result["dilation"]).Value.ToArray<int>();
        int groups = ((TensorConst)result["groups"]).Value.ToScalar<int>();
        Expr is16Quant = (Expr)result["is16Quant"];
        Expr padValue = (Expr)result["padValue"];
        Expr weightsQInt = (Expr)result["weightsQInt8"];
        return GetReplace(conv, ldCall, convCall, lwCall, lwBiasCall, weightsBiasQint, act, actQint, deqBias, shiftBits,
            shiftBitsQint, qint8Qp, padding, stride, dilation, groups, is16Quant, padValue, weightsQInt);
    }
}
