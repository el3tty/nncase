// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.Passes.Analysis;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class DWToPdp : RewriteRule<Pattern>, IRewriteRule
{
    public override Pattern Pattern { get; } = Nncase.PatternMatch.F.K230.IsGNNEConv2D("dw", "dwCall",
        (GNNEConv2D _) => true,
        Nncase.PatternMatch.F.K230.IsGNNELoad("ld", "ldCall", (GNNELoad _) => true,
            Nncase.PatternMatch.F.K230.IsGNNEStore("st", "stCall", (GNNEStore _) => true,
                Nncase.PatternMatch.F.K230.IsGNNEConv2D("conv", "convCall", (GNNEConv2D _) => true,
                    Nncase.PatternMatch.F.K230.IsGNNELoad("convInput", "convInputCall", (GNNELoad _) => true), null,
                    null, null, null, null, null, null, null, null, null, null, null,
                    Nncase.PatternMatch.Utility.IsTensorConst("convGroups")))),
        Nncase.PatternMatch.F.K230.IsGNNELoadW("weights", "weightsCall", (GNNELoadW _) => true),
        Nncase.PatternMatch.Utility.IsWildcard("weightsBias"),
        Nncase.PatternMatch.Utility.IsWildcard("weightsBiasQint8"), Nncase.PatternMatch.Utility.IsWildcard("act"),
        Nncase.PatternMatch.Utility.IsWildcard("actQint8"), Nncase.PatternMatch.Utility.IsWildcard("deqBias"),
        Nncase.PatternMatch.Utility.IsWildcard("shiftBits"), Nncase.PatternMatch.Utility.IsWildcard("shiftBitsQint8"),
        Nncase.PatternMatch.Utility.IsWildcard("qint8Qp"), Nncase.PatternMatch.Utility.IsWildcard("padding"),
        Nncase.PatternMatch.Utility.IsWildcard("stride"), Nncase.PatternMatch.Utility.IsWildcard("dilation"),
        Nncase.PatternMatch.Utility.IsTensorConst("groups"), Nncase.PatternMatch.Utility.IsWildcard("is16Quant"),
        Nncase.PatternMatch.Utility.IsWildcard("padValue"), Nncase.PatternMatch.Utility.IsWildcard("weightsQInt8"));

    private Expr? GetReplace(Call dwCall, GNNEConv2D dw, Call ldCall, Call stCall, Call convCall, Call convInputCall,
        Expr convGroups, Call weightsCall, Expr weightsBias, Expr weightsBiasQint8, Expr act, Expr actQint8,
        Expr deqBias, Expr shiftBits, Expr shiftBitsQint8, Expr qint8Qp, Expr padding, Expr stride, Expr dilation,
        Expr groups, Expr is16Quant, Expr padValue, Expr weightsQInt8, RunPassContext context)
    {
        IExprUserAnalysisResult analysis = context.GetAnalysis<IExprUserAnalysisResult>();
        int fixedValue = ldCall.CheckedShape[1].FixedValue;
        int fixedValue2 = dwCall.CheckedShape[1].FixedValue;
        bool isDwConv = fixedValue == fixedValue2 && fixedValue2 == groups && groups != 1;
        int[] subArray = weightsCall.CheckedShape.ToValueArray()[2..];
        int fixedValue3 = convInputCall.CheckedShape[1].FixedValue;
        int fixedValue4 = convCall.CheckedShape[1].FixedValue;
        bool isConvDepthwise = fixedValue3 == fixedValue4 && fixedValue4 == convGroups && convGroups != 1;
        DataType inputDataType = ldCall.CheckedDataType;
        DataType weightsDataType = weightsCall.CheckedDataType;
        if (analysis[stCall].Count() <= 1 && analysis[ldCall].Count() <= 1 && isDwConv && subArray[0] <= 3 &&
            subArray[1] <= 3 && !isConvDepthwise && (inputDataType == DataTypes.UInt8 || inputDataType == DataTypes.Int8) &&
            (weightsDataType == DataTypes.UInt8 || weightsDataType == DataTypes.Int8))
        {
            int[] dwWeightShape = dwCall[GNNEConv2D.Weights].CheckedShape.ToValueArray();
            dwWeightShape[0] = TileUtilities.GetAlignedNum(dwWeightShape[0], GNNEEnv.PuWidth);
            byte[] weightBytes = ((TensorConst)weightsCall[GNNELoadW.Input]).Value.ToArray<byte>();
            byte[] paddedWeightBytes = Enumerable.Repeat((byte)0, dwWeightShape.Aggregate((int d0, int d1) => d0 * d1)).ToArray();
            Array.Copy(weightBytes, paddedWeightBytes, weightBytes.Length);
            weightsCall =
                Nncase.IR.K230.F.Tensors.GNNELoadW((PrimType)weightsDataType, Tensor.FromBytes<byte>(paddedWeightBytes, dwWeightShape));
            return Nncase.IR.K230.F.Tensors.GNNEPdp0DW(dw.DestType, ldCall, weightsCall, weightsQInt8, weightsBias,
                weightsBiasQint8, act, actQint8, deqBias, shiftBits, shiftBitsQint8, qint8Qp, padding, stride, dilation,
                groups, is16Quant, padValue, dw.ActParam, dw.ActParamQInt8);
        }

        return null;
    }

    public override Expr? GetReplace(IMatchResult __result, RunPassContext __context)
    {
        Call dwCall = (Call)__result["dwCall"];
        GNNEConv2D dw = (GNNEConv2D)__result["dw"];
        Call ldCall = (Call)__result["ldCall"];
        Call stCall = (Call)__result["stCall"];
        Call convCall = (Call)__result["convCall"];
        Call convInputCall = (Call)__result["convInputCall"];
        Expr convGroups = (Expr)__result["convGroups"];
        Call weightsCall = (Call)__result["weightsCall"];
        Expr weightsBias = (Expr)__result["weightsBias"];
        Expr weightsBiasQint = (Expr)__result["weightsBiasQint8"];
        Expr act = (Expr)__result["act"];
        Expr actQint = (Expr)__result["actQint8"];
        Expr deqBias = (Expr)__result["deqBias"];
        Expr shiftBits = (Expr)__result["shiftBits"];
        Expr shiftBitsQint = (Expr)__result["shiftBitsQint8"];
        Expr qint8Qp = (Expr)__result["qint8Qp"];
        Expr padding = (Expr)__result["padding"];
        Expr stride = (Expr)__result["stride"];
        Expr dilation = (Expr)__result["dilation"];
        Expr groups = (Expr)__result["groups"];
        Expr is16Quant = (Expr)__result["is16Quant"];
        Expr padValue = (Expr)__result["padValue"];
        Expr weightsQInt = (Expr)__result["weightsQInt8"];
        return GetReplace(dwCall, dw, ldCall, stCall, convCall, convInputCall, convGroups, weightsCall, weightsBias,
            weightsBiasQint, act, actQint, deqBias, shiftBits, shiftBitsQint, qint8Qp, padding, stride, dilation,
            groups, is16Quant, padValue, weightsQInt, __context);
    }
}
