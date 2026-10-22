// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.PatternMatch;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class Reshape1x1Conv : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        Nncase.PatternMatch.Utility.IsCallWildcard("fusionCall", FusionPattern.IsGNNEFusion<GNNEConv2D>());

    private Expr? GetReplace(Call call, Call ld, Call st, Call fusionCall, IReadOnlyList<Expr> fusionCallParams)
    {
        Expr input = call[GNNEConv2D.Input];
        Expr expr = call[GNNEConv2D.Weights];
        Expr weightsBias = call[GNNEConv2D.WeightsBias];
        Expr weightsBiasQint = call[GNNEConv2D.WeightsBiasQint8];
        Expr act = call[GNNEConv2D.Act];
        Expr actQint = call[GNNEConv2D.ActQint8];
        Expr deqBias = call[GNNEConv2D.DeqBias];
        Expr shiftBits = call[GNNEConv2D.ShiftBits];
        Expr shiftBitsQint = call[GNNEConv2D.ShiftBitsQint8];
        Expr qint8Qp = call[GNNEConv2D.Qint8Qp];
        Expr expr2 = call[GNNEConv2D.Padding];
        Expr expr3 = call[GNNEConv2D.Stride];
        Expr dilation = call[GNNEConv2D.Dilation];
        Expr expr4 = call[GNNEConv2D.Groups];
        Expr is16Quant = call[GNNEConv2D.Is16Quant];
        Expr padValue = call[GNNEConv2D.PadValue];
        Expr weightsQInt = call[GNNEConv2D.WeightsQInt8];
        GNNEConv2D gNNEConv2D = (GNNEConv2D)call.Target;
        Fusion fusion = (Fusion)fusionCall.Target;
        if ((input.CheckedShape[0].FixedValue == 1 || input.CheckedShape[2].FixedValue != 1 ||
             input.CheckedShape[3].FixedValue != 1) && ((TensorConst)expr4).Value.ToScalar<int>() == 1 &&
            ((TensorConst)expr3).Value.ToArray<int>().All((int x) => x == 1) &&
            expr.CheckedShape.ToValueArray()[2..].All((int x) => x == 1) &&
            ((TensorConst)expr2).Value.ToArray<int>().All((int x) => x == 0) &&
            (input.CheckedDataType == DataTypes.UInt8 || input.CheckedDataType == DataTypes.Int8) &&
            ((input.CheckedShape[1].FixedValue == expr.CheckedShape[1].FixedValue &&
              new int[3] { 24, 20, 16 }.Any((int x) =>
                  input.CheckedShape[1].Value % x == 0 && input.CheckedShape[1].Value != x)) ||
             input.CheckedShape[1].FixedValue != expr.CheckedShape[1].FixedValue) &&
            input.CheckedShape[2].FixedValue * input.CheckedShape[3].FixedValue < 65536)
        {
            int[] channelAlignments = new int[4] { 24, 20, 16, 0 };
            channelAlignments[3] = input.CheckedShape[1].FixedValue;
            int channelAlignment = channelAlignments.FirstOrDefault((int d) => input.CheckedShape[1].FixedValue % d == 0);
            int elementScale = (!(expr.CheckedDataType == DataTypes.Int16)) ? 1 : 2;
            int fixedValue = (expr.CheckedShape[0] * input.CheckedShape[1] * expr.CheckedShape[2] *
                              expr.CheckedShape[3] * elementScale).FixedValue;
            byte[] weightBytes = ((TensorConst)((Call)expr)[GNNELoadW.Input]).Value.BytesBuffer.ToArray();
            byte[] reshapedWeights = new byte[fixedValue];
            for (int index = 0; index < fixedValue; index++)
            {
                reshapedWeights[index] =
                    weightBytes[
                        index / (input.CheckedShape[1].FixedValue * elementScale) * expr.CheckedShape[1].FixedValue * elementScale +
                        index % (input.CheckedShape[1].FixedValue * elementScale)];
            }

            Shape shape = new Shape(expr.CheckedShape[0], channelAlignment, input.CheckedShape[1] / channelAlignment, 1);
            Call weights = Nncase.IR.K230.F.Tensors.GNNELoadW(
                (PrimType)expr.CheckedDataType,
                Tensor.FromBytes(new TensorType(expr.CheckedDataType, shape), reshapedWeights));
            Shape shape2 = new Shape(input.CheckedShape[0], channelAlignment, input.CheckedShape[1] / channelAlignment,
                input.CheckedShape[2] * input.CheckedShape[3]);
            Call call2 = Nncase.IR.F.Tensors.Reshape(fusionCallParams[0], shape2);
            call2.CheckedType = new TensorType(input.CheckedDataType, shape2);
            Var var = new Var("input", new TensorType(call2.CheckedDataType, call2.CheckedShape));
            Call input2 = Nncase.IR.K230.F.Tensors.GNNELoad((PrimType)ld.CheckedDataType, var);
            Call input3 = Nncase.IR.K230.F.Tensors.GNNEConv2D(gNNEConv2D.DestType, input2, weights, weightsQInt,
                weightsBias, weightsBiasQint, act, actQint, deqBias, shiftBits, shiftBitsQint, qint8Qp, expr2, expr3,
                dilation, expr4, is16Quant, padValue, gNNEConv2D.ActParam, gNNEConv2D.ActParamQInt8);
            Call body = Nncase.IR.K230.F.Tensors.GNNEStore(st.CheckedDataType, input3);
            return Nncase.IR.F.Tensors.Reshape(
                new Call(new Fusion(fusion.Name, fusion.ModuleKind, body, var), call2),
                st.CheckedShape);
        }

        return null;
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Call call = (Call)result["call"];
        Call ld = (Call)result["ld"];
        Call st = (Call)result["st"];
        Call fusionCall = (Call)result["fusionCall"];
        IReadOnlyList<Expr> fusionCallParams = (IReadOnlyList<Expr>)result["fusionCallParams"];
        return GetReplace(call, ld, st, fusionCall, fusionCallParams);
    }
}
