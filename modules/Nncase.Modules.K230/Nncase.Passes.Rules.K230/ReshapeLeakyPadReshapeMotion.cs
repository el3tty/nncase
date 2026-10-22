// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class ReshapeLeakyPadReshapeMotion : IRewriteRule
{
    public IPattern Pattern { get; } = Nncase.PatternMatch.F.Tensors.IsReshape("bc2", "call", (Reshape _) => true,
        Nncase.PatternMatch.F.NN.IsPad("p", (Pad _) => true, Nncase.PatternMatch.F.Math.IsBinary("bn2", BinaryOp.Max,
                Nncase.PatternMatch.F.Math.IsBinary("bn1", BinaryOp.Mul, Nncase.PatternMatch.F.Tensors.IsReshape(
                    "bc1",
                    (Reshape _) => true,
                    Nncase.PatternMatch.Utility.IsTensorConst("input") with
                        {
                            TypePattern = TypePatternUtility.HasFixedShape(),
                        }),
                    Nncase.PatternMatch.Utility.IsTensorConst("rhs") with
                    {
                        TypePattern = TypePatternUtility.HasFixedShape(),
                    })), Nncase.PatternMatch.Utility.IsTensorConst("pads"),
            Nncase.PatternMatch.Utility.IsTensorConst("value")));

    private Expr? GetReplace(Reshape bc2, Call call, Pad p, Binary bn2, Binary bn1, Reshape bc1, Expr rhs,
        TensorConst pads, TensorConst value, TensorConst input)
    {
        if (call.CheckedShape[1] != p.CheckedShape[0] || call.CheckedShape[2] != p.CheckedShape[1] ||
            call.CheckedShape[3] != p.CheckedShape[2])
        {
            return null;
        }

        Shape checkedShape = call.CheckedShape;
        Call input2 = Nncase.IR.F.Tensors.Reshape(
            bc1,
            new Dimension[4] { 1, checkedShape[0], checkedShape[1], checkedShape[2] });
        Tensor<int> tensor = pads.Value.Cast<int>();
        int[][] paddings = new int[4][];
        for (int i = 0; i < 4; i++)
        {
            if (i != 0)
            {
                for (int j = 0; j < tensor.Length; j++)
                {
                    paddings[i][j] = tensor[new int[2] { i - 1, j }];
                }
            }
        }

        return Nncase.IR.F.NN.Pad(input2, paddings, p.PadMode, value);
    }

    public Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Reshape bc = (Reshape)result["bc2"];
        Call call = (Call)result["call"];
        Pad p = (Pad)result["p"];
        Binary bn = (Binary)result["bn2"];
        Binary bn2 = (Binary)result["bn1"];
        Reshape bc2 = (Reshape)result["bc1"];
        Expr rhs = (Expr)result["rhs"];
        TensorConst pads = (TensorConst)result["pads"];
        TensorConst value = (TensorConst)result["value"];
        TensorConst input = (TensorConst)result["input"];
        return GetReplace(bc, call, p, bn, bn2, bc2, rhs, pads, value, input);
    }
}
