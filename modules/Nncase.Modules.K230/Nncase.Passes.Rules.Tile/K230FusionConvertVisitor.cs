// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.K230;
using Nncase.Passes.Rules.K230;
using Nncase.PatternMatch;
using Nncase.TIR;

namespace Nncase.Passes.Rules.Tile;

public sealed class K230FusionConvertVisitor : ExprRewriter
{
    private readonly RunPassContext _options;

    public K230FusionConvertVisitor(RunPassContext options)
    {
        _options = options;
    }

    protected override Expr RewriteLeafFusion(Fusion expr)
    {
        return Process(expr);
    }

    private PrimFunction Process(Fusion fusion)
    {
        IRewriteRule rewriteRule;
        if (fusion.Name.StartsWith("TileTransposeCase"))
        {
            rewriteRule = new TileTranspose();
        }
        else if (fusion.Name.StartsWith("TilePdp1Case"))
        {
            rewriteRule = new TilePdp1();
        }
        else if (fusion.Name.StartsWith("TileAct1Case"))
        {
            rewriteRule = new TileAct1();
        }
        else if (fusion.Name.StartsWith("TileConv2dCase"))
        {
            Call conv = ((Call)fusion.Body)[GNNEStore.Input] as Call;
            int[] inputShape = conv[GNNEConv2D.Input].CheckedShape.ToValueArray();
            int[] outputShape = conv.CheckedShape.ToValueArray();
            int[] weightsShape = conv[GNNEConv2D.Weights].CheckedShape.ToValueArray();
            int[] source = ((TensorConst)conv[GNNEConv2D.Padding]).Value.ToArray<int>();
            int strideH = ((TensorConst)conv[GNNEConv2D.Stride]).Value.ToArray<int>()[0];
            int strideW = ((TensorConst)conv[GNNEConv2D.Stride]).Value.ToArray<int>()[1];
            int dilationH = ((TensorConst)conv[GNNEConv2D.Dilation]).Value.ToArray<int>()[0];
            int dilationW = ((TensorConst)conv[GNNEConv2D.Dilation]).Value.ToArray<int>()[1];
            int groups = ((TensorConst)conv[GNNEConv2D.Groups]).Value.ToScalar<int>();
            rewriteRule =
                (inputShape[0] <= 1 || outputShape[2] != 1 || outputShape[3] != 1 || inputShape[2] != 1 || inputShape[3] != 1 ||
                  weightsShape[2] != 1 || weightsShape[3] != 1 || strideH != 1 || strideW != 1 || dilationH != 1 || dilationW != 1 ||
                  source.Sum() != 0 || groups != 1)
                    ? ((RewriteRule<Pattern>)new TileConv2D())
                    : ((RewriteRule<Pattern>)new TileTransposeMatmul());
        }
        else if (fusion.Name.StartsWith("TileConv2dTransposeCase"))
        {
            rewriteRule = new TileConv2dTranspose();
        }
        else if (fusion.Name.StartsWith("TilePadCase"))
        {
            rewriteRule = new TilePad();
        }
        else if (fusion.Name.StartsWith("TileResizeCase"))
        {
            rewriteRule = new TileResize();
        }
        else if (fusion.Name.StartsWith("TileMatMulCase"))
        {
            rewriteRule = new TileMatMul();
        }
        else if (fusion.Name.StartsWith("TileLSTMCase"))
        {
            rewriteRule = new TileLSTM();
        }
        else if (fusion.Name.StartsWith("TileLoadStoreCase"))
        {
            rewriteRule = new TileLoadStore();
        }
        else if (fusion.Name.StartsWith("TileConcatCase"))
        {
            rewriteRule = new TileConcat();
        }
        else
        {
            if (!fusion.Name.StartsWith("TileSplitCase"))
            {
                throw new NotImplementedException();
            }

            rewriteRule = new TileSplit();
        }

        if (!CompilerServices.TryMatchRoot(fusion, rewriteRule.Pattern, out IMatchResult result))
        {
            throw new NotImplementedException();
        }

        Expr replace = rewriteRule.GetReplace(result, _options);
        if ((object)replace == null)
        {
            return (PrimFunction)new TileConv2D().GetReplace(result, _options);
        }

        return (PrimFunction)replace;
    }
}
