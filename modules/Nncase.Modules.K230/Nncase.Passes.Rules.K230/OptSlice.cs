// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230.F;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class OptSlice : IRewriteRule
{
    public IPattern Pattern { get; } = Nncase.PatternMatch.F.Tensors.IsSlice("slice", "call", (Slice _) => true,
        Nncase.PatternMatch.Utility.IsWildcard("input") with { TypePattern = TypePatternUtility.HasFixedShape() },
        Nncase.PatternMatch.Utility.IsTensorConst("begin"), Nncase.PatternMatch.Utility.IsTensorConst("ends"),
        Nncase.PatternMatch.Utility.IsTensorConst("axes"), Nncase.PatternMatch.Utility.IsTensorConst("strides"));

    private Expr? GetReplace(Slice slice, Call call, Expr input, TensorConst begin, TensorConst ends, TensorConst axes,
        TensorConst strides)
    {
        if (!(strides == new int[4] { 1, 1, 1, 1 }))
        {
            Shape checkedShape = input.CheckedShape;
            if (checkedShape[checkedShape.Count - 1].FixedValue %
                strides.Value.ToArray<int>()[strides.Value.ToArray<int>().Length - 1] == 0 ||
                strides.Value.ToArray<int>()[strides.Value.ToArray<int>().Length - 1] == 1)
            {
                int[] strideValues;
                if (strides.Value.Length == 4)
                {
                    int stridedDimCount = 0;
                    strideValues = strides.Value.ToArray<int>();
                    for (int i = 0; i < strideValues.Length; i++)
                    {
                        if (strideValues[i] > 1 && ++stridedDimCount == 4)
                        {
                            return null;
                        }
                    }
                }

                strideValues = strides.CheckedShape.ToValueArray();
                for (int i = 0; i < strideValues.Length; i++)
                {
                    if (strideValues[i] < 0)
                    {
                        return null;
                    }
                }

                List<int> beginDims = new List<int>();
                List<int> endDims = new List<int>();
                List<int> strideDims = new List<int>();
                List<int> inputDims = new List<int>();
                for (int j = 0; j < input.CheckedShape.Count; j++)
                {
                    if (input.CheckedShape[j].FixedValue > 1)
                    {
                        beginDims.Add(begin.Value.ToArray<int>()[j]);
                        endDims.Add(ends.Value.ToArray<int>()[j]);
                        strideDims.Add(strides.Value.ToArray<int>()[j]);
                        inputDims.Add(input.CheckedShape[j].FixedValue);
                    }
                }

                if (strides.Value.ToArray<int>()[3] > 1)
                {
                    int lastBegin = beginDims[beginDims.Count - 1];
                    int lastStride = strideDims[strideDims.Count - 1];
                    int lastEnd = endDims[endDims.Count - 1];
                    int lastInputDim = inputDims[inputDims.Count - 1];
                    if (beginDims.Count != 0)
                    {
                        beginDims.RemoveAt(beginDims.Count - 1);
                    }

                    if (strideDims.Count != 0)
                    {
                        strideDims.RemoveAt(strideDims.Count - 1);
                    }

                    if (endDims.Count != 0)
                    {
                        endDims.RemoveAt(endDims.Count - 1);
                    }

                    if (inputDims.Count != 0)
                    {
                        inputDims.RemoveAt(inputDims.Count - 1);
                    }

                    beginDims.Add(lastBegin / lastStride);
                    beginDims.Add(lastBegin % lastStride);
                    strideDims.Add(1);
                    strideDims.Add(1);
                    endDims.Add(lastEnd / lastStride + ((lastEnd % lastStride > lastBegin % lastStride) ? 1 : 0));
                    endDims.Add(lastBegin % lastStride + 1);
                    inputDims.Add(lastInputDim / lastStride);
                    inputDims.Add(lastStride);
                }

                for (int k = inputDims.Count; k < 4; k++)
                {
                    beginDims.Insert(beginDims[0], 0);
                    endDims.Insert(endDims[0], 1);
                    strideDims.Insert(strideDims[0], 1);
                }

                Call input2 = Nncase.IR.F.Tensors.Slice(
                    Nncase.IR.F.Tensors.Reshape(input, Enumerable.ToArray(inputDims)),
                    beginDims.ToArray(), endDims.ToArray(), new int[4], new int[4] { 1, 1, 1, 1 });
                DataType storeType = (call.CheckedDataType == DataTypes.Float32) ? DataTypes.Float16 : call.CheckedDataType;
                return Nncase.IR.F.Tensors.Reshape(
                    Nncase.IR.K230.F.Tensors.GNNEStore(storeType, Nncase.IR.K230.F.Tensors.GNNELoad((PrimType)storeType, input2)),
                    call.CheckedShape);
            }
        }

        return null;
    }

    public Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        Slice slice = (Slice)result["slice"];
        Call call = (Call)result["call"];
        Expr input = (Expr)result["input"];
        TensorConst begin = (TensorConst)result["begin"];
        TensorConst ends = (TensorConst)result["ends"];
        TensorConst axes = (TensorConst)result["axes"];
        TensorConst strides = (TensorConst)result["strides"];
        return GetReplace(slice, call, input, begin, ends, axes, strides);
    }
}
