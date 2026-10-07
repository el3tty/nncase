// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using OrtKISharp;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class GNNEPadEvaluator : IEvaluator<GNNEPad>, IEvaluator, ITypeInferencer<GNNEPad>, ITypeInferencer,
    ICostEvaluator<GNNEPad>, ICostEvaluator
{
    // ONNX TensorProto data type code of float16, the target of the final Cast.
    private const long OnnxFloat16 = 10L;

    public Cost Visit(ICostEvaluateContext context, GNNEPad target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    public IValue Visit(IEvaluateContext context, GNNEPad p)
    {
        OrtKISharp.Tensor input = context.GetArgumentValue(p, GNNEPad.Input).AsTensor().Cast<float>()
            .ToOrtTensor();
        OrtKISharp.Tensor pads = context.GetInt64OrtTensorArgumentValue(p, GNNEPad.Pads);
        OrtKISharp.Tensor padValue = context.GetArgumentValue(p, GNNEPad.Value).AsTensor().Cast<float>()
            .ToOrtTensor();

        // Constant-mode pad in float, then cast the result to float16.
        return OrtKI
            .Cast(
                OrtKI.Pad(input, (OrtKISharp.Tensor)EvaluatorUtil.ToOnnxPadFormat(pads), padValue, null, "constant"),
                1, OnnxFloat16).ToValue();
    }

    public IRType Visit(ITypeInferenceContext context, GNNEPad target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, GNNEPad.Input);
        context.CheckArgumentType<IRType>(target, GNNEPad.Input);
        context.CheckArgumentType<IRType>(target, GNNEPad.Pads);
        context.CheckArgumentType<IRType>(target, GNNEPad.Value);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, GNNEPad target, TensorType input)
    {
        Expr pads = context.GetArgument(target, GNNEPad.Pads);
        Expr padValue = context.GetArgument(target, GNNEPad.Value);
        return TypeInference.PadType(input, pads, padValue);
    }
}
