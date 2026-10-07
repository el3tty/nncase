using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K230;
using OrtKISharp;

namespace Nncase.Evaluator.K230;

[TypeInferGenerator]
public class Ai2dPadEvaluator : IEvaluator<Ai2dPad>, IEvaluator, ITypeInferencer<Ai2dPad>, ITypeInferencer,
    ICostEvaluator<Ai2dPad>, ICostEvaluator
{
    // ONNX TensorProto data type codes used as the target of the final Cast.
    private const long OnnxUInt8 = 2L;
    private const long OnnxInt8 = 3L;
    private const long OnnxInt16 = 5L;
    private const long OnnxFloat16 = 10L;

    public IValue Visit(IEvaluateContext context, Ai2dPad r)
    {
        Tensor input = context.GetArgumentValue(r, Ai2dPad.Input).AsTensor();
        OrtKISharp.Tensor padding = context.GetInt64OrtTensorArgumentValue(r, Ai2dPad.Padding);
        Tensor padValue = context.GetArgumentValue(r, Ai2dPad.Value).AsTensor();
        float[] inputData = input.ToArray<float>();
        float[] padValueData = padValue.ToArray<float>();

        // The pad is always done in float; the result is cast to the output type (float16 when not listed).
        long onnxOutputType;
        if (r.OutputType == DataTypes.UInt8)
        {
            onnxOutputType = OnnxUInt8;
        }
        else if (r.OutputType == DataTypes.Int8)
        {
            onnxOutputType = OnnxInt8;
        }
        else if (r.OutputType == DataTypes.Int16)
        {
            onnxOutputType = OnnxInt16;
        }
        else
        {
            onnxOutputType = OnnxFloat16;
        }

        // NOTE: string.ToLower(null) is kept as decompiled (it looks like it should be ToLower()).
        return OrtKI
            .Cast(
                OrtKI.Pad(
                    MakeOrtTensor(inputData, input), (OrtKISharp.Tensor)EvaluatorUtil.ToOnnxPadFormat(padding),
                    MakeOrtTensor(padValueData, padValue), null, r.Mode.ToString().ToLower(null)), 1,
                onnxOutputType).ToValue();
    }

    public Cost Visit(ICostEvaluateContext context, Ai2dPad target)
    {
        return new Cost { [CostFactorNames.CPUCycles] = (byte)1 };
    }

    public IRType Visit(ITypeInferenceContext context, Ai2dPad target)
    {
        TensorType input = context.CheckArgumentType<TensorType>(target, Ai2dPad.Input);
        context.CheckArgumentType<IRType>(target, Ai2dPad.Input);
        context.CheckArgumentType<IRType>(target, Ai2dPad.Padding);
        context.CheckArgumentType<IRType>(target, Ai2dPad.Value);
        context.CheckArgumentType<IRType>(target, Ai2dPad.InDeqBias);
        context.CheckArgumentType<IRType>(target, Ai2dPad.OutQuantParam);
        return Visit(context, target, input);
    }

    /// <summary>Wraps float data into an ORT tensor with the dimensions of <paramref name="shapeSource"/>.</summary>
    private static OrtKISharp.Tensor MakeOrtTensor(float[] data, Tensor shapeSource)
    {
        return OrtKISharp.Tensor.MakeTensor(
            data,
            ((IEnumerable<int>)shapeSource.Dimensions.ToArray()).Select((Func<int, long>)((int i) => i)).ToArray());
    }

    private IRType Visit(ITypeInferenceContext context, Ai2dPad target, TensorType input)
    {
        Expr padding = context.GetArgument(target, Ai2dPad.Padding);
        Expr padValue = context.GetArgument(target, Ai2dPad.Value);
        return TypeInference.PadType(input, padding, padValue);
    }
}
