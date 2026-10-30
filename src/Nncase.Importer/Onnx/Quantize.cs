// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using Onnx;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitQuantizeLinear(in NodeProto op)
        {
            var (input, scale) = GetInputExprs(op, 0, 1);
            var bias = GetOptionInputExpr(op, 2, 0);
            if (scale is TensorConst scaleConst && bias is TensorConst biasConst)
            {
                CheckPerTensorQuantParam(op, scaleConst);
                var param = new QuantParam(biasConst.Value.ToScalar<int>(), scaleConst.Value.ToScalar<float>());
                _quantizedTensors[op.Output[0]] = (param, biasConst.CheckedDataType);
                return Quantize(input, param, biasConst.CheckedDataType);
            }

            throw new NotImplementedException("Onnx importer not impl for dynamic scale and bias");
        }

        private Expr VisitDequantizeLinear(in NodeProto op)
        {
            var (input, scale) = GetInputExprs(op, 0, 1);
            var bias = GetOptionInputExpr(op, 2, 0);
            if (scale is TensorConst scaleConst && bias is TensorConst biasConst)
            {
                CheckPerTensorQuantParam(op, scaleConst);
                var scaleV = scaleConst.Value.ToScalar<float>();
                var biasV = biasConst.Value.ToScalar<int>();
                return Dequantize(
                    input,
                    new QuantParam(biasV, scaleV),
                    DataTypes.Float32);
            }

            throw new NotImplementedException("Onnx importer not impl for dynamic scale and bias");
        }

        private static void CheckPerTensorQuantParam(in NodeProto op, TensorConst scale)
        {
            if (scale.Value.Length != 1)
            {
                throw new NotSupportedException($"{op.OpType} {op.Name}: per-axis quantization is not supported (IR Quantize/Dequantize take one scale and zero point; export the weights baked, without Q/DQ).");
            }
        }
    }
}
