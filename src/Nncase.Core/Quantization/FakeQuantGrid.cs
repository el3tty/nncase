// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Quantization;

/// <summary>
/// The integer grid a constant was fake-quantized to by the model (e.g. folded ONNX QuantizeLinear/DequantizeLinear on
/// weights): value = (q - ZeroPoints[c]) * Scales[c] with q in the range of <see cref="QuantType"/> and c the index along
/// <see cref="Axis"/>, or one scale and zero point for the whole tensor. The IR Quantize/Dequantize ops only take a scalar
/// quant param, so a per-channel grid is carried this way.
/// </summary>
/// <param name="QuantType">Integer type of the grid.</param>
/// <param name="Scales">Scale per channel, or one.</param>
/// <param name="ZeroPoints">Zero point per channel, or one.</param>
/// <param name="Axis">Channel axis.</param>
public sealed record FakeQuantGrid(DataType QuantType, float[] Scales, int[] ZeroPoints, int Axis);
