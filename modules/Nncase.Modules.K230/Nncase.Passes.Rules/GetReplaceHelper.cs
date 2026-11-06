// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Runtime.InteropServices;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K230;
using Nncase.IR.K230.F;
using Nncase.PatternMatch;

namespace Nncase.Passes.Rules;

public static class GetReplaceHelper
{
    public static Func<Expr, Expr> WithTmpGNNEShape(Func<Expr, Expr> inputCtor, int[] originOutShape)
    {
        return Utility.WithTmp4DShape(inputCtor, originOutShape);
    }

    public static Func<Expr, Expr> WithTmpGNNEShape(Func<Expr, Expr> inputCtor)
    {
        return (Expr input) => WithTmpGNNEShape(inputCtor, input.CheckedShape.ToValueArray())(input);
    }

    public static Expr ToGNNEShape(Expr input)
    {
        return Nncase.IR.F.Tensors.Reshape(
            input,
            GNNETypePatternUtility.GetGNNEShape(input.CheckedShape.ToValueArray()));
    }

    public static Expr ToF16(Expr input)
    {
        Call call = Nncase.IR.F.Tensors.Cast(input, DataTypes.Float16);
        if (input is TensorConst)
        {
            return call.Evaluate().AsTensor();
        }

        return call;
    }

    public static Tensor ToTensor(Expr expr)
    {
        if (expr is TensorConst tensorConst)
        {
            return tensorConst.Value;
        }

        throw new InvalidOperationException("Expr is not a TensorCosnt");
    }

    public static T[] ToArray<T>(Expr expr)
        where T : unmanaged, IEquatable<T>
    {
        return ToTensor(expr).ToArray<T>();
    }

    public static T ToScalar<T>(Expr expr)
        where T : unmanaged, IEquatable<T>
    {
        return ToTensor(expr).ToScalar<T>();
    }

    public static Func<Expr, Expr> WithFullLower(Func<Expr, Expr> inputCtor, int[] outShape)
    {
        return WithTmpGNNEShape(WithLoadStore(inputCtor), outShape);
    }

    public static Func<Expr, Expr> WithLoadStore(Func<Expr, Expr> inputCtor)
    {
        return Utility.Apply(WithLoadStoreImpl, inputCtor);

        static Func<Expr, Expr> WithLoadStoreImpl(Func<Expr, Expr> inCtor)
        {
            return (Expr input) => Nncase.IR.K230.F.Tensors.GNNEStore(
                input.CheckedDataType,
                inCtor(Nncase.IR.K230.F.Tensors.GNNELoad(
                    (input.CheckedDataType == DataTypes.Float32)
                        ? DataTypes.Float16
                        : ((PrimType)input.CheckedDataType), input)));
        }
    }

    public static Expr LoadActIF(Expr input)
    {
        return Nncase.IR.K230.F.Tensors.GNNELoad(DataTypes.Float16, input);
    }

    public static Expr LoadAct0(ActParam2 actParam)
    {
        return Nncase.IR.K230.F.Tensors.GNNELoadW(DataTypes.Float16, actParam.ToAct0Data());
    }

    public static Expr LoadAct1(ActParamBase actParam)
    {
        return Nncase.IR.K230.F.Tensors.GNNELoadW(DataTypes.Float16, actParam.ToAct1Data());
    }

    public static Expr SuppressPattern(RunPassContext options, Expr expr, IPattern pattern)
    {
        options.MatchOptions.SuppressPattern(expr, pattern);
        return expr;
    }

    public static ActParam2? FoldAct0WithAct1(ActParam2 act0Param, Expr is16Segments, TensorConst outChannels,
        ActParam2 act1Param, TensorConst deqParams = null)
    {
        if (is16Segments == true)
        {
            return null;
        }

        int channels = act0Param.Channels;
        int act1Channels = outChannels.Value.ToScalar<int>();
        if (channels != act1Channels && act1Channels != 1)
        {
            return null;
        }

        if (deqParams != null)
        {
            for (int i = 0; i < channels; i++)
            {
                float deqScale = MemoryMarshal.Cast<byte, float>(deqParams.Value.BytesBuffer)[1];
                int deqZeroPoint = MemoryMarshal.Cast<byte, int>(deqParams.Value.BytesBuffer)[0];
                act0Param.Ks[0, i] *= deqScale;
                act0Param.Ks[1, i] *= deqScale;
                act0Param.Bs[0, i] = (act0Param.Bs[0, i] - (float)deqZeroPoint) * deqScale;
                act0Param.Bs[1, i] = (act0Param.Bs[1, i] - (float)deqZeroPoint) * deqScale;
                act0Param.FusedClamp[i].Max = (act0Param.FusedClamp[i].Max - (float)deqZeroPoint) * deqScale;
                act0Param.FusedClamp[i].Min = (act0Param.FusedClamp[i].Min - (float)deqZeroPoint) * deqScale;
            }
        }

        ActFoldParam actFoldParam = new ActFoldParam(channels);
        for (int j = 0; j < channels; j++)
        {
            float act1K0 = (act1Channels == 1) ? act1Param.Ks[0, 0] : act1Param.Ks[0, j];
            float act1K1 = (act1Channels == 1) ? act1Param.Ks[1, 0] : act1Param.Ks[1, j];
            float act1B0 = (act1Channels == 1) ? act1Param.Bs[0, 0] : act1Param.Bs[0, j];
            float act1B1 = (act1Channels == 1) ? act1Param.Bs[1, 0] : act1Param.Bs[1, j];
            float act1X0 = (act1Channels == 1) ? act1Param.Xs[0, 0] : act1Param.Xs[0, j];
            float act1ClampMin = (act1Channels == 1) ? act1Param.FusedClamp[0].Min : act1Param.FusedClamp[j].Min;
            float act1ClampMax = (act1Channels == 1) ? act1Param.FusedClamp[0].Max : act1Param.FusedClamp[j].Max;
            if (act1X0 >= act0Param.FusedClamp[j].Max)
            {
                if (act1K0 > 0f)
                {
                    float foldedMax = double.IsPositiveInfinity(act0Param.FusedClamp[j].Max)
                        ? act0Param.FusedClamp[j].Max
                        : (act0Param.FusedClamp[j].Max * act1K0 + act1B0);
                    float foldedMin = double.IsNegativeInfinity(act0Param.FusedClamp[j].Min)
                        ? act0Param.FusedClamp[j].Min
                        : (act0Param.FusedClamp[j].Min * act1K0 + act1B0);
                    actFoldParam.FusedClamp[j].Max = (foldedMax < act1ClampMax) ? ((Half)foldedMax) : ((Half)act1ClampMax);
                    actFoldParam.FusedClamp[j].Min = (foldedMin > act1ClampMin) ? ((Half)foldedMin) : ((Half)act1ClampMin);
                    if (actFoldParam.FusedClamp[j].Max < actFoldParam.FusedClamp[j].Min)
                    {
                        return null;
                    }
                }
                else if (act1K0 < 0f)
                {
                    float foldedMin = double.IsPositiveInfinity(act0Param.FusedClamp[j].Max)
                        ? (0f - act0Param.FusedClamp[j].Max)
                        : (act0Param.FusedClamp[j].Max * act1K0 + act1B0);
                    float foldedMax = double.IsNegativeInfinity(act0Param.FusedClamp[j].Min)
                        ? (0f - act0Param.FusedClamp[j].Min)
                        : (act0Param.FusedClamp[j].Min * act1K0 + act1B0);
                    actFoldParam.FusedClamp[j].Max = (foldedMax < act1ClampMax) ? ((Half)foldedMax) : ((Half)act1ClampMax);
                    actFoldParam.FusedClamp[j].Min = (foldedMin > act1ClampMin) ? ((Half)foldedMin) : ((Half)act1ClampMin);
                    if (actFoldParam.FusedClamp[j].Max < actFoldParam.FusedClamp[j].Min)
                    {
                        return null;
                    }
                }
                else
                {
                    actFoldParam.FusedClamp[j].Max = (Half)act1ClampMax;
                    actFoldParam.FusedClamp[j].Min = (Half)act1ClampMin;
                }

                continue;
            }

            if (act1X0 <= act0Param.FusedClamp[j].Min)
            {
                if (!(act1K1 > 0f))
                {
                    if (act1K1 < 0f)
                    {
                        float foldedMin = double.IsPositiveInfinity(act0Param.FusedClamp[j].Max)
                            ? (0f - act0Param.FusedClamp[j].Max)
                            : (act0Param.FusedClamp[j].Max * act1K1 + act1B1);
                        float foldedMax = double.IsNegativeInfinity(act0Param.FusedClamp[j].Min)
                            ? (0f - act0Param.FusedClamp[j].Min)
                            : (act0Param.FusedClamp[j].Min * act1K1 + act1B1);
                        actFoldParam.FusedClamp[j].Max = (foldedMax < act1ClampMax) ? ((Half)foldedMax) : ((Half)act1ClampMax);
                        actFoldParam.FusedClamp[j].Min = (foldedMin > act1ClampMin) ? ((Half)foldedMin) : ((Half)act1ClampMin);
                        if (actFoldParam.FusedClamp[j].Max < actFoldParam.FusedClamp[j].Min)
                        {
                            return null;
                        }
                    }
                    else
                    {
                        actFoldParam.FusedClamp[j].Max = (Half)act1ClampMax;
                        actFoldParam.FusedClamp[j].Min = (Half)act1ClampMin;
                    }
                }
                else
                {
                    float foldedMax = double.IsPositiveInfinity(act0Param.FusedClamp[j].Max)
                        ? act0Param.FusedClamp[j].Max
                        : (act0Param.FusedClamp[j].Max * act1K1 + act1B1);
                    float foldedMin = double.IsNegativeInfinity(act0Param.FusedClamp[j].Min)
                        ? act0Param.FusedClamp[j].Min
                        : (act0Param.FusedClamp[j].Min * act1K1 + act1B1);
                    actFoldParam.FusedClamp[j].Max = (foldedMax < act1ClampMax) ? ((Half)foldedMax) : ((Half)act1ClampMax);
                    actFoldParam.FusedClamp[j].Min = (foldedMin > act1ClampMin) ? ((Half)foldedMin) : ((Half)act1ClampMin);
                    if (actFoldParam.FusedClamp[j].Max < actFoldParam.FusedClamp[j].Min)
                    {
                        return null;
                    }
                }

                continue;
            }

            float midSlope = act1K0;
            if (!(midSlope > 0f))
            {
                if (!(midSlope < 0f))
                {
                    if (midSlope != 0f)
                    {
                        continue;
                    }

                    if (act1K1 > 0f)
                    {
                        float foldedMax = double.IsPositiveInfinity(act0Param.FusedClamp[j].Max)
                            ? act0Param.FusedClamp[j].Max
                            : (act0Param.FusedClamp[j].Max * act1K1 + act1B1);
                        actFoldParam.FusedClamp[j].Max = (foldedMax < act1ClampMax) ? ((Half)foldedMax) : ((Half)act1ClampMax);
                        actFoldParam.FusedClamp[j].Min = (Half)act1ClampMin;
                        if (actFoldParam.FusedClamp[j].Max < actFoldParam.FusedClamp[j].Min)
                        {
                            return null;
                        }
                    }
                    else if (act1K1 < 0f)
                    {
                        float foldedMin = double.IsPositiveInfinity(act0Param.FusedClamp[j].Max)
                            ? (0f - act0Param.FusedClamp[j].Max)
                            : (act0Param.FusedClamp[j].Max * act1K1 + act1B1);
                        actFoldParam.FusedClamp[j].Max = (Half)act1ClampMax;
                        actFoldParam.FusedClamp[j].Min = (foldedMin > act1ClampMin) ? ((Half)foldedMin) : ((Half)act1ClampMin);
                        if (actFoldParam.FusedClamp[j].Max < actFoldParam.FusedClamp[j].Min)
                        {
                            return null;
                        }
                    }
                    else if (act1K1 == 0f)
                    {
                        actFoldParam.FusedClamp[j].Max = (Half)act1ClampMax;
                        actFoldParam.FusedClamp[j].Min = (Half)act1ClampMin;
                    }
                }
                else if (act1K1 < 0f)
                {
                    float foldedMin = double.IsPositiveInfinity(act0Param.FusedClamp[j].Max)
                        ? (0f - act0Param.FusedClamp[j].Max)
                        : (act0Param.FusedClamp[j].Max * act1K1 + act1B1);
                    float foldedMax = double.IsNegativeInfinity(act0Param.FusedClamp[j].Min)
                        ? (0f - act0Param.FusedClamp[j].Min)
                        : (act0Param.FusedClamp[j].Min * act1K0 + act1B0);
                    actFoldParam.FusedClamp[j].Max = (foldedMax < act1ClampMax) ? ((Half)foldedMax) : ((Half)act1ClampMax);
                    actFoldParam.FusedClamp[j].Min = (foldedMin > act1ClampMin) ? ((Half)foldedMin) : ((Half)act1ClampMin);
                    if (actFoldParam.FusedClamp[j].Max < actFoldParam.FusedClamp[j].Min)
                    {
                        return null;
                    }
                }
                else if (act1K1 > 0f)
                {
                    float foldedAtMax = double.IsPositiveInfinity(act0Param.FusedClamp[j].Max)
                        ? act0Param.FusedClamp[j].Max
                        : (act0Param.FusedClamp[j].Max * act1K1 + act1B1);
                    float foldedAtMin = double.IsNegativeInfinity(act0Param.FusedClamp[j].Min)
                        ? (0f - act0Param.FusedClamp[j].Min)
                        : (act0Param.FusedClamp[j].Min * act1K0 + act1B0);
                    float foldedMax = (foldedAtMax > foldedAtMin) ? foldedAtMin : foldedAtMax;
                    actFoldParam.FusedClamp[j].Max = (foldedMax < act1ClampMax) ? ((Half)foldedMax) : ((Half)act1ClampMax);
                    actFoldParam.FusedClamp[j].Min = (Half)act1ClampMin;
                    if (actFoldParam.FusedClamp[j].Max < actFoldParam.FusedClamp[j].Min)
                    {
                        return null;
                    }
                }
                else if (act1K1 == 0f)
                {
                    float foldedMax = double.IsNegativeInfinity(act0Param.FusedClamp[j].Min)
                        ? (0f - act0Param.FusedClamp[j].Min)
                        : (act0Param.FusedClamp[j].Min * act1K0 + act1B0);
                    actFoldParam.FusedClamp[j].Max = (foldedMax < act1ClampMax) ? ((Half)foldedMax) : ((Half)act1ClampMax);
                    actFoldParam.FusedClamp[j].Min = (Half)act1ClampMin;
                    if (actFoldParam.FusedClamp[j].Max < actFoldParam.FusedClamp[j].Min)
                    {
                        return null;
                    }
                }
            }
            else if (act1K1 > 0f)
            {
                float foldedMax = double.IsPositiveInfinity(act0Param.FusedClamp[j].Max)
                    ? act0Param.FusedClamp[j].Max
                    : (act0Param.FusedClamp[j].Max * act1K1 + act1B1);
                float foldedMin = double.IsNegativeInfinity(act0Param.FusedClamp[j].Min)
                    ? act0Param.FusedClamp[j].Min
                    : (act0Param.FusedClamp[j].Min * act1K0 + act1B0);
                actFoldParam.FusedClamp[j].Max = (foldedMax < act1ClampMax) ? ((Half)foldedMax) : ((Half)act1ClampMax);
                actFoldParam.FusedClamp[j].Min = (foldedMin > act1ClampMin) ? ((Half)foldedMin) : ((Half)act1ClampMin);
                if (actFoldParam.FusedClamp[j].Max < actFoldParam.FusedClamp[j].Min)
                {
                    return null;
                }
            }
            else if (act1K1 < 0f)
            {
                float foldedAtMax = double.IsPositiveInfinity(act0Param.FusedClamp[j].Max)
                    ? (0f - act0Param.FusedClamp[j].Max)
                    : (act0Param.FusedClamp[j].Max * act1K1 + act1B1);
                float foldedAtMin = double.IsNegativeInfinity(act0Param.FusedClamp[j].Min)
                    ? act0Param.FusedClamp[j].Min
                    : (act0Param.FusedClamp[j].Min * act1K0 + act1B0);
                float foldedMin = (foldedAtMax > foldedAtMin) ? foldedAtMax : foldedAtMin;
                actFoldParam.FusedClamp[j].Max = (Half)act1ClampMax;
                actFoldParam.FusedClamp[j].Min = (foldedMin > act1ClampMin) ? ((Half)foldedMin) : ((Half)act1ClampMin);
                if (actFoldParam.FusedClamp[j].Max < actFoldParam.FusedClamp[j].Min)
                {
                    return null;
                }
            }
            else if (act1K1 == 0f)
            {
                float foldedMin = double.IsNegativeInfinity(act0Param.FusedClamp[j].Min)
                    ? act0Param.FusedClamp[j].Min
                    : (act0Param.FusedClamp[j].Min * act1K0 + act1B0);
                actFoldParam.FusedClamp[j].Max = (Half)act1ClampMax;
                actFoldParam.FusedClamp[j].Min = (foldedMin > act1ClampMin) ? ((Half)foldedMin) : ((Half)act1ClampMin);
                if (actFoldParam.FusedClamp[j].Max < actFoldParam.FusedClamp[j].Min)
                {
                    return null;
                }
            }
        }

        for (int k = 0; k < channels; k++)
        {
            float act1K0 = (act1Channels == 1) ? act1Param.Ks[0, 0] : act1Param.Ks[0, k];
            float act1K1 = (act1Channels == 1) ? act1Param.Ks[1, 0] : act1Param.Ks[1, k];
            float act1B0 = (act1Channels == 1) ? act1Param.Bs[0, 0] : act1Param.Bs[0, k];
            float act1B1 = (act1Channels == 1) ? act1Param.Bs[1, 0] : act1Param.Bs[1, k];
            float act1X0 = (act1Channels == 1) ? act1Param.Xs[0, 0] : act1Param.Xs[0, k];
            if (act1Channels != 1)
            {
                _ = act1Param.FusedClamp[k].Min;
            }
            else
            {
                _ = act1Param.FusedClamp[0].Min;
            }

            if (act1Channels != 1)
            {
                _ = act1Param.FusedClamp[k].Max;
            }
            else
            {
                _ = act1Param.FusedClamp[0].Max;
            }

            float act0K0 = act0Param.Ks[0, k];
            if (!(act0K0 > 0f))
            {
                if (!(act0K0 < 0f))
                {
                    if (act0K0 == 0f)
                    {
                        if (act0Param.Ks[1, k] > 0f)
                        {
                            if ((act0Param.Bs[0, k] >= act1X0 ||
                                 act0Param.Bs[1, k] + act0Param.Xs[0, k] * act0Param.Ks[1, k] <= act1X0) &&
                                (act1K0 != act1K1 || act1B0 != act1B1))
                            {
                                return null;
                            }

                            actFoldParam.Ks[0, k] = 0f;
                            actFoldParam.Ks[1, k] = act0Param.Ks[1, k] * act1K1;
                            actFoldParam.Bs[0, k] = act0Param.Bs[0, k] * act1K0 + act1B0;
                            actFoldParam.Bs[1, k] = act0Param.Bs[1, k] * act1K1 + act1B1;
                            actFoldParam.Xs[0, k] = act0Param.Xs[0, k];
                            continue;
                        }

                        if (act0Param.Ks[1, k] < 0f)
                        {
                            if ((act0Param.Bs[0, k] <= act1X0 ||
                                 act0Param.Bs[1, k] - act0Param.Xs[0, k] * act0Param.Ks[1, k] >= act1X0) &&
                                (act1K0 != act1K1 || act1B0 != act1B1))
                            {
                                return null;
                            }

                            actFoldParam.Ks[0, k] = 0f;
                            actFoldParam.Ks[1, k] = act0Param.Ks[1, k] * act1K0;
                            actFoldParam.Bs[0, k] = act0Param.Bs[0, k] * act1K1 + act1B1;
                            actFoldParam.Bs[1, k] = act0Param.Bs[1, k] * act1K0 + act1B0;
                            actFoldParam.Xs[0, k] = act0Param.Xs[0, k];
                            continue;
                        }

                        if (act0Param.Ks[1, k] == 0f)
                        {
                            if ((act0Param.Bs[0, k] >= act1X0 || act0Param.Bs[1, k] <= act1X0) &&
                                (act1K0 != act1K1 || act1B0 != act1B1))
                            {
                                return null;
                            }

                            actFoldParam.Ks[0, k] = 0f;
                            actFoldParam.Ks[1, k] = 0f;
                            actFoldParam.Bs[0, k] = act0Param.Bs[0, k] * act1K0 + act1B0;
                            actFoldParam.Bs[1, k] = act0Param.Bs[1, k] * act1K1 + act1B1;
                            actFoldParam.Xs[0, k] = act0Param.Xs[0, k];
                            continue;
                        }
                    }
                }
                else
                {
                    if (act0Param.Ks[1, k] < 0f)
                    {
                        if (act0Param.Xs[0, k] != 0f &&
                            System.Math.Abs(act1X0 - (act0Param.Xs[0, k] * act0Param.Ks[0, k] + act0Param.Bs[0, k])) >
                            float.Epsilon)
                        {
                            return null;
                        }

                        if ((act0Param.Bs[0, k] < act1X0 || act0Param.Bs[1, k] > act1X0) &&
                            (act1K0 != act1K1 || act1B0 != act1B1) && (act0Param.Ks[0, k] != act0Param.Ks[1, k] ||
                                                                   act0Param.Bs[0, k] != act0Param.Bs[1, k]))
                        {
                            return null;
                        }

                        if (act0Param.Ks[0, k] == act0Param.Ks[1, k] && act0Param.Bs[0, k] == act0Param.Bs[1, k] &&
                            (act1K0 != act1K1 || act1B0 != act1B1))
                        {
                            actFoldParam.Ks[0, k] = act0Param.Ks[0, k] * act1K1;
                            actFoldParam.Ks[1, k] = act0Param.Ks[1, k] * act1K0;
                            actFoldParam.Bs[0, k] = act0Param.Bs[0, k] * act1K1 + act1B1;
                            actFoldParam.Bs[1, k] = act0Param.Bs[1, k] * act1K0 + act1B0;
                            actFoldParam.Xs[0, k] = (act1X0 - act0Param.Bs[0, k]) / act0Param.Ks[0, k];
                        }
                        else
                        {
                            actFoldParam.Ks[0, k] = act0Param.Ks[0, k] * act1K0;
                            actFoldParam.Ks[1, k] = act0Param.Ks[1, k] * act1K1;
                            actFoldParam.Bs[0, k] = act0Param.Bs[0, k] * act1K0 + act1B0;
                            actFoldParam.Bs[1, k] = act0Param.Bs[1, k] * act1K1 + act1B1;
                            actFoldParam.Xs[0, k] = act0Param.Xs[0, k];
                        }

                        continue;
                    }

                    if (act0Param.Ks[1, k] == 0f)
                    {
                        if ((act0Param.Bs[0, k] - act0Param.Xs[0, k] * act0Param.Ks[0, k] <= act1X0 ||
                             act0Param.Bs[1, k] >= act1X0) && (act1K0 != act1K1 || act1B0 != act1B1))
                        {
                            return null;
                        }

                        actFoldParam.Ks[0, k] = act0Param.Ks[0, k] * act1K1;
                        actFoldParam.Ks[1, k] = 0f;
                        actFoldParam.Bs[0, k] = act0Param.Bs[0, k] * act1K1 + act1B1;
                        actFoldParam.Bs[1, k] = act0Param.Bs[1, k] * act1K0 + act1B0;
                        actFoldParam.Xs[0, k] = act0Param.Xs[0, k];
                        continue;
                    }
                }
            }
            else
            {
                if (act0Param.Ks[1, k] > 0f)
                {
                    if (act0Param.Xs[0, k] != 0f &&
                        System.Math.Abs(act1X0 - (act0Param.Xs[0, k] * act0Param.Ks[0, k] + act0Param.Bs[0, k])) >
                        float.Epsilon)
                    {
                        return null;
                    }

                    if ((act0Param.Bs[0, k] > act1X0 || act0Param.Bs[1, k] < act1X0) &&
                        (act1K0 != act1K1 || act1B0 != act1B1) && (act0Param.Ks[0, k] != act0Param.Ks[1, k] ||
                                                               act0Param.Bs[0, k] != act0Param.Bs[1, k]))
                    {
                        return null;
                    }

                    if (act0Param.Ks[0, k] == act0Param.Ks[1, k] && act0Param.Bs[0, k] == act0Param.Bs[1, k] &&
                        (act1K0 != act1K1 || act1B0 != act1B1))
                    {
                        actFoldParam.Ks[0, k] = act0Param.Ks[0, k] * act1K0;
                        actFoldParam.Ks[1, k] = act0Param.Ks[1, k] * act1K1;
                        actFoldParam.Bs[0, k] = act0Param.Bs[0, k] * act1K0 + act1B0;
                        actFoldParam.Bs[1, k] = act0Param.Bs[1, k] * act1K1 + act1B1;
                        actFoldParam.Xs[0, k] = (act1X0 - act0Param.Bs[0, k]) / act0Param.Ks[0, k];
                    }
                    else
                    {
                        actFoldParam.Ks[0, k] = act0Param.Ks[0, k] * act1K0;
                        actFoldParam.Ks[1, k] = act0Param.Ks[1, k] * act1K1;
                        actFoldParam.Bs[0, k] = act0Param.Bs[0, k] * act1K0 + act1B0;
                        actFoldParam.Bs[1, k] = act0Param.Bs[1, k] * act1K1 + act1B1;
                        actFoldParam.Xs[0, k] = act0Param.Xs[0, k];
                    }

                    continue;
                }

                if (act0Param.Ks[1, k] == 0f)
                {
                    if ((act0Param.Bs[0, k] + act0Param.Xs[0, k] * act0Param.Ks[0, k] >= act1X0 ||
                         act0Param.Bs[1, k] <= act1X0) && (act1K0 != act1K1 || act1B0 != act1B1))
                    {
                        return null;
                    }

                    actFoldParam.Ks[0, k] = act0Param.Ks[0, k] * act1K0;
                    actFoldParam.Ks[1, k] = 0f;
                    actFoldParam.Bs[0, k] = act0Param.Bs[0, k] * act1K0 + act1B0;
                    actFoldParam.Bs[1, k] = act0Param.Bs[1, k] * act1K1 + act1B1;
                    actFoldParam.Xs[0, k] = act0Param.Xs[0, k];
                    continue;
                }
            }

            if (act1K0 != act1K1 || act1B0 != act1B1)
            {
                return null;
            }

            actFoldParam.Ks[0, k] = act0Param.Ks[0, k] * act1K0;
            actFoldParam.Ks[1, k] = act0Param.Ks[1, k] * act1K1;
            actFoldParam.Bs[0, k] = act0Param.Bs[0, k] * act1K0 + act1B0;
            actFoldParam.Bs[1, k] = act0Param.Bs[1, k] * act1K1 + act1B1;
            actFoldParam.Xs[0, k] = act0Param.Xs[0, k];
        }

        ActParam2 actParam = new ActParam2(channels);
        for (int l = 0; l < channels; l++)
        {
            actParam.Ks[0, l] = actFoldParam.Ks[0, l];
            actParam.Ks[1, l] = actFoldParam.Ks[1, l];
            actParam.Bs[0, l] = actFoldParam.Bs[0, l];
            actParam.Bs[1, l] = actFoldParam.Bs[1, l];
            actParam.Xs[0, l] = actFoldParam.Xs[0, l];
            actParam.FusedClamp[l].Min = (float)actFoldParam.FusedClamp[l].Min;
            actParam.FusedClamp[l].Max = (float)actFoldParam.FusedClamp[l].Max;
        }

        actParam.Qp = act0Param.Qp;
        return actParam;
    }
}
