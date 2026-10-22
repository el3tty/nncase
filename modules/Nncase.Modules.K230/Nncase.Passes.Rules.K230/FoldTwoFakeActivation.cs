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
using Nncase.Utilities;

namespace Nncase.Passes.Rules.K230;

[RuleGenerator]
public sealed class FoldTwoFakeActivation : RewriteRule<Pattern>, IRewriteRule
{
    public override Pattern Pattern { get; } = Nncase.PatternMatch.F.K230.IsFakeActivation("fakeAct2", "call2",
        (FakeActivation _) => true,
        Nncase.PatternMatch.Utility.IsRangeOfMarker(
            "range",
            Nncase.PatternMatch.F.K230.IsFakeActivation("fakeAct1", "call1", (FakeActivation _) => true,
                Nncase.PatternMatch.Utility.IsWildcard("inputa"), Nncase.PatternMatch.Utility.IsWildcard("inputb"),
                Nncase.PatternMatch.Utility.IsTensorConst("act1"),
                Nncase.PatternMatch.Utility.IsTensorConst("outchannels1"),
                Nncase.PatternMatch.Utility.IsTensorConst("inAShiftBits1"),
                Nncase.PatternMatch.Utility.IsTensorConst("inBShiftBits1"),
                Nncase.PatternMatch.Utility.IsTensorConst("outShiftBits1"),
                Nncase.PatternMatch.Utility.IsTensorConst("is16Segments1")),
            Nncase.PatternMatch.Utility.IsTensorConst("fakeAct2InputRange")), Nncase.PatternMatch.Utility.IsNone(),
        Nncase.PatternMatch.Utility.IsTensorConst("act2"), Nncase.PatternMatch.Utility.IsTensorConst("outchannels2"),
        Nncase.PatternMatch.Utility.IsTensorConst("inAShiftBits2"),
        Nncase.PatternMatch.Utility.IsTensorConst("inBShiftBits2"),
        Nncase.PatternMatch.Utility.IsTensorConst("outShiftBits2"),
        Nncase.PatternMatch.Utility.IsTensorConst("is16Segments2"));

    private Expr? GetReplace(FakeActivation fakeAct1, Expr inputa, Expr inputb, TensorConst act1,
        TensorConst outchannels1, FakeActivation fakeAct2, Call call1, Call call2, TensorConst act2,
        TensorConst outchannels2, Expr is16Segments1, Expr is16Segments2, Expr range, RunPassContext context)
    {
        if (call1.CheckedShape.Size < call2.CheckedShape.Size)
        {
            return null;
        }

        if (context.Driver is DataflowPass && context.GetAnalysis<IExprUserAnalysisResult>()[range].Count() > 1)
        {
            return null;
        }

        if (is16Segments1 == true || is16Segments2 == true)
        {
            return null;
        }

        int outChannels1 = outchannels1.Value.ToScalar<int>();
        int outChannels2 = outchannels2.Value.ToScalar<int>();
        if (outChannels1 != outChannels2)
        {
            return null;
        }

        ActFoldParam actFoldParam = new ActFoldParam(outChannels1);
        for (int i = 0; i < outChannels1; i++)
        {
            if (act2.Value.Cast<float>()[new int[2] { i, 0 }] >= fakeAct1.ActParam.FusedClamp[i].Max)
            {
                if (act2.Value.Cast<float>()[new int[2] { i, 1 }] > 0f)
                {
                    float clampMax = double.IsPositiveInfinity(fakeAct1.ActParam.FusedClamp[i].Max)
                        ? fakeAct1.ActParam.FusedClamp[i].Max
                        : (fakeAct1.ActParam.FusedClamp[i].Max * act2.Value.Cast<float>()[new int[2] { i, 1 }] +
                           act2.Value.Cast<float>()[new int[2] { i, 3 }]);
                    float clampMin = double.IsNegativeInfinity(fakeAct1.ActParam.FusedClamp[i].Min)
                        ? fakeAct1.ActParam.FusedClamp[i].Min
                        : (fakeAct1.ActParam.FusedClamp[i].Min * act2.Value.Cast<float>()[new int[2] { i, 1 }] +
                           act2.Value.Cast<float>()[new int[2] { i, 3 }]);
                    actFoldParam.FusedClamp[i] = new ValueRange<Half>
                    {
                        Min = (clampMin > fakeAct2.ActParam.FusedClamp[i].Min)
                            ? ((Half)clampMin)
                            : ((Half)fakeAct2.ActParam.FusedClamp[i].Min),
                        Max = (clampMax < fakeAct2.ActParam.FusedClamp[i].Max)
                            ? ((Half)clampMax)
                            : ((Half)fakeAct2.ActParam.FusedClamp[i].Max),
                    };
                    if (actFoldParam.FusedClamp[i].Max < actFoldParam.FusedClamp[i].Min)
                    {
                        return null;
                    }
                }
                else if (act2.Value.Cast<float>()[new int[2] { i, 1 }] < 0f)
                {
                    float clampMin = double.IsPositiveInfinity(fakeAct1.ActParam.FusedClamp[i].Max)
                        ? (0f - fakeAct1.ActParam.FusedClamp[i].Max)
                        : (fakeAct1.ActParam.FusedClamp[i].Max * act2.Value.Cast<float>()[new int[2] { i, 1 }] +
                           act2.Value.Cast<float>()[new int[2] { i, 3 }]);
                    float clampMax = double.IsNegativeInfinity(fakeAct1.ActParam.FusedClamp[i].Min)
                        ? (0f - fakeAct1.ActParam.FusedClamp[i].Min)
                        : (fakeAct1.ActParam.FusedClamp[i].Min * act2.Value.Cast<float>()[new int[2] { i, 1 }] +
                           act2.Value.Cast<float>()[new int[2] { i, 3 }]);
                    actFoldParam.FusedClamp[i] = new ValueRange<Half>
                    {
                        Min = (clampMin > fakeAct2.ActParam.FusedClamp[i].Min)
                            ? ((Half)clampMin)
                            : ((Half)fakeAct2.ActParam.FusedClamp[i].Min),
                        Max = (clampMax < fakeAct2.ActParam.FusedClamp[i].Max)
                            ? ((Half)clampMax)
                            : ((Half)fakeAct2.ActParam.FusedClamp[i].Max),
                    };
                    if (actFoldParam.FusedClamp[i].Max < actFoldParam.FusedClamp[i].Min)
                    {
                        return null;
                    }
                }
                else
                {
                    actFoldParam.FusedClamp[i] = new ValueRange<Half>
                    {
                        Min = (Half)fakeAct2.ActParam.FusedClamp[i].Min,
                        Max = (Half)fakeAct2.ActParam.FusedClamp[i].Max,
                    };
                }
            }
            else if (act2.Value.Cast<float>()[new int[2] { i, 0 }] <= fakeAct1.ActParam.FusedClamp[i].Min)
            {
                if (act2.Value.Cast<float>()[new int[2] { i, 1 }] > 0f)
                {
                    float clampMax = double.IsPositiveInfinity(fakeAct1.ActParam.FusedClamp[i].Max)
                        ? fakeAct1.ActParam.FusedClamp[i].Max
                        : (fakeAct1.ActParam.FusedClamp[i].Max * act2.Value.Cast<float>()[new int[2] { i, 2 }] +
                           act2.Value.Cast<float>()[new int[2] { i, 4 }]);
                    float clampMin = double.IsNegativeInfinity(fakeAct1.ActParam.FusedClamp[i].Min)
                        ? fakeAct1.ActParam.FusedClamp[i].Min
                        : (fakeAct1.ActParam.FusedClamp[i].Min * act2.Value.Cast<float>()[new int[2] { i, 2 }] +
                           act2.Value.Cast<float>()[new int[2] { i, 4 }]);
                    actFoldParam.FusedClamp[i] = new ValueRange<Half>
                    {
                        Min = (clampMin > fakeAct2.ActParam.FusedClamp[i].Min)
                            ? ((Half)clampMin)
                            : ((Half)fakeAct2.ActParam.FusedClamp[i].Min),
                        Max = (clampMax < fakeAct2.ActParam.FusedClamp[i].Max)
                            ? ((Half)clampMax)
                            : ((Half)fakeAct2.ActParam.FusedClamp[i].Max),
                    };
                    if (actFoldParam.FusedClamp[i].Max < actFoldParam.FusedClamp[i].Min)
                    {
                        return null;
                    }
                }
                else if (act2.Value.Cast<float>()[new int[2] { i, 1 }] < 0f)
                {
                    float clampMin = double.IsPositiveInfinity(fakeAct1.ActParam.FusedClamp[i].Max)
                        ? (0f - fakeAct1.ActParam.FusedClamp[i].Max)
                        : (fakeAct1.ActParam.FusedClamp[i].Max * act2.Value.Cast<float>()[new int[2] { i, 2 }] +
                           act2.Value.Cast<float>()[new int[2] { i, 4 }]);
                    float clampMax = double.IsNegativeInfinity(fakeAct1.ActParam.FusedClamp[i].Min)
                        ? (0f - fakeAct1.ActParam.FusedClamp[i].Min)
                        : (fakeAct1.ActParam.FusedClamp[i].Min * act2.Value.Cast<float>()[new int[2] { i, 2 }] +
                           act2.Value.Cast<float>()[new int[2] { i, 4 }]);
                    actFoldParam.FusedClamp[i] = new ValueRange<Half>
                    {
                        Min = (clampMin > fakeAct2.ActParam.FusedClamp[i].Min)
                            ? ((Half)clampMin)
                            : ((Half)fakeAct2.ActParam.FusedClamp[i].Min),
                        Max = (clampMax < fakeAct2.ActParam.FusedClamp[i].Max)
                            ? ((Half)clampMax)
                            : ((Half)fakeAct2.ActParam.FusedClamp[i].Max),
                    };
                    if (actFoldParam.FusedClamp[i].Max < actFoldParam.FusedClamp[i].Min)
                    {
                        return null;
                    }
                }
                else
                {
                    actFoldParam.FusedClamp[i] = new ValueRange<Half>
                    {
                        Min = (Half)fakeAct2.ActParam.FusedClamp[i].Min,
                        Max = (Half)fakeAct2.ActParam.FusedClamp[i].Max,
                    };
                }
            }
            else if (act2.Value.Cast<float>()[new int[2] { i, 1 }] > 0f &&
                     act2.Value.Cast<float>()[new int[2] { i, 2 }] > 0f)
            {
                float clampMax = double.IsPositiveInfinity(fakeAct1.ActParam.FusedClamp[i].Max)
                    ? fakeAct1.ActParam.FusedClamp[i].Max
                    : (fakeAct1.ActParam.FusedClamp[i].Max * act2.Value.Cast<float>()[new int[2] { i, 2 }] +
                       act2.Value.Cast<float>()[new int[2] { i, 4 }]);
                float clampMin = double.IsNegativeInfinity(fakeAct1.ActParam.FusedClamp[i].Min)
                    ? fakeAct1.ActParam.FusedClamp[i].Min
                    : (fakeAct1.ActParam.FusedClamp[i].Min * act2.Value.Cast<float>()[new int[2] { i, 1 }] +
                       act2.Value.Cast<float>()[new int[2] { i, 3 }]);
                actFoldParam.FusedClamp[i] = new ValueRange<Half>
                {
                    Min = (clampMin > fakeAct2.ActParam.FusedClamp[i].Min)
                        ? ((Half)clampMin)
                        : ((Half)fakeAct2.ActParam.FusedClamp[i].Min),
                    Max = (clampMax < fakeAct2.ActParam.FusedClamp[i].Max)
                        ? ((Half)clampMax)
                        : ((Half)fakeAct2.ActParam.FusedClamp[i].Max),
                };
                if (actFoldParam.FusedClamp[i].Max < actFoldParam.FusedClamp[i].Min)
                {
                    return null;
                }
            }
            else if (act2.Value.Cast<float>()[new int[2] { i, 1 }] < 0f &&
                     act2.Value.Cast<float>()[new int[2] { i, 2 }] < 0f)
            {
                float clampMin = double.IsPositiveInfinity(fakeAct1.ActParam.FusedClamp[i].Max)
                    ? (0f - fakeAct1.ActParam.FusedClamp[i].Max)
                    : (fakeAct1.ActParam.FusedClamp[i].Max * act2.Value.Cast<float>()[new int[2] { i, 2 }] +
                       act2.Value.Cast<float>()[new int[2] { i, 4 }]);
                float clampMax = double.IsNegativeInfinity(fakeAct1.ActParam.FusedClamp[i].Min)
                    ? (0f - fakeAct1.ActParam.FusedClamp[i].Min)
                    : (fakeAct1.ActParam.FusedClamp[i].Min * act2.Value.Cast<float>()[new int[2] { i, 1 }] +
                       act2.Value.Cast<float>()[new int[2] { i, 3 }]);
                actFoldParam.FusedClamp[i] = new ValueRange<Half>
                {
                    Min = (clampMin > fakeAct2.ActParam.FusedClamp[i].Min)
                        ? ((Half)clampMin)
                        : ((Half)fakeAct2.ActParam.FusedClamp[i].Min),
                    Max = (clampMax < fakeAct2.ActParam.FusedClamp[i].Max)
                        ? ((Half)clampMax)
                        : ((Half)fakeAct2.ActParam.FusedClamp[i].Max),
                };
                if (actFoldParam.FusedClamp[i].Max < actFoldParam.FusedClamp[i].Min)
                {
                    return null;
                }
            }
            else if (act2.Value.Cast<float>()[new int[2] { i, 1 }] > 0f &&
                     act2.Value.Cast<float>()[new int[2] { i, 2 }] < 0f)
            {
                float posBound = double.IsPositiveInfinity(fakeAct1.ActParam.FusedClamp[i].Max)
                    ? (0f - fakeAct1.ActParam.FusedClamp[i].Max)
                    : (fakeAct1.ActParam.FusedClamp[i].Max * act2.Value.Cast<float>()[new int[2] { i, 2 }] +
                       act2.Value.Cast<float>()[new int[2] { i, 4 }]);
                float negBound = double.IsNegativeInfinity(fakeAct1.ActParam.FusedClamp[i].Min)
                    ? fakeAct1.ActParam.FusedClamp[i].Min
                    : (fakeAct1.ActParam.FusedClamp[i].Min * act2.Value.Cast<float>()[new int[2] { i, 1 }] +
                       act2.Value.Cast<float>()[new int[2] { i, 3 }]);
                float clampMin = (posBound > negBound) ? posBound : negBound;
                actFoldParam.FusedClamp[i] = new ValueRange<Half>
                {
                    Min = (clampMin > fakeAct2.ActParam.FusedClamp[i].Min)
                        ? ((Half)clampMin)
                        : ((Half)fakeAct2.ActParam.FusedClamp[i].Min),
                    Max = (Half)fakeAct2.ActParam.FusedClamp[i].Max,
                };
                if (actFoldParam.FusedClamp[i].Max < actFoldParam.FusedClamp[i].Min)
                {
                    return null;
                }
            }
            else if (act2.Value.Cast<float>()[new int[2] { i, 1 }] < 0f &&
                     act2.Value.Cast<float>()[new int[2] { i, 2 }] > 0f)
            {
                float posBound = double.IsPositiveInfinity(fakeAct1.ActParam.FusedClamp[i].Max)
                    ? fakeAct1.ActParam.FusedClamp[i].Max
                    : (fakeAct1.ActParam.FusedClamp[i].Max * act2.Value.Cast<float>()[new int[2] { i, 2 }] +
                       act2.Value.Cast<float>()[new int[2] { i, 4 }]);
                float negBound = double.IsNegativeInfinity(fakeAct1.ActParam.FusedClamp[i].Min)
                    ? (0f - fakeAct1.ActParam.FusedClamp[i].Min)
                    : (fakeAct1.ActParam.FusedClamp[i].Min * act2.Value.Cast<float>()[new int[2] { i, 1 }] +
                       act2.Value.Cast<float>()[new int[2] { i, 3 }]);
                float clampMax = (posBound > negBound) ? negBound : posBound;
                actFoldParam.FusedClamp[i] = new ValueRange<Half>
                {
                    Min = (Half)fakeAct2.ActParam.FusedClamp[i].Min,
                    Max = (clampMax < fakeAct2.ActParam.FusedClamp[i].Max)
                        ? ((Half)clampMax)
                        : ((Half)fakeAct2.ActParam.FusedClamp[i].Max),
                };
                if (actFoldParam.FusedClamp[i].Max < actFoldParam.FusedClamp[i].Min)
                {
                    return null;
                }
            }
            else if (act2.Value.Cast<float>()[new int[2] { i, 1 }] > 0f &&
                     act2.Value.Cast<float>()[new int[2] { i, 2 }] == 0f)
            {
                float clampMin = double.IsNegativeInfinity(fakeAct1.ActParam.FusedClamp[i].Min)
                    ? fakeAct1.ActParam.FusedClamp[i].Min
                    : (fakeAct1.ActParam.FusedClamp[i].Min * act2.Value.Cast<float>()[new int[2] { i, 1 }] +
                       act2.Value.Cast<float>()[new int[2] { i, 3 }]);
                actFoldParam.FusedClamp[i] = new ValueRange<Half>
                {
                    Min = (clampMin > fakeAct2.ActParam.FusedClamp[i].Min)
                        ? ((Half)clampMin)
                        : ((Half)fakeAct2.ActParam.FusedClamp[i].Min),
                    Max = (Half)fakeAct2.ActParam.FusedClamp[i].Max,
                };
                if (actFoldParam.FusedClamp[i].Max < actFoldParam.FusedClamp[i].Min)
                {
                    return null;
                }
            }
            else if (act2.Value.Cast<float>()[new int[2] { i, 1 }] == 0f &&
                     act2.Value.Cast<float>()[new int[2] { i, 2 }] > 0f)
            {
                float clampMax = double.IsPositiveInfinity(fakeAct1.ActParam.FusedClamp[i].Max)
                    ? fakeAct1.ActParam.FusedClamp[i].Max
                    : (fakeAct1.ActParam.FusedClamp[i].Max * act2.Value.Cast<float>()[new int[2] { i, 2 }] +
                       act2.Value.Cast<float>()[new int[2] { i, 4 }]);
                actFoldParam.FusedClamp[i] = new ValueRange<Half>
                {
                    Min = (Half)fakeAct2.ActParam.FusedClamp[i].Min,
                    Max = (clampMax < fakeAct2.ActParam.FusedClamp[i].Max)
                        ? ((Half)clampMax)
                        : ((Half)fakeAct2.ActParam.FusedClamp[i].Max),
                };
                if (actFoldParam.FusedClamp[i].Max < actFoldParam.FusedClamp[i].Min)
                {
                    return null;
                }
            }
            else if (act2.Value.Cast<float>()[new int[2] { i, 1 }] < 0f &&
                     act2.Value.Cast<float>()[new int[2] { i, 2 }] == 0f)
            {
                float clampMax = double.IsNegativeInfinity(fakeAct1.ActParam.FusedClamp[i].Min)
                    ? (0f - fakeAct1.ActParam.FusedClamp[i].Min)
                    : (fakeAct1.ActParam.FusedClamp[i].Min * act2.Value.Cast<float>()[new int[2] { i, 1 }] +
                       act2.Value.Cast<float>()[new int[2] { i, 3 }]);
                actFoldParam.FusedClamp[i] = new ValueRange<Half>
                {
                    Min = (Half)fakeAct2.ActParam.FusedClamp[i].Min,
                    Max = (clampMax < fakeAct2.ActParam.FusedClamp[i].Max)
                        ? ((Half)clampMax)
                        : ((Half)fakeAct2.ActParam.FusedClamp[i].Max),
                };
                if (actFoldParam.FusedClamp[i].Max < actFoldParam.FusedClamp[i].Min)
                {
                    return null;
                }
            }
            else if (act2.Value.Cast<float>()[new int[2] { i, 1 }] == 0f &&
                     act2.Value.Cast<float>()[new int[2] { i, 2 }] < 0f)
            {
                float clampMin = double.IsPositiveInfinity(fakeAct1.ActParam.FusedClamp[i].Max)
                    ? (0f - fakeAct1.ActParam.FusedClamp[i].Max)
                    : (fakeAct1.ActParam.FusedClamp[i].Max * act2.Value.Cast<float>()[new int[2] { i, 2 }] +
                       act2.Value.Cast<float>()[new int[2] { i, 4 }]);
                actFoldParam.FusedClamp[i] = new ValueRange<Half>
                {
                    Min = (clampMin > fakeAct2.ActParam.FusedClamp[i].Min)
                        ? ((Half)clampMin)
                        : ((Half)fakeAct2.ActParam.FusedClamp[i].Min),
                    Max = (Half)fakeAct2.ActParam.FusedClamp[i].Max,
                };
                if (actFoldParam.FusedClamp[i].Max < actFoldParam.FusedClamp[i].Min)
                {
                    return null;
                }
            }
            else if (act2.Value.Cast<float>()[new int[2] { i, 1 }] == 0f &&
                     act2.Value.Cast<float>()[new int[2] { i, 2 }] == 0f)
            {
                actFoldParam.FusedClamp[i] = new ValueRange<Half>
                {
                    Min = (Half)fakeAct2.ActParam.FusedClamp[i].Min, Max = (Half)fakeAct2.ActParam.FusedClamp[i].Max,
                };
            }
        }

        for (int j = 0; j < outchannels1.Value.ToScalar<int>(); j++)
        {
            if (act1.Value.Cast<float>()[new int[2] { j, 1 }] > 0f &&
                act1.Value.Cast<float>()[new int[2] { j, 2 }] > 0f)
            {
                if (act1.Value.Cast<float>()[new int[2] { j, 0 }] != 0f && System.Math.Abs(
                        act2.Value.Cast<float>()[new int[2] { j, 0 }] -
                        (act1.Value.Cast<float>()[new int[2] { j, 0 }] * act1.Value.Cast<float>()[new int[2] { j, 1 }] +
                         act1.Value.Cast<float>()[new int[2] { j, 3 }])) > float.Epsilon)
                {
                    return null;
                }

                if ((act1.Value.Cast<float>()[new int[2] { j, 3 }] > act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                     act1.Value.Cast<float>()[new int[2] { j, 4 }] < act2.Value.Cast<float>()[new int[2] { j, 0 }]) &&
                    (act2.Value.Cast<float>()[new int[2] { j, 1 }] != act2.Value.Cast<float>()[new int[2] { j, 2 }] ||
                     act2.Value.Cast<float>()[new int[2] { j, 3 }] != act2.Value.Cast<float>()[new int[2] { j, 4 }]) &&
                    (act1.Value.Cast<float>()[new int[2] { j, 1 }] != act1.Value.Cast<float>()[new int[2] { j, 2 }] ||
                     act1.Value.Cast<float>()[new int[2] { j, 3 }] != act1.Value.Cast<float>()[new int[2] { j, 4 }]))
                {
                    return null;
                }

                if (fakeAct1.ActParam.FusedClamp[j].Max < act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                    fakeAct1.ActParam.FusedClamp[j].Min > act2.Value.Cast<float>()[new int[2] { j, 0 }])
                {
                    return null;
                }

                if (act1.Value.Cast<float>()[new int[2] { j, 1 }] == act1.Value.Cast<float>()[new int[2] { j, 2 }] &&
                    act1.Value.Cast<float>()[new int[2] { j, 3 }] == act1.Value.Cast<float>()[new int[2] { j, 4 }] &&
                    (act2.Value.Cast<float>()[new int[2] { j, 1 }] != act2.Value.Cast<float>()[new int[2] { j, 2 }] ||
                     act2.Value.Cast<float>()[new int[2] { j, 3 }] != act2.Value.Cast<float>()[new int[2] { j, 4 }]))
                {
                    actFoldParam.Ks[0, j] = act1.Value.Cast<float>()[new int[2] { j, 1 }] *
                                            act2.Value.Cast<float>()[new int[2] { j, 1 }];
                    actFoldParam.Ks[1, j] = act1.Value.Cast<float>()[new int[2] { j, 2 }] *
                                            act2.Value.Cast<float>()[new int[2] { j, 2 }];
                    actFoldParam.Bs[0, j] =
                        act1.Value.Cast<float>()[new int[2] { j, 3 }] * act2.Value.Cast<float>()[new int[2] { j, 1 }] +
                        act2.Value.Cast<float>()[new int[2] { j, 3 }];
                    actFoldParam.Bs[1, j] =
                        act1.Value.Cast<float>()[new int[2] { j, 4 }] * act2.Value.Cast<float>()[new int[2] { j, 2 }] +
                        act2.Value.Cast<float>()[new int[2] { j, 4 }];
                    actFoldParam.Xs[0, j] =
                        (act2.Value.Cast<float>()[new int[2] { j, 0 }] -
                         act1.Value.Cast<float>()[new int[2] { j, 3 }]) / act1.Value.Cast<float>()[new int[2] { j, 1 }];
                }
                else
                {
                    actFoldParam.Ks[0, j] = act1.Value.Cast<float>()[new int[2] { j, 1 }] *
                                            act2.Value.Cast<float>()[new int[2] { j, 1 }];
                    actFoldParam.Ks[1, j] = act1.Value.Cast<float>()[new int[2] { j, 2 }] *
                                            act2.Value.Cast<float>()[new int[2] { j, 2 }];
                    actFoldParam.Bs[0, j] =
                        act1.Value.Cast<float>()[new int[2] { j, 3 }] * act2.Value.Cast<float>()[new int[2] { j, 1 }] +
                        act2.Value.Cast<float>()[new int[2] { j, 3 }];
                    actFoldParam.Bs[1, j] =
                        act1.Value.Cast<float>()[new int[2] { j, 4 }] * act2.Value.Cast<float>()[new int[2] { j, 2 }] +
                        act2.Value.Cast<float>()[new int[2] { j, 4 }];
                    actFoldParam.Xs[0, j] = act1.Value.Cast<float>()[new int[2] { j, 0 }];
                }
            }
            else if (act1.Value.Cast<float>()[new int[2] { j, 1 }] < 0f &&
                     act1.Value.Cast<float>()[new int[2] { j, 2 }] < 0f)
            {
                if (act1.Value.Cast<float>()[new int[2] { j, 0 }] != 0f && System.Math.Abs(
                        act2.Value.Cast<float>()[new int[2] { j, 0 }] -
                        (act1.Value.Cast<float>()[new int[2] { j, 0 }] * act1.Value.Cast<float>()[new int[2] { j, 1 }] +
                         act1.Value.Cast<float>()[new int[2] { j, 3 }])) > float.Epsilon)
                {
                    return null;
                }

                if ((act1.Value.Cast<float>()[new int[2] { j, 3 }] < act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                     act1.Value.Cast<float>()[new int[2] { j, 4 }] > act2.Value.Cast<float>()[new int[2] { j, 0 }]) &&
                    (act2.Value.Cast<float>()[new int[2] { j, 1 }] != act2.Value.Cast<float>()[new int[2] { j, 2 }] ||
                     act2.Value.Cast<float>()[new int[2] { j, 3 }] != act2.Value.Cast<float>()[new int[2] { j, 4 }]) &&
                    (act1.Value.Cast<float>()[new int[2] { j, 1 }] != act1.Value.Cast<float>()[new int[2] { j, 2 }] ||
                     act1.Value.Cast<float>()[new int[2] { j, 3 }] != act1.Value.Cast<float>()[new int[2] { j, 4 }]))
                {
                    return null;
                }

                if (fakeAct1.ActParam.FusedClamp[j].Max < act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                    fakeAct1.ActParam.FusedClamp[j].Min > act2.Value.Cast<float>()[new int[2] { j, 0 }])
                {
                    return null;
                }

                if (act1.Value.Cast<float>()[new int[2] { j, 1 }] == act1.Value.Cast<float>()[new int[2] { j, 2 }] &&
                    act1.Value.Cast<float>()[new int[2] { j, 3 }] == act1.Value.Cast<float>()[new int[2] { j, 4 }] &&
                    (act2.Value.Cast<float>()[new int[2] { j, 1 }] != act2.Value.Cast<float>()[new int[2] { j, 2 }] ||
                     act2.Value.Cast<float>()[new int[2] { j, 3 }] != act2.Value.Cast<float>()[new int[2] { j, 4 }]))
                {
                    actFoldParam.Ks[0, j] = act1.Value.Cast<float>()[new int[2] { j, 1 }] *
                                            act2.Value.Cast<float>()[new int[2] { j, 2 }];
                    actFoldParam.Ks[1, j] = act1.Value.Cast<float>()[new int[2] { j, 2 }] *
                                            act2.Value.Cast<float>()[new int[2] { j, 1 }];
                    actFoldParam.Bs[0, j] =
                        act1.Value.Cast<float>()[new int[2] { j, 3 }] * act2.Value.Cast<float>()[new int[2] { j, 2 }] +
                        act2.Value.Cast<float>()[new int[2] { j, 4 }];
                    actFoldParam.Bs[1, j] =
                        act1.Value.Cast<float>()[new int[2] { j, 4 }] * act2.Value.Cast<float>()[new int[2] { j, 1 }] +
                        act2.Value.Cast<float>()[new int[2] { j, 3 }];
                    actFoldParam.Xs[0, j] =
                        (act2.Value.Cast<float>()[new int[2] { j, 0 }] -
                         act1.Value.Cast<float>()[new int[2] { j, 3 }]) / act1.Value.Cast<float>()[new int[2] { j, 1 }];
                }
                else
                {
                    actFoldParam.Ks[0, j] = act1.Value.Cast<float>()[new int[2] { j, 1 }] *
                                            act2.Value.Cast<float>()[new int[2] { j, 1 }];
                    actFoldParam.Ks[1, j] = act1.Value.Cast<float>()[new int[2] { j, 2 }] *
                                            act2.Value.Cast<float>()[new int[2] { j, 2 }];
                    actFoldParam.Bs[0, j] =
                        act1.Value.Cast<float>()[new int[2] { j, 3 }] * act2.Value.Cast<float>()[new int[2] { j, 1 }] +
                        act2.Value.Cast<float>()[new int[2] { j, 3 }];
                    actFoldParam.Bs[1, j] =
                        act1.Value.Cast<float>()[new int[2] { j, 4 }] * act2.Value.Cast<float>()[new int[2] { j, 2 }] +
                        act2.Value.Cast<float>()[new int[2] { j, 4 }];
                    actFoldParam.Xs[0, j] = act1.Value.Cast<float>()[new int[2] { j, 0 }];
                }
            }
            else if (act1.Value.Cast<float>()[new int[2] { j, 1 }] == 0f &&
                     act1.Value.Cast<float>()[new int[2] { j, 2 }] > 0f)
            {
                if (act2.Value.Cast<float>()[new int[2] { j, 0 }] > act1.Value.Cast<float>()[new int[2] { j, 0 }] &&
                    act1.Value.Cast<float>()[new int[2] { j, 1 }] == 0f &&
                    act1.Value.Cast<float>()[new int[2] { j, 2 }] == 1f &&
                    act1.Value.Cast<float>()[new int[2] { j, 4 }] == 0f &&
                    act2.Value.Cast<float>()[new int[2] { j, 1 }] == 0f &&
                    act2.Value.Cast<float>()[new int[2] { j, 2 }] == 1f &&
                    act2.Value.Cast<float>()[new int[2] { j, 3 }] == 0f)
                {
                    actFoldParam.Ks[0, j] = 1f;
                    actFoldParam.Ks[1, j] = 0f;
                    actFoldParam.Bs[0, j] = 0f;
                    actFoldParam.Bs[1, j] = act2.Value.Cast<float>()[new int[2] { j, 0 }];
                    actFoldParam.Xs[0, j] = act2.Value.Cast<float>()[new int[2] { j, 0 }];
                    actFoldParam.FusedClamp[j].Min = (Half)act1.Value.Cast<float>()[new int[2] { j, 0 }];
                    actFoldParam.FusedClamp[j].Max = (Half)act2.Value.Cast<float>()[new int[2] { j, 0 }];
                    continue;
                }

                if ((act1.Value.Cast<float>()[new int[2] { j, 3 }] >= act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                     act1.Value.Cast<float>()[new int[2] { j, 4 }] + act1.Value.Cast<float>()[new int[2] { j, 0 }] *
                     act1.Value.Cast<float>()[new int[2] { j, 2 }] <= act2.Value.Cast<float>()[new int[2] { j, 0 }]) &&
                    (act2.Value.Cast<float>()[new int[2] { j, 1 }] != act2.Value.Cast<float>()[new int[2] { j, 2 }] ||
                     act2.Value.Cast<float>()[new int[2] { j, 3 }] != act2.Value.Cast<float>()[new int[2] { j, 4 }]))
                {
                    return null;
                }

                if (fakeAct1.ActParam.FusedClamp[j].Max < act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                    fakeAct1.ActParam.FusedClamp[j].Min > act2.Value.Cast<float>()[new int[2] { j, 0 }])
                {
                    return null;
                }

                actFoldParam.Ks[0, j] = 0f;
                actFoldParam.Ks[1, j] = act1.Value.Cast<float>()[new int[2] { j, 2 }] *
                                        act2.Value.Cast<float>()[new int[2] { j, 2 }];
                actFoldParam.Bs[0, j] =
                    act1.Value.Cast<float>()[new int[2] { j, 3 }] * act2.Value.Cast<float>()[new int[2] { j, 1 }] +
                    act2.Value.Cast<float>()[new int[2] { j, 3 }];
                actFoldParam.Bs[1, j] =
                    act1.Value.Cast<float>()[new int[2] { j, 4 }] * act2.Value.Cast<float>()[new int[2] { j, 2 }] +
                    act2.Value.Cast<float>()[new int[2] { j, 4 }];
                actFoldParam.Xs[0, j] = act1.Value.Cast<float>()[new int[2] { j, 0 }];
            }
            else if (act1.Value.Cast<float>()[new int[2] { j, 1 }] > 0f &&
                     act1.Value.Cast<float>()[new int[2] { j, 2 }] == 0f)
            {
                if (act2.Value.Cast<float>()[new int[2] { j, 0 }] < act1.Value.Cast<float>()[new int[2] { j, 0 }] &&
                    act1.Value.Cast<float>()[new int[2] { j, 1 }] == 1f &&
                    act1.Value.Cast<float>()[new int[2] { j, 2 }] == 0f &&
                    act1.Value.Cast<float>()[new int[2] { j, 3 }] == 0f &&
                    act2.Value.Cast<float>()[new int[2] { j, 1 }] == 0f &&
                    act2.Value.Cast<float>()[new int[2] { j, 2 }] == 1f &&
                    act2.Value.Cast<float>()[new int[2] { j, 4 }] == 0f)
                {
                    actFoldParam.Ks[0, j] = 1f;
                    actFoldParam.Ks[1, j] = 0f;
                    actFoldParam.Bs[0, j] = 0f;
                    actFoldParam.Bs[1, j] = act1.Value.Cast<float>()[new int[2] { j, 0 }];
                    actFoldParam.Xs[0, j] = act1.Value.Cast<float>()[new int[2] { j, 0 }];
                    actFoldParam.FusedClamp[j].Min = (Half)act2.Value.Cast<float>()[new int[2] { j, 0 }];
                    actFoldParam.FusedClamp[j].Max = (Half)act1.Value.Cast<float>()[new int[2] { j, 0 }];
                    continue;
                }

                if ((act1.Value.Cast<float>()[new int[2] { j, 3 }] + act1.Value.Cast<float>()[new int[2] { j, 0 }] *
                     act1.Value.Cast<float>()[new int[2] { j, 1 }] >=
                     act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                     act1.Value.Cast<float>()[new int[2] { j, 4 }] <=
                     act2.Value.Cast<float>()[new int[2] { j, 0 }]) &&
                    (act2.Value.Cast<float>()[new int[2] { j, 1 }] != act2.Value.Cast<float>()[new int[2] { j, 2 }] ||
                     act2.Value.Cast<float>()[new int[2] { j, 3 }] != act2.Value.Cast<float>()[new int[2] { j, 4 }]))
                {
                    return null;
                }

                if (fakeAct1.ActParam.FusedClamp[j].Max < act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                    fakeAct1.ActParam.FusedClamp[j].Min > act2.Value.Cast<float>()[new int[2] { j, 0 }])
                {
                    return null;
                }

                actFoldParam.Ks[0, j] = act1.Value.Cast<float>()[new int[2] { j, 1 }] *
                                        act2.Value.Cast<float>()[new int[2] { j, 1 }];
                actFoldParam.Ks[1, j] = 0f;
                actFoldParam.Bs[0, j] =
                    act1.Value.Cast<float>()[new int[2] { j, 3 }] * act2.Value.Cast<float>()[new int[2] { j, 1 }] +
                    act2.Value.Cast<float>()[new int[2] { j, 3 }];
                actFoldParam.Bs[1, j] =
                    act1.Value.Cast<float>()[new int[2] { j, 4 }] * act2.Value.Cast<float>()[new int[2] { j, 2 }] +
                    act2.Value.Cast<float>()[new int[2] { j, 4 }];
                actFoldParam.Xs[0, j] = act1.Value.Cast<float>()[new int[2] { j, 0 }];
            }
            else if (act1.Value.Cast<float>()[new int[2] { j, 1 }] == 0f &&
                     act1.Value.Cast<float>()[new int[2] { j, 2 }] < 0f)
            {
                if (act2.Value.Cast<float>()[new int[2] { j, 0 }] > act1.Value.Cast<float>()[new int[2] { j, 0 }] &&
                    act1.Value.Cast<float>()[new int[2] { j, 1 }] == 0f &&
                    act1.Value.Cast<float>()[new int[2] { j, 2 }] == -1f &&
                    act1.Value.Cast<float>()[new int[2] { j, 4 }] == 0f &&
                    act2.Value.Cast<float>()[new int[2] { j, 1 }] == -1f &&
                    act2.Value.Cast<float>()[new int[2] { j, 2 }] == 0f &&
                    act2.Value.Cast<float>()[new int[2] { j, 3 }] == 0f)
                {
                    actFoldParam.Ks[0, j] = 1f;
                    actFoldParam.Ks[1, j] = -1f;
                    actFoldParam.Bs[0, j] = act1.Value.Cast<float>()[new int[2] { j, 0 }];
                    actFoldParam.Bs[1, j] = 0f;
                    actFoldParam.Xs[0, j] = act1.Value.Cast<float>()[new int[2] { j, 0 }];
                    actFoldParam.FusedClamp[j].Min = (Half)(0f - act2.Value.Cast<float>()[new int[2] { j, 0 }]);
                    actFoldParam.FusedClamp[j].Max = (Half)(0f - act1.Value.Cast<float>()[new int[2] { j, 0 }]);
                    continue;
                }

                if ((act1.Value.Cast<float>()[new int[2] { j, 3 }] <= act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                     act1.Value.Cast<float>()[new int[2] { j, 4 }] - act1.Value.Cast<float>()[new int[2] { j, 0 }] *
                     act1.Value.Cast<float>()[new int[2] { j, 2 }] >= act2.Value.Cast<float>()[new int[2] { j, 0 }]) &&
                    (act2.Value.Cast<float>()[new int[2] { j, 1 }] != act2.Value.Cast<float>()[new int[2] { j, 2 }] ||
                     act2.Value.Cast<float>()[new int[2] { j, 3 }] != act2.Value.Cast<float>()[new int[2] { j, 4 }]))
                {
                    return null;
                }

                if (fakeAct1.ActParam.FusedClamp[j].Max < act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                    fakeAct1.ActParam.FusedClamp[j].Min > act2.Value.Cast<float>()[new int[2] { j, 0 }])
                {
                    return null;
                }

                actFoldParam.Ks[0, j] = 0f;
                actFoldParam.Ks[1, j] = act1.Value.Cast<float>()[new int[2] { j, 2 }] *
                                        act2.Value.Cast<float>()[new int[2] { j, 1 }];
                actFoldParam.Bs[0, j] =
                    act1.Value.Cast<float>()[new int[2] { j, 3 }] * act2.Value.Cast<float>()[new int[2] { j, 2 }] +
                    act2.Value.Cast<float>()[new int[2] { j, 4 }];
                actFoldParam.Bs[1, j] =
                    act1.Value.Cast<float>()[new int[2] { j, 4 }] * act2.Value.Cast<float>()[new int[2] { j, 1 }] +
                    act2.Value.Cast<float>()[new int[2] { j, 3 }];
                actFoldParam.Xs[0, j] = act1.Value.Cast<float>()[new int[2] { j, 0 }];
            }
            else if (act1.Value.Cast<float>()[new int[2] { j, 1 }] < 0f &&
                     act1.Value.Cast<float>()[new int[2] { j, 2 }] == 0f)
            {
                if (act2.Value.Cast<float>()[new int[2] { j, 0 }] < act1.Value.Cast<float>()[new int[2] { j, 0 }] &&
                    act1.Value.Cast<float>()[new int[2] { j, 1 }] == -1f &&
                    act1.Value.Cast<float>()[new int[2] { j, 2 }] == 0f &&
                    act1.Value.Cast<float>()[new int[2] { j, 3 }] == 0f &&
                    act2.Value.Cast<float>()[new int[2] { j, 1 }] == 0f &&
                    act2.Value.Cast<float>()[new int[2] { j, 2 }] == -1f &&
                    act2.Value.Cast<float>()[new int[2] { j, 4 }] == 0f)
                {
                    actFoldParam.Ks[0, j] = 1f;
                    actFoldParam.Ks[1, j] = -1f;
                    actFoldParam.Bs[0, j] = 0f - act1.Value.Cast<float>()[new int[2] { j, 0 }];
                    actFoldParam.Bs[1, j] = 0f;
                    actFoldParam.Xs[0, j] = act1.Value.Cast<float>()[new int[2] { j, 0 }];
                    actFoldParam.FusedClamp[j].Min = (Half)(0f - act1.Value.Cast<float>()[new int[2] { j, 0 }]);
                    actFoldParam.FusedClamp[j].Max = (Half)(0f - act2.Value.Cast<float>()[new int[2] { j, 0 }]);
                    continue;
                }

                if ((act1.Value.Cast<float>()[new int[2] { j, 3 }] - act1.Value.Cast<float>()[new int[2] { j, 0 }] *
                     act1.Value.Cast<float>()[new int[2] { j, 1 }] <=
                     act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                     act1.Value.Cast<float>()[new int[2] { j, 4 }] >=
                     act2.Value.Cast<float>()[new int[2] { j, 0 }]) &&
                    (act2.Value.Cast<float>()[new int[2] { j, 1 }] != act2.Value.Cast<float>()[new int[2] { j, 2 }] ||
                     act2.Value.Cast<float>()[new int[2] { j, 3 }] != act2.Value.Cast<float>()[new int[2] { j, 4 }]))
                {
                    return null;
                }

                if (fakeAct1.ActParam.FusedClamp[j].Max < act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                    fakeAct1.ActParam.FusedClamp[j].Min > act2.Value.Cast<float>()[new int[2] { j, 0 }])
                {
                    return null;
                }

                actFoldParam.Ks[0, j] = act1.Value.Cast<float>()[new int[2] { j, 1 }] *
                                        act2.Value.Cast<float>()[new int[2] { j, 2 }];
                actFoldParam.Ks[1, j] = 0f;
                actFoldParam.Bs[0, j] =
                    act1.Value.Cast<float>()[new int[2] { j, 3 }] * act2.Value.Cast<float>()[new int[2] { j, 2 }] +
                    act2.Value.Cast<float>()[new int[2] { j, 4 }];
                actFoldParam.Bs[1, j] =
                    act1.Value.Cast<float>()[new int[2] { j, 4 }] * act2.Value.Cast<float>()[new int[2] { j, 1 }] +
                    act2.Value.Cast<float>()[new int[2] { j, 3 }];
                actFoldParam.Xs[0, j] = act1.Value.Cast<float>()[new int[2] { j, 0 }];
            }
            else if (act1.Value.Cast<float>()[new int[2] { j, 1 }] == 0f &&
                     act1.Value.Cast<float>()[new int[2] { j, 2 }] == 0f)
            {
                if ((act1.Value.Cast<float>()[new int[2] { j, 3 }] >= act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                     act1.Value.Cast<float>()[new int[2] { j, 4 }] <= act2.Value.Cast<float>()[new int[2] { j, 0 }]) &&
                    (act2.Value.Cast<float>()[new int[2] { j, 1 }] != act2.Value.Cast<float>()[new int[2] { j, 2 }] ||
                     act2.Value.Cast<float>()[new int[2] { j, 3 }] != act2.Value.Cast<float>()[new int[2] { j, 4 }]))
                {
                    return null;
                }

                if (fakeAct1.ActParam.FusedClamp[j].Max <= act2.Value.Cast<float>()[new int[2] { j, 0 }] ||
                    fakeAct1.ActParam.FusedClamp[j].Min >= act2.Value.Cast<float>()[new int[2] { j, 0 }])
                {
                    return null;
                }

                actFoldParam.Ks[0, j] = 0f;
                actFoldParam.Ks[1, j] = 0f;
                actFoldParam.Bs[0, j] =
                    act1.Value.Cast<float>()[new int[2] { j, 3 }] * act2.Value.Cast<float>()[new int[2] { j, 1 }] +
                    act2.Value.Cast<float>()[new int[2] { j, 3 }];
                actFoldParam.Bs[1, j] =
                    act1.Value.Cast<float>()[new int[2] { j, 4 }] * act2.Value.Cast<float>()[new int[2] { j, 2 }] +
                    act2.Value.Cast<float>()[new int[2] { j, 4 }];
                actFoldParam.Xs[0, j] = act1.Value.Cast<float>()[new int[2] { j, 0 }];
            }
            else
            {
                if (act2.Value.Cast<float>()[new int[2] { j, 1 }] != act2.Value.Cast<float>()[new int[2] { j, 2 }] ||
                    act2.Value.Cast<float>()[new int[2] { j, 3 }] != act2.Value.Cast<float>()[new int[2] { j, 4 }])
                {
                    return null;
                }

                actFoldParam.Ks[0, j] = act1.Value.Cast<float>()[new int[2] { j, 1 }] *
                                        act2.Value.Cast<float>()[new int[2] { j, 1 }];
                actFoldParam.Ks[1, j] = act1.Value.Cast<float>()[new int[2] { j, 2 }] *
                                        act2.Value.Cast<float>()[new int[2] { j, 2 }];
                actFoldParam.Bs[0, j] =
                    act1.Value.Cast<float>()[new int[2] { j, 3 }] * act2.Value.Cast<float>()[new int[2] { j, 1 }] +
                    act2.Value.Cast<float>()[new int[2] { j, 3 }];
                actFoldParam.Bs[1, j] =
                    act1.Value.Cast<float>()[new int[2] { j, 4 }] * act2.Value.Cast<float>()[new int[2] { j, 2 }] +
                    act2.Value.Cast<float>()[new int[2] { j, 4 }];
                actFoldParam.Xs[0, j] = act1.Value.Cast<float>()[new int[2] { j, 0 }];
            }
        }

        ActParam2 actParam = new ActParam2(outChannels1);
        for (int k = 0; k < outchannels1.Value.ToScalar<int>(); k++)
        {
            actParam.Ks[0, k] = actFoldParam.Ks[0, k];
            actParam.Ks[1, k] = actFoldParam.Ks[1, k];
            actParam.Bs[0, k] = actFoldParam.Bs[0, k];
            actParam.Bs[1, k] = actFoldParam.Bs[1, k];
            actParam.Xs[0, k] = actFoldParam.Xs[0, k];
            actParam.FusedClamp[k].Min = (float)actFoldParam.FusedClamp[k].Min;
            actParam.FusedClamp[k].Max = (float)actFoldParam.FusedClamp[k].Max;
        }

        Tensor<float> tensor = new Tensor<float>(actParam.GetAct1Data, new int[2] { outChannels1, 7 });
        return Nncase.IR.K230.F.Tensors.FakeActivation(inputa, inputb, tensor, outChannels1, 0, 0, 0, false, fakeAct1.Type,
            actParam, call2.CheckedShape.ToValueArray()).InheritMetaData(call2);
    }

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        FakeActivation fakeAct = (FakeActivation)result["fakeAct1"];
        Expr inputa = (Expr)result["inputa"];
        Expr inputb = (Expr)result["inputb"];
        TensorConst act = (TensorConst)result["act1"];
        TensorConst outchannels = (TensorConst)result["outchannels1"];
        FakeActivation fakeAct2 = (FakeActivation)result["fakeAct2"];
        Call call = (Call)result["call1"];
        Call call2 = (Call)result["call2"];
        TensorConst act2 = (TensorConst)result["act2"];
        TensorConst outchannels2 = (TensorConst)result["outchannels2"];
        Expr is16Segments = (Expr)result["is16Segments1"];
        Expr is16Segments2 = (Expr)result["is16Segments2"];
        Expr range = (Expr)result["range"];
        return GetReplace(fakeAct, inputa, inputb, act, outchannels, fakeAct2, call, call2, act2, outchannels2,
            is16Segments, is16Segments2, range, context);
    }
}
