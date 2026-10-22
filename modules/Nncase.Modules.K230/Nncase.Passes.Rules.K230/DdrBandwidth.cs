// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.K230;

namespace Nncase.Passes.Rules.K230;

public class DdrBandwidth
{
    private long[] _ofBw;

    public long[] TotalBw { get; private set; }

    public long[] IfBw { get; private set; }

    public long[] WBw { get; private set; }

    public long[] OtherBw { get; private set; }

    public DdrBandwidth()
    {
        IfBw = new long[2];
        WBw = new long[2];
        _ofBw = new long[2];
        OtherBw = new long[2];
        TotalBw = new long[2];
    }

    public static int GetBytesPerElement(DataType type)
    {
        if ((object)type != null)
        {
            if (type == DataTypes.Int8 || type == DataTypes.UInt8)
            {
                return 1;
            }

            DataType dataType = type;
            if (dataType == DataTypes.BFloat16 || dataType == DataTypes.Int16 || dataType == DataTypes.Float16)
            {
                return 2;
            }

            DataType typeCopy = type;
            if (typeCopy == DataTypes.Float32 || typeCopy == DataTypes.UInt32 || typeCopy == DataTypes.Int32)
            {
                return 4;
            }
        }

        throw new ArgumentOutOfRangeException(type.GetDisplayName());
    }

    public void CalcBw(GnneAction action)
    {
        long bytes = 0L;
        switch (action.Name)
        {
            case GnneActionName.L2Load:
                {
                    GnneActionL2Load l2Load = (GnneActionL2Load)action;
                    int loadSize = l2Load.SliceInfo.Shape_size;
                    int bytesPerElement3 = GetBytesPerElement(l2Load.Input[GNNELoad.Input].CheckedDataType);
                    IfBw[0] += loadSize * bytesPerElement3;
                    TotalBw[0] += loadSize * bytesPerElement3;
                    break;
                }

            case GnneActionName.L2LoadWConf:
                {
                    GnneActionL2LoadWConf l2LoadWConf = (GnneActionL2LoadWConf)action;
                    int bytesPerElement2 = GetBytesPerElement(l2LoadWConf.DdrDatatype);
                    bytes = ((TensorConst)l2LoadWConf.LenCompressed.Value).Value.ToScalar<int>() * bytesPerElement2;
                    break;
                }

            case GnneActionName.L2LoadW:
                WBw[0] += bytes;
                TotalBw[0] += bytes;
                break;
            case GnneActionName.L2Store:
                {
                    GnneActionL2Store l2Store = (GnneActionL2Store)action;
                    int bytesPerElement = GetBytesPerElement(l2Store.Output.CheckedDataType);
                    int storeSize = l2Store.SliceInfo.Shape_size;
                    _ofBw[0] += storeSize * bytesPerElement;
                    TotalBw[0] += storeSize * bytesPerElement;
                    break;
                }

            case GnneActionName.L2StoreConf:
                break;
        }
    }

    public void ResetBw()
    {
        IfBw = new long[2];
        WBw = new long[2];
        _ofBw = new long[2];
        OtherBw = new long[2];
        TotalBw = new long[2];
    }
}
