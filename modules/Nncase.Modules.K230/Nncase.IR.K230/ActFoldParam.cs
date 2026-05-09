using System;
using System.Linq;

namespace Nncase.IR.K230;

public class ActFoldParam
{
    private float[,] _xs;

    private float[,] _ks;

    private float[,] _bs;

    private ValueRange<Half>[] _fusedClamp;

    public Shape Shape => new Shape(1, 1, Channels, DataSizePerChannel);

    public float[,] Xs
    {
        get { return _xs; }
        set { _xs = value ?? throw new ArgumentNullException("value"); }
    }

    public float[,] Ks
    {
        get { return _ks; }
        set { _ks = value ?? throw new ArgumentNullException("value"); }
    }

    public float[,] Bs
    {
        get { return _bs; }
        set { _bs = value ?? throw new ArgumentNullException("value"); }
    }

    public ValueRange<Half>[] FusedClamp
    {
        get { return _fusedClamp; }
        set { _fusedClamp = value ?? throw new ArgumentNullException("value"); }
    }

    public ValueRange<Half>[] GetFusedClampArray => FusedClamp;

    private int DataSizePerChannel => 7;

    private int Channels => Xs.GetLength(1);

    public ActFoldParam()
    {
    }

    public ActFoldParam(int channels)
    {
        Enumerable.Repeat(0, channels).Select((Func<int, float>)((int x) => x));
        Enumerable.Repeat(1, channels).Select((Func<int, float>)((int x) => x));
        Xs = new float[1, channels];
        Ks = new float[2, channels];
        Bs = new float[2, channels];
        for (int num = 0; num < 2; num++)
        {
            for (int num2 = 0; num2 < channels; num2++)
            {
                Ks[num, num2] = 1f;
            }
        }

        SetFusedClamp(ValueRange<Half>.Full);
    }

    public void SetFusedClamp(ValueRange<Half> clamp)
    {
        FusedClamp = Enumerable.Repeat(clamp, Channels).ToArray();
    }

    public void SetFusedClamp1(ValueRange<float> clamp)
    {
        FusedClamp = (from _ in Enumerable.Range(0, Channels)
            select new ValueRange<Half> { Min = (Half)clamp.Min, Max = (Half)clamp.Max }).ToArray();
    }
}
