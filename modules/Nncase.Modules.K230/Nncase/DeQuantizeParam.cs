using System;

namespace Nncase;

public struct DeQuantizeParam(int zeroPoint, float scale) : IEquatable<DeQuantizeParam>
{
    public int ZeroPoint = zeroPoint;

    public float Scale = scale;

    public bool Equals(DeQuantizeParam other)
    {
        if (Scale.Equals(other.Scale))
        {
            return ZeroPoint == other.ZeroPoint;
        }

        return false;
    }

    public override string ToString()
    {
        return $"<{ZeroPoint}, {Scale}>";
    }

    public override bool Equals(object? obj)
    {
        if (obj is DeQuantizeParam other)
        {
            return Equals(other);
        }

        return false;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(ZeroPoint, Scale);
    }
}
