using System;
using System.IO;
using System.Reactive;
using Nncase.IR;
using Nncase.IR.K230;

namespace Nncase.Passes;

internal sealed class INIVisitor : ExprVisitor<Unit, Unit>
{
    private readonly string _dir;

    private int _layerNum;

    public INIVisitor(string dir)
        : base(false)
    {
        _dir = dir;
        Directory.CreateDirectory(dir);
    }

    public void Dump(string content)
    {
        using StreamWriter streamWriter = new StreamWriter(_dir + "/" + _layerNum + ".ini");
        streamWriter.WriteLine(content);
    }

    protected override Unit DefaultVisitLeaf(Expr expr)
    {
        //IL_0002: Unknown result type (might be due to invalid IL or missing references)
        //IL_0008: Unknown result type (might be due to invalid IL or missing references)
        return default(Unit);
    }

    protected override Unit VisitLeafCall(Call expr)
    {
        //IL_0088: Unknown result type (might be due to invalid IL or missing references)
        //IL_008e: Unknown result type (might be due to invalid IL or missing references)
        if ((object)expr != null)
        {
            if (!(expr.Target is FakeMatMul))
            {
                throw new NotImplementedException("not supported Op: " + expr.Target.ToString());
            }

            string content = "fake_matmul";
            _layerNum++;
            Dump(content);
        }
        else if ((object)expr == null || !(expr.Target is Op))
        {
            throw new ArgumentOutOfRangeException("expr",
                "Target Type " + expr.Target.GetType().Name + " not support!");
        }

        return default(Unit);
    }
}
