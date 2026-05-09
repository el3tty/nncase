using Nncase.IR.K230;
using Nncase.Passes.Rules.Neutral;

namespace Nncase.Passes.Rules.K230;

public class GNNELoadStoreFusion : DataTransferFusion<GNNELoad, GNNEStore>
{
    public override string Name { get; } = "TileLoadStoreCase";

    public override string ModuleKind { get; } = "k230";
}
