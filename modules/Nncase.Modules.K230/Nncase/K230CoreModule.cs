using DryIoc;
using Nncase.Hosting;
using Nncase.Targets;

namespace Nncase;

internal sealed class K230CoreModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        Registrator.Register<ITarget, K230Target>(registrator, Reuse.Singleton, (Made)null, (Setup)null,
            (IfAlreadyRegistered?)null, (object)null);
        Registrator.Register<ValueType, QuantizeParamType>(registrator, Reuse.Singleton, (Made)null, (Setup)null,
            (IfAlreadyRegistered?)null, (object)null);
        Registrator.Register<ValueType, DeQuantizeParamType>(registrator, Reuse.Singleton, (Made)null, (Setup)null,
            (IfAlreadyRegistered?)null, (object)null);
        Registrator.Register<ValueType, CropBBoxType>(registrator, Reuse.Singleton, (Made)null, (Setup)null,
            (IfAlreadyRegistered?)null, (object)null);
    }
}
