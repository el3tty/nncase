// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase;

public sealed class K230Plugin : IPlugin, IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.AddK230();
    }
}
