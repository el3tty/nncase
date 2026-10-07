// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Evaluator.K230;
using Nncase.Hosting;

namespace Nncase;

public static class K230ApplicationPart
{
    public static IRegistrator AddK230(this IRegistrator registrator)
    {
        return registrator.RegisterModule<K230CoreModule>().RegisterModule<K230EvalModule>();
    }
}
