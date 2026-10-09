#include "k230_runtime_module.h"
#include "k230_runtime_function.h"
#include <nncase/functional/k230/dynamic_gnne_matmul.h>
#include <nncase/runtime/runtime_loader.h>
#include <new>

BEGIN_NS_NNCASE_RT_K230

result<void> k230_runtime_module::initialize_before_functions(runtime_module_init_context &context) noexcept
{
    try_set(rdata_, context.get_or_read_section(".rdata", rdata_storage_, true));
    try_set(text_, context.get_or_read_section(".text", text_storage_, true));
    return ok();
}

result<std::unique_ptr<runtime_function>> k230_runtime_module::create_function() noexcept
{
    std::unique_ptr<runtime_function> function(new (std::nothrow) k230_runtime_function(*this));
    if (!function)
        return err(std::errc::not_enough_memory);
    return ok(std::move(function));
}

result<std::unique_ptr<runtime_module>> create_k230_runtime_module()
{
    std::unique_ptr<runtime_module> module(new (std::nothrow) k230_runtime_module());
    if (!module)
        return err(std::errc::not_enough_memory);
    return ok(std::move(module));
}

result<std::vector<std::pair<std::string, runtime_module::custom_call_type>>> create_k230_custom_calls()
{
    std::vector<std::pair<std::string, runtime_module::custom_call_type>> calls;
    calls.emplace_back("K230DynamicGNNEMatMul", nncase::functional::k230::dynamic_gnne_matmul);
    return ok(std::move(calls));
}

END_NS_NNCASE_RT_K230


extern "C" NNCASE_MODULES_K230_API
void create_runtime_module(nncase::result<std::unique_ptr<nncase::runtime::runtime_module>> &result)
{
    result = nncase::runtime::k230::create_k230_runtime_module();
}

extern "C" NNCASE_MODULES_K230_API void collect_custom_call(
    nncase::result<std::vector<std::pair<std::string, nncase::runtime::runtime_module::custom_call_type>>> &result)
{
    result = nncase::runtime::k230::create_k230_custom_calls();
}


BEGIN_NS_NNCASE_RUNTIME
runtime_registration builtin_runtimes[] = {"k230", create_runtime_module,
                                           collect_custom_call};
END_NS_NNCASE_RUNTIME