#include <poll.h>
#include <nncase/functional/k230/dynamic_function.h>
#include <nncase/runtime/k230/gnne.h>

BEGIN_NS_NNCASE_FUNCTIONAL_K230
using namespace nncase;
using namespace nncase::runtime;

result<void> dynamic_function_invoke_core(
    gsl::span<const gsl::byte> text,
    gsl::span<const object_t<value_node>> /*params*/,
    object_t<value_node> output,
    const kernels::kernel_context & /*context*/) noexcept {

    // 1. Enable the GNNE device with the compiled kernel text.
    //    On failure, dbg() prints "gnne_enable(...) == 0 = false (bool)" to stderr
    //    (source line 557 in the original file).
    if (gnne_enable((uint64_t)text.begin(), (uint64_t)text.end(), 0) != 0) {
        std::cerr << "gnne_enable failed\n";
        return err(std::errc::device_or_resource_busy);
    }

    // 2. Block until the device signals completion (stdin fd, infinite timeout).
    pollfd fds{};
    fds.fd = 0;
    fds.events = POLLIN;
    poll(&fds, 1, -1);

    // 3. Nothing to synchronize if there is no output object.
    if (!output.get())
        return ok();

    // 4. Single tensor: invalidate the host cache so the CPU sees device writes.
    if (output.is_a<tensor>()) {
        try_var(t, output.as<tensor>());
        try_var(host, t->buffer().as_host());
        (void)host.sync(sync_invalidate, true);
        return ok();
    }

    // 5. Tuple: do the same for every tensor field.
    if (output.is_a<tuple>()) {
        try_var(tp, output.as<tuple>());
        for (auto &field : tp->fields()) {
            try_var(t, field.as<tensor>());
            try_var(host, t->buffer().as_host());
            (void)host.sync(sync_invalidate, true);
        }
        return ok();
    }

    // Any other object type: nothing to do.
    return ok();
}

END_NS_NNCASE_FUNCTIONAL_K230
