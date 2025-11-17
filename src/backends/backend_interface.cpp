#include "backends/backend_interface.hpp"

using CreateFn = FuzzBackend*();
using DestroyFn = void(FuzzBackend*);

struct BackendHandle {
    void* handle;
    DestroyFn* destroy;
};

static BackendHandle g_backend_handle;

FuzzBackend* load_backend(const std::string& so_path) {
    void* handle = dlopen(so_path.c_str(), RTLD_NOW);
    if (!handle) {
        std::cerr << "dlopen failed: " << dlerror() << std::endl;
        return nullptr;
    }

    auto create_fn = (CreateFn*)dlsym(handle, "create_backend");
    g_backend_handle.destroy = (DestroyFn*)dlsym(handle, "destroy_backend");

    if (!create_fn || !g_backend_handle.destroy) {
        std::cerr << "Missing backend symbols in " << so_path << std::endl;
        dlclose(handle);
        return nullptr;
    }

    g_backend_handle.handle = handle;
    return create_fn();
}

void unload_backend(FuzzBackend* backend) {
    if (g_backend_handle.destroy) g_backend_handle.destroy(backend);
    if (g_backend_handle.handle) dlclose(g_backend_handle.handle);
    g_backend_handle = {};
}
