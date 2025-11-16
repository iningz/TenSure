// src/main.cpp
#include <nlohmann/json.hpp>
#include <signal.h>
#include <thread>
#include <chrono>
#include <filesystem>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <dlfcn.h>
#include <future>
#include "tensure/logger.hpp"

#include "tensure/random_gen.hpp"                // your generator helpers (tsTensor, etc.)
#include "backends/backend_interface.hpp"       // FuzzBackend interface

namespace fs = std::filesystem;
using namespace std::chrono_literals;
using namespace std;

static inline bool g_terminate = false;
void signal_handler(int signum) {
    std::cerr << "Signal " << signum << " received. Will terminate after current iteration.\n";
    g_terminate = true;
}

// timestamp helper (kept from your original)
std::string timestamp_str() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t = system_clock::to_time_t(now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", std::localtime(&t));
    return std::string(buf);
}

// ---------- timeout runner ----------
int run_with_timeout(FuzzBackend* backend, const std::string& kernel_path, const std::string& out_dir, uint64_t timeout_ms)
{
    // Define the callable you want to run asynchronously
    auto task = [backend, kernel_path, out_dir]() -> int {
        // Example: actually run your kernel
        return backend->execute_kernel(kernel_path, out_dir);
    };

    // Launch asynchronously
    std::future<int> fut = std::async(std::launch::async, task);

    // Wait for completion or timeout
    if (fut.wait_for(std::chrono::milliseconds(timeout_ms)) == std::future_status::ready) {
        try {
            return fut.get();  // Get result if finished
        } catch (const std::exception& e) {
            std::cerr << "Exception from timed task: " << e.what() << std::endl;
            return -1;  // Error during execution
        }
    } else {
        std::cerr << "Execution timed out after " << timeout_ms << " ms\n";
        return -2;  // Timeout
    }
}

// ---------- backend plugin loader ----------
struct PluginHandle {
    void* dl = nullptr;
    FuzzBackend* inst = nullptr;
    void (*destroy_fn)(FuzzBackend*) = nullptr;
};

PluginHandle load_plugin(const string &so_path) {
    PluginHandle ph{};
    ph.dl = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!ph.dl) {
        throw runtime_error(string("dlopen failed: ") + dlerror());
    }

    using create_fn_t = FuzzBackend* (*)();
    using destroy_fn_t = void (*)(FuzzBackend*);

    auto create_fn = (create_fn_t)dlsym(ph.dl, "create_backend");
    auto destroy_fn = (destroy_fn_t)dlsym(ph.dl, "destroy_backend");

    if (!create_fn) {
        dlclose(ph.dl);
        throw runtime_error("create_backend symbol not found in " + so_path);
    }
    if (!destroy_fn) {
        dlclose(ph.dl);
        throw runtime_error("destroy_backend symbol not found in " + so_path);
    }

    ph.inst = create_fn();
    ph.destroy_fn = destroy_fn;
    return ph;
}

void unload_plugin(PluginHandle &ph) {
    if (!ph.dl) return;
    if (ph.destroy_fn && ph.inst) ph.destroy_fn(ph.inst);
    dlclose(ph.dl);
    ph = {};
}

// ---------- helper: archive failure case (best-effort copy) ----------
void archive_failure_case(const fs::path &case_dir, const fs::path &fail_dir, const string &reason) {
    try {
        fs::create_directories(fail_dir);
        fs::path dest = fail_dir / case_dir.filename();
        // if exists, append timestamp
        if (fs::exists(dest)) {
            dest = fail_dir / (case_dir.filename().string() + "_" + timestamp_str());
        }
        fs::create_directories(dest);
        // recursive copy (best-effort)
        for (auto &entry : fs::recursive_directory_iterator(case_dir)) {
            auto rel = fs::relative(entry.path(), case_dir);
            auto target = dest / rel;
            if (entry.is_directory()) {
                fs::create_directories(target);
            } else if (entry.is_regular_file()) {
                fs::create_directories(target.parent_path());
                fs::copy_file(entry.path(), target, fs::copy_options::overwrite_existing);
            }
        }
        // write reason log
        std::ofstream(dest / "failure.log") << reason << "\n";
    } catch (const std::exception &e) {
        std::cerr << "archive_failure_case() failed: " << e.what() << "\n";
    }
}

// ---------- Program entry ----------
int main(int argc, char* argv[]) {
    // CLI: minimal arg parsing for backend selection
    string backend_so;
    string ref_backend_so; // optional
    uint64_t executor_timeout_ms = 30'000;
    // read CLI args simply
    for (int i = 1; i < argc; ++i) {
        string s = argv[i];
        if ((s == "--backend" || s == "-b") && i + 1 < argc) {
            backend_so = argv[++i];
        } else if ((s == "--ref-backend") && i + 1 < argc) {
            ref_backend_so = argv[++i];
        } else if ((s == "--timeout") && i + 1 < argc) {
            executor_timeout_ms = stoull(argv[++i]);
        } else {
            cerr << "Unknown arg: " << s << "\n";
        }
    }

    // allow env fallback
    if (backend_so.empty()) {
        if (const char* env = getenv("BACKEND_LIB")) backend_so = env;
    }
    if (ref_backend_so.empty()) {
        if (const char* env = getenv("REF_BACKEND_LIB")) ref_backend_so = env;
    }

    if (backend_so.empty()) {
        cerr << "No backend specified. Use --backend /path/to/libbackend.so or set BACKEND_LIB env var\n";
        return 1;
    }

    // signal handling
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Configurable parameters
    uint64_t seed = 42;
    size_t max_iterations = 1000000;
    fs::path out_root = "fuzz_output";
    fs::path fail_dir = out_root / "failures";
    fs::path corpus_dir = out_root / "corpus";

    if (const char* env = getenv("FUZZ_SEED")) seed = std::stoull(env);
    if (const char* env2 = getenv("FUZZ_ITERS")) max_iterations = std::stoull(env2);

    mt19937 rng(seed);
    cout << "Starting fuzz loop with seed=" << seed << " up to " << max_iterations << " iterations\n";

    // Create dirs
    fs::create_directories(out_root);
    fs::create_directories(corpus_dir);
    fs::create_directories(fail_dir);
    Logger::instance().setLogFile("fuzzer.log");

    // Load backend plugin (target)
    PluginHandle target_ph;
    try {
        target_ph = load_plugin(backend_so);
        cout << "Loaded backend: " << backend_so << "\n";
    } catch (const std::exception &e) {
        cerr << "Failed to load backend " << backend_so << ": " << e.what() << "\n";
        return 1;
    }
    FuzzBackend* target_backend = target_ph.inst;

    // Load reference backend if provided, otherwise reference uses same backend instance
    PluginHandle ref_ph;
    FuzzBackend* ref_backend = nullptr;
    bool ref_is_separate = false;
    if (!ref_backend_so.empty()) {
        try {
            ref_ph = load_plugin(ref_backend_so);
            ref_backend = ref_ph.inst;
            ref_is_separate = true;
            cout << "Loaded ref backend: " << ref_backend_so << "\n";
        } catch (const std::exception &e) {
            cerr << "Failed to load ref backend " << ref_backend_so << ": " << e.what() << "\n";
            unload_plugin(target_ph);
            return 1;
        }
    } else {
        ref_backend = target_backend; // use same implementation as trusted one if none provided
        cout << "No separate ref backend provided — using target backend as reference.\n";
    }

    // Main fuzz loop
    for (size_t iter = 0; iter < max_iterations && !g_terminate; ++iter) {
        try {
            string iter_id = "iter_" + to_string(iter) + "_" + timestamp_str();
            fs::path iter_dir = corpus_dir / iter_id;
            fs::path iter_data_dir = iter_dir / "data";
            fs::create_directories(iter_dir);
            fs::create_directories(iter_data_dir);

            // 1) Generate random kernel specification (einsum equations + tensor meta)
            auto [tensors, einsum] = generate_random_einsum(2, 6);

            // 2) Generate and store data for tensors
            vector<string> datafile_names = generate_random_tensor_data(tensors, iter_data_dir, "");
            if (!generate_ref_kernel(tensors, {einsum}, datafile_names, (iter_dir / "kernel.json").string()))
            {
                continue;
            }
            
            vector<string> mutated_file_names = mutate_equivalent_kernel(iter_dir, "kernel.json", MutationOperator::SPARSITY, 100);

            // 3) Generate backend-specific kernel
            fs::path backend_kernel = iter_dir / "backend_kernel"; // plugin decides extension/format
            fs::create_directories(backend_kernel);
            bool gen_ok = target_backend->generate_kernel(mutated_file_names, backend_kernel);
            if (!gen_ok) {
                cerr << "generate_kernel failed for iter " << iter << "\n";
                continue;
            }

            // 4) Run reference executor (trusted) once to produce expected outputs
            fs::path ref_out_dir = iter_data_dir / "ref_out";
            fs::create_directories(ref_out_dir);
            string ref_kernel_filename = (backend_kernel / "kernel/backend_kernel.cpp");
            int result = run_with_timeout(target_backend, ref_kernel_filename, "", 4000);
            // break;
            if (result!=0)
            {
                // most likely a crashing code bug
                // report the crashing code bug
                archive_failure_case(iter_data_dir, fail_dir, "Reference run failed or timed out (code " + to_string(result) + ")");
            } else
            {
                // reference execution successfully executed
                // run the mutations
                
            }

            // if (ref_ret != 0) {
            //     string reason = "Reference run failed or timed out (code " + to_string(ref_ret) + ")";
            //     cerr << reason << " for " << iter_id << "\n";
            //     archive_failure_case(iter_data_dir, fail_dir, reason);
            //     continue;
            // }

            // // 5) OPTIONAL: Mutate canonical kernel into equivalent variants (if you have a mutator)
            // // If you have mutate_equivalent_kernel implemented in your codebase, you can call it here.
            // // Example (uncomment and adapt signature if you have it):
            // // vector<string> mutated_files = mutate_equivalent_kernel(backend_kernel.string(), MutationOperator::SPARSITY, 100);
            // // For now we will just validate the canonical kernel as a single "mutant".
            // vector<string> mutated_files = { backend_kernel.string() };

            // // 6) Run target on each mutant and compare outputs
            // for (size_t mi = 0; mi < mutated_files.size() && !g_terminate; ++mi) {
            //     fs::path mutant_path = mutated_files[mi];
            //     fs::path mutant_out_dir = iter_data_dir / ("mutant_" + to_string(mi) + "_out");
            //     fs::create_directories(mutant_out_dir);

            //     int tgt_ret = run_with_timeout([&]() {
            //         try {
            //             bool ok = target_backend->execute_kernel(mutant_path.string(), mutant_out_dir.string());
            //             return ok ? 0 : 1;
            //         } catch (...) { return -1; }
            //     }, executor_timeout_ms);

            //     if (tgt_ret != 0) {
            //         string reason = "Target executor error/timeout (code " + to_string(tgt_ret) + ")";
            //         cerr << reason << " for mutant: " << mutant_path << "\n";
            //         archive_failure_case(iter_data_dir, fail_dir, reason);
            //         continue;
            //     }

            //     // Compare using the target's comparator (or ref comparator)
            //     // compare_results returns bool: true == equal
            //     bool equal = false;
            //     string diff_message;
            //     try {
            //         equal = target_backend->compare_results(ref_out_dir.string(), mutant_out_dir.string());
            //     } catch (const std::exception &e) {
            //         diff_message = string("compare_results threw: ") + e.what();
            //     }

            //     if (!equal) {
            //         string reason = "Mismatch for mutant: " + mutant_path.string() + ". " + diff_message;
            //         cerr << reason << "\n";
            //         archive_failure_case(iter_data_dir, fail_dir, reason);
            //         // optionally break after first mismatch
            //         // break;
            //     }
            // }

            // // 7) Save passing case to corpus for future shrinking/replay (copy iter_data_dir -> corpus)
            // for (auto &entry : fs::directory_iterator(iter_data_dir)) {
            //     fs::path dest = iter_dir / entry.path().filename();
            //     if (fs::is_directory(entry.path()))
            //         fs::copy(entry.path(), dest, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
            //     else
            //         fs::copy_file(entry.path(), dest, fs::copy_options::overwrite_existing);
            // }

            // if (iter % 100 == 0) {
            //     cout << "Iteration " << iter << " OK. Saved to " << iter_dir << endl;
            // }
        } catch (const std::exception &e) {
            cerr << "Exception in iteration " << iter << ": " << e.what() << std::endl;
            try {
                string iter_id = "iter_except_" + to_string(iter) + "_" + timestamp_str();
                fs::path except_dir = fail_dir / iter_id;
                fs::create_directories(except_dir);
            } catch (...) {}
        }
        break;
    }

    cout << "Fuzzing loop finished (terminated=" << g_terminate << ")\n";

    // unload plugins
    unload_plugin(target_ph);
    if (ref_is_separate) unload_plugin(ref_ph);

    return 0;
}
