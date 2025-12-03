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
            LOG_ERROR((std::ostringstream{} << "Exception from timed task: " << e.what()).str());
            return -1;  // Error during execution
        }
    } else {
        std::cerr << "Execution timed out after " << timeout_ms << " ms\n";
        std::cerr << "Execution timed out after " << timeout_ms << " ms\n";
        LOG_ERROR((std::ostringstream{} << "Execution timed out after " << timeout_ms).str());
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
static void copy_tree(const fs::path& src, const fs::path& dst) {
    for (auto& entry : fs::recursive_directory_iterator(src)) {
        auto rel = fs::relative(entry.path(), src);
        auto target = dst / rel;

        if (entry.is_directory()) {
            fs::create_directories(target);
        } else if (entry.is_regular_file()) {
            fs::create_directories(target.parent_path());
            fs::copy_file(entry.path(), target,
                          fs::copy_options::overwrite_existing);
        }
    }
}

static void append_log(const fs::path& file, const std::string& reason) {
    std::ofstream out(file, std::ios::app);
    out << reason << "\n";
}

void archive_failure_case(const fs::path &dir_name, const fs::path &kernel_dir, const fs::path &fail_dir, const string &reason) {
     try {
        fs::create_directories(fail_dir);
        fs::path case_failure_dir = fail_dir / dir_name;
        fs::create_directories(case_failure_dir);

        // 1. Copy the kernel_dir -> case_failure_dir/<kernel-name>
        copy_tree(kernel_dir, case_failure_dir / kernel_dir.stem());

        // 2. Copy ref kernel
        if (kernel_dir.stem().string() != "kernel") {
            fs::path ref_kernel = kernel_dir.parent_path() / "kernel";
            copy_tree(ref_kernel, case_failure_dir / ref_kernel.stem());
        }

        // 3. Copy shared data directory
        fs::path data_dir = kernel_dir.parent_path().parent_path() / "data";
        copy_tree(data_dir, case_failure_dir / "data");

        // write reason log
        append_log(case_failure_dir / "failure.log", reason);

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
    string tensor_file_format = "tns";
    // read CLI args simply
    for (int i = 1; i < argc; ++i) {
        string s = argv[i];
        if ((s == "--backend" || s == "-b") && i + 1 < argc) {
            backend_so = argv[++i];
        } else if ((s == "--ref-backend") && i + 1 < argc) {
            ref_backend_so = argv[++i];
        } else if ((s == "--timeout") && i + 1 < argc) {
            executor_timeout_ms = stoull(argv[++i]);
        } else if ((s == "--tensor-format" || s == "--tfmt") && i + 1 < argc) {
            string user_tfmt = argv[++i];
            std::transform(user_tfmt.begin(), user_tfmt.end(), user_tfmt.begin(), 
                            [](unsigned char c) { 
                                return std::tolower(c); 
                            });
            if (user_tfmt != "tns" && user_tfmt != "ttx")
            {
                cerr << "Unsupported tensor storage format: " << user_tfmt << "\n";
            } else {
                tensor_file_format = user_tfmt;
            }
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

    // Create dirs
    fs::create_directories(out_root);
    fs::create_directories(corpus_dir);
    fs::create_directories(fail_dir);

    // Logging
    Logger::instance().setLogFile("./fuzzer.log");
    LOG_INFO("Fuzzer starting...");
    Logger::instance().setConsoleOnly(false);
    cout << "Starting fuzz loop with seed=" << seed << " up to " << max_iterations << " iterations\n";
    LOG_INFO("Starting fuzz loop with seed = " + to_string(seed) + " up to " + to_string(max_iterations) + " iterations");

    // Load backend plugin (target)
    PluginHandle target_ph;
    try {
        target_ph = load_plugin(backend_so);
        cout << "Loaded backend: " << backend_so << "\n";
        LOG_INFO("Loaded backend: " + backend_so);
    } catch (const std::exception &e) {
        cerr << "Failed to load backend " << backend_so << ": " << e.what() << "\n";
        LOG_ERROR("Failed to load backend: " + backend_so + ": " + e.what());
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
            LOG_INFO("Loaded ref backend: " + ref_backend_so);
        } catch (const std::exception &e) {
            cerr << "Failed to load ref backend " << ref_backend_so << ": " << e.what() << "\n";
            LOG_ERROR("Failed to load ref backend: " + ref_backend_so + ": " + e.what());
            unload_plugin(target_ph);
            return 1;
        }
    } else {
        ref_backend = target_backend; // use same implementation as trusted one if none provided
        LOG_WARN("No separate ref backend provided — using target backend as reference.");
        cout << "No separate ref backend provided — using target backend as reference.\n";
    }

    // Main fuzz loop
    for (size_t iter = 0; iter < max_iterations && !g_terminate; ++iter) {
        try {
            string iter_id = "iter_" + to_string(iter) + "_" + timestamp_str();
            LOG_INFO("Starting Fuzzing Loop: " + iter_id);
            fs::path iter_dir = corpus_dir / iter_id;
            fs::path iter_data_dir = iter_dir / "data";
            fs::create_directories(iter_dir);
            fs::create_directories(iter_data_dir);

            // 1) Generate random kernel specification (einsum equations + tensor meta)
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dist_tensor_count(2, 5);
            auto [tensors, einsum] = generate_random_einsum(dist_tensor_count(gen), 6);

            LOG_INFO("Generated Random Einsum: " + einsum);
            // 2) Generate and store data for tensors
            vector<string> datafile_names = generate_random_tensor_data(tensors, iter_data_dir, "", tensor_file_format);
            // check whether we got data for all input tensors.
            if (datafile_names.size() != tensors.size() - 1)  { // -1 to restrict the input tensor
                LOG_ERROR("Tensor data generation failed for fuzzing iteration: " + iter_id);
                continue;
            }
            if (!generate_ref_kernel(tensors, {einsum}, datafile_names, (iter_dir / "kernel.json").string()))
            {
                LOG_WARN("Backend Kernel Generation Failed: " + backend_so);
                continue;
            }
            
            vector<string> mutated_file_names = mutate_equivalent_kernel(iter_dir, "kernel.json", 10);
            LOG_INFO("Generated " + to_string(mutated_file_names.size() - 1) + " Equivalent Mutants.");

            // 3) Generate backend-specific kernel
            fs::path backend_kernel = iter_dir / "backend_kernel"; // plugin decides extension/format
            fs::create_directories(backend_kernel);
            bool gen_ok = target_backend->generate_kernel(mutated_file_names, backend_kernel);
            if (!gen_ok) {
                cerr << "generate_kernel failed for iter " << iter_id << "\n";
                LOG_WARN("generate_kernel failed for iter " + iter_id + " to generate mutated backend kernels.");
                continue;
            }

            // 4) Run reference executor (trusted) once to produce expected outputs
            uint64_t timeout = 8000;
            fs::path ref_out_dir = iter_data_dir / "ref_out";
            fs::create_directories(ref_out_dir);
            string ref_kernel_filename = (backend_kernel / "kernel/backend_kernel.cpp");
            int result = run_with_timeout(target_backend, ref_kernel_filename, "", timeout);

            if (result!=0)
            {
                // most likely a crashing code bug
                // report the crashing code bug
                std::string message;
                if (result == -2)
                {
                    message = "Reference Kernel execution timed out (code " + to_string(result) + ")";
                    LOG_INFO("Reference Kernel execution timed out in " + iter_id);
                } else {
                    message = "Refernce Kernel execution failed with (code " + to_string(result) + ")";
                    LOG_INFO("Reference Kernel execution failed in " + iter_id);
                }

                archive_failure_case(iter_dir.stem().string(), (iter_dir)/"backend_kernel"/"kernel", fail_dir / "crash", message);
                continue;
            }


            // reference execution successfully executed
            // run the mutations
            // 5) Run target on each mutant and compare outputs
            LOG_INFO("Running mutants...");
            for (size_t mi = 1; mi < mutated_file_names.size() && !g_terminate; ++mi) {
                fs::path mutant_path = backend_kernel / ("kernel" + to_string(mi)) / "backend_kernel.cpp";
                int result = run_with_timeout(target_backend, mutant_path.string(), "", timeout);
                if (result !=0) {
                    // crashing bug or timeout
                    if (result == -2) {
                        // timeout
                        mi--;
                        timeout += 4000;
                        continue;
                    }
                    // crashing bug
                    LOG_INFO("ACTUAL CRASHING BUG FOUND IN " + iter_id);
                    // archive the bug and break this iteration.
                    archive_failure_case(iter_dir.stem().string(), mutant_path.parent_path(), fail_dir / "crash", "Mutated Kernel execution failed with (code " + to_string(result) + ")");
                    break; // don't break, if you want to check whether the remaining mutants also can find any other bugs.
                } else {
                    // compare the results for a wrong code bug
                    string ref_out_file = (iter_data_dir / "ref_out" / "results.tns").string();
                    string mutant_out_file = mutant_path.parent_path() / "results.tns";
                    bool equal = target_backend->compare_results(ref_out_file, mutant_out_file);
                    if (!equal) {
                        LOG_INFO("WRONG CODE BUG FOUND IN " + iter_id);
                        // archive the bug and break this iteration.
                        archive_failure_case(iter_dir.stem().string(), mutant_path.parent_path(), fail_dir / "wc", "Mutated Kernel produced incorrect results.");
                        break; // don't break, if you want to check whether the remaining mutants also can find any other bugs.
                    }
                }
            }

            // Not Completed
            // // 6) Save passing case to corpus for future shrinking/replay (copy iter_data_dir -> corpus)
            // for (auto &entry : fs::directory_iterator(iter_data_dir)) {
            //     fs::path dest = iter_dir / entry.path().filename();
            //     if (fs::is_directory(entry.path()))
            //         fs::copy(entry.path(), dest, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
            //     else
            //         fs::copy_file(entry.path(), dest, fs::copy_options::overwrite_existing);
            // }

            if (iter % 100 == 0) {
                LOG_INFO("Completed iteration " + to_string(iter));
                LOG_INFO("Iteration directory: " + iter_dir.string());
                cout << "Iteration " << iter << " OK. Saved to " << iter_dir << endl;
            }
        } catch (const std::exception &e) {
            cerr << "Exception in iteration " << iter << ": " << e.what() << std::endl;
            try {
                string iter_id = "iter_except_" + to_string(iter) + "_" + timestamp_str();
                fs::path except_dir = fail_dir / iter_id;
                fs::create_directories(except_dir);
            } catch (...) {}
        }
        // break;
    }

    cout << "Fuzzing loop finished (terminated=" << g_terminate << ")\n";

    // unload plugins
    unload_plugin(target_ph);
    if (ref_is_separate) unload_plugin(ref_ph);

    return 0;
}
