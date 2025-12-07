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
#include "tensure/ThreadPool.hpp"

namespace fs = std::filesystem;
using namespace std::chrono_literals;
using namespace std;

static inline bool g_terminate = false;
void signal_handler(int signum) {
    std::cerr << "Signal " << signum << " received. Will terminate after current iteration.\n";
    g_terminate = true;
}

// Global counter to track fuzzing progress across all threads
std::atomic<size_t> g_completed_runs = 0;
std::atomic<size_t> g_ref_crash_count = 0;
std::atomic<size_t> g_crash_bug_count = 0;
std::atomic<size_t> g_wrong_code_count = 0;

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

/**
 * @brief The core fuzzing task executed by a single worker thread.
 */
void FuzzingJob(size_t iter, FuzzBackend* target_backend, std::mt19937::result_type seed_offset, fs::path& out_root, const std::string& tensor_file_format, const uint64_t executor_timeout_ms
    ) {
    // Create a thread-local RNG based on the global seed offset
    std::mt19937 local_rng(seed_offset + iter);
    
    // The distributed is kept local to ensure each thread uses fresh randomness
    std::uniform_int_distribution<int> dist_tensor_count(2, 5);

    try {
        if (g_terminate) return;

        std::string iter_id = "iter_" + std::to_string(iter) + "_" + timestamp_str();
        LOG_INFO("Starting Fuzzing Job: " + iter_id);
        
        // Define paths
        fs::path iter_dir = out_root / "corpus" / iter_id;
        fs::path fail_dir = out_root / "failures";
        fs::path iter_data_dir = iter_dir / "data";
        fs::create_directories(iter_dir);
        fs::create_directories(iter_data_dir);

        // --- RAII GUARD: GUARANTEES CLEANUP & INCREMENT G_COUNTER ON RETURN ---
        struct JobFinalizer {
            fs::path& _iter_dir;
            fs::path& _fail_dir;
            std::string& _iter_id;

            ~JobFinalizer() {
                g_completed_runs++;

                bool is_iter_archived = fs::exists(_fail_dir / "ref_crash" / _iter_id) ||
                                        fs::exists(_fail_dir / "crash" / _iter_id) ||
                                        fs::exists(_fail_dir / "wc" / _iter_id);
                
                if (!is_iter_archived && fs::exists(_iter_dir)) {
                    try {
                        fs::remove_all(_iter_dir);
                    } catch (...) {}
                }
            }
        } finalizer{iter_dir, fail_dir, iter_id};

        // Generate random kernel specification
        auto [tensors, einsum] = generate_random_einsum(dist_tensor_count(local_rng), 6);
        
        LOG_INFO("Generated Random Einsum: " + einsum);
        
        // Generate and store data for tensors
        std::vector<std::string> datafile_names = generate_random_tensor_data(tensors, iter_data_dir, "", tensor_file_format);

        if (datafile_names.size() != tensors.size() - 1) { 
            LOG_ERROR("Tensor data generation failed for job: " + iter_id);
            return;
        }

        // Generate Reference Kernel (using the ref_backend)
        if (!generate_ref_kernel(tensors, {einsum}, datafile_names, (iter_dir / "kernel.json").string())) {
            LOG_WARN("Reference Backend Kernel Generation Failed.");
            return;
        }

        // Generate Mutants
        // We reuse the existing logic which mutates the kernel.json file directly
        vector<string> mutated_file_names = mutate_equivalent_kernel(iter_dir, "kernel.json", 10);
        LOG_INFO("Generated " + to_string(mutated_file_names.size() - 1) + " Equivalent Mutants.");

        // Generate the backend specific kernel
        fs::path backend_kernel = iter_dir / "backend_kernel";
        fs::create_directories(backend_kernel);
        bool gen_ok = target_backend->generate_kernel(mutated_file_names, backend_kernel);
        if (!gen_ok) {
            cerr << "generate_kernel failed for iter " << iter_id << "\n";
            LOG_WARN("generate_kernel failed for iter " + iter_id + " to generate mutated backend kernels.");
            return;
        }

        // Run reference executor (trusted) once to produce expected outputs
        uint64_t timeout = executor_timeout_ms; // Use CLI-defined timeout
        fs::path ref_out_dir = iter_data_dir / "ref_out";
        fs::create_directories(ref_out_dir);
        
        // Use the generated reference kernel path
        // TODO: Make it generic
        string ref_kernel_filename = (backend_kernel / "kernel/backend_kernel.cpp");

        int ref_result = run_with_timeout(target_backend, ref_kernel_filename, "", timeout);

        if (ref_result != 0) {
            g_ref_crash_count++;
            std::string message;
            if (ref_result == -2) message = "Reference Kernel execution timed out";
            else message = "Reference Kernel execution failed with code " + to_string(ref_result);
            
            LOG_INFO("Reference Kernel crash/timeout: " + iter_id);
            // archive_failure_case(iter_dir.stem().string(), iter_dir / "kernel", fail_dir / "ref_crash", message);
            return; 
        }

        // Run target on each mutant and compare outputs
        LOG_INFO("Running mutants...");
        
        // Path to the reference result output file
        string ref_out_file = (ref_out_dir / "results.tns").string();
        
        for (size_t mi = 1; mi < mutated_file_names.size() && !g_terminate; ++mi) {
            fs::path mutant_path = backend_kernel / ("kernel" + to_string(mi)) / "backend_kernel.cpp";
            
            // Run target backend on the mutated kernel
            int result = run_with_timeout(target_backend, mutant_path.string(), "", timeout);
            
            if (result != 0) {
                // Crashing bug or timeout
                if (result == -2) {
                    // Timeout: Increase timeout and retry this mutant
                    timeout += 4000;
                    mi--; // Decrement to retry the current mutant
                    continue;
                }
                // Actual Crashing Bug
                g_crash_bug_count++;
                LOG_INFO("CRASHING BUG FOUND IN MUTANT " + to_string(mi) + " of " + iter_id);
                // archive_failure_case(iter_id, mutant_path.parent_path(), fail_dir / "crash", "Mutated Kernel execution failed with code " + to_string(result));
                break; // don't break, if you want to check whether other mutants also induce bugs
            } 
            
            // Compare the results for a wrong code bug
            string ref_out_file = (iter_data_dir / "ref_out" / "results.tns").string();
            string mutant_out_file = mutant_path.parent_path() / "results.tns";
            bool equal = target_backend->compare_results(ref_out_file, mutant_out_file);
            
            if (!equal) {
                LOG_INFO("WRONG CODE BUG FOUND IN MUTANT " + to_string(mi) + " of " + iter_id);
                g_wrong_code_count++;
                // archive_failure_case(iter_id, mutant_path.parent_path(), fail_dir / "wc", "Mutated Kernel produced incorrect results.");
                break; // don't break, if you want to check whether other mutants also induce bugs
            }
        }
        
        // Logging for progress
        if (iter % 100 == 0) {
            LOG_INFO("Completed iteration " + to_string(iter));
            std::cout << "Iteration " << iter << " OK. Runs/sec: " << (g_completed_runs.load() / std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()) << endl;
        }

    } catch (const std::exception &e) {
        // Exception in the fuzzing pipeline (e.g., file system error, generator failure)
        cerr << "Exception in iteration " << iter << ": " << e.what() << std::endl;
        LOG_ERROR("Pipeline exception in iter " + std::to_string(iter) + ": " + e.what());
    }
}

// ---------- Program entry ----------
int main(int argc, char* argv[]) {
    // CLI: minimal arg parsing for backend selection
    string backend_so;
    uint64_t executor_timeout_ms = 30'000;
    string tensor_file_format = "tns";
    // read CLI args simply
    for (int i = 1; i < argc; ++i) {
        string s = argv[i];
        if ((s == "--backend" || s == "-b") && i + 1 < argc) {
            backend_so = argv[++i];
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

    if (backend_so.empty()) {
        cerr << "No backend specified. Use --backend /path/to/libbackend.so or set BACKEND_LIB env var\n";
        return 1;
    }

    // signal handling
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Configurable parameters
    uint64_t seed = 42;
    size_t max_iterations = 1000;
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
    std::cout << "Starting fuzz loop with seed=" << seed << " up to " << max_iterations << " iterations\n";
    LOG_INFO("Starting fuzz loop with seed = " + to_string(seed) + " up to " + to_string(max_iterations) + " iterations");

    // Load backend plugin (target)
    PluginHandle target_ph;
    try {
        target_ph = load_plugin(backend_so);
        std::cout << "Loaded backend: " << backend_so << "\n";
        LOG_INFO("Loaded backend: " + backend_so);
    } catch (const std::exception &e) {
        cerr << "Failed to load backend " << backend_so << ": " << e.what() << "\n";
        LOG_ERROR("Failed to load backend: " + backend_so + ": " + e.what());
        return 1;
    }
    FuzzBackend* target_backend = target_ph.inst;

    const size_t num_threads = std::thread::hardware_concurrency();
    size_t actual_threads = (num_threads == 0) ? 4 : num_threads;
    std::cout << "Starting Thread Pool with " << actual_threads << " workers.\n";

    ThreadPool pool(actual_threads);

    // The Producer Loop: Queues tasks up to max_iterations
    for (size_t iter = 0; iter < max_iterations && !g_terminate; ++iter) {
        
        // Enqueue the fuzzing job (wrapped in a lambda)
        // We capture shared read-only pointers and config by value/reference.
        // We pass the RNG seed offset (iter) instead of the RNG object itself.
        pool.enqueue([=, &out_root]() mutable {
            FuzzingJob(iter, target_backend, rng(), out_root, tensor_file_format, executor_timeout_ms);
        });

        // Throttle the producer if too far ahead (optional, but prevents massive queueing if workers are slow)
        // Check if the number of tasks in the queue exceeds a safe threshold (e.g., 2x threads)
        while ((iter - g_completed_runs.load()) > actual_threads * 2 && !g_terminate) {
             std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    std::cout << "All fuzzing jobs successfully queued.\n";

    // Monitoring Loop (Kept as is)
    size_t last_count = 0;
    while (g_completed_runs < max_iterations && !g_terminate) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        size_t current_count = g_completed_runs.load();
        size_t rate = (current_count - last_count) / 2;
        std::cout << "Progress: " << current_count << " / " << max_iterations 
                  << " | Rate: " << rate << " runs/sec\n";
        last_count = current_count;
    }

    std::cout << "Fuzzing loop finished (terminated=" << g_terminate << ")\n";
    LOG_INFO("Fuzzing loop finished (terminated=" + to_string(g_terminate));

    // unload plugins
    unload_plugin(target_ph);

    return 0;
}
