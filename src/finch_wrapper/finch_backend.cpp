#include "finch_wrapper/finch_backend.hpp"
#include "finch_wrapper/executor.hpp"
#include "finch_wrapper/generator.hpp"
#include "tensure/utils.hpp"
#include <filesystem>
#include <iostream>

using namespace finch_wrapper;
using namespace std;

bool FinchBackend::generate_kernel(
    const vector<string> &mutated_kernel_file_names,
    const fs::path &output_dir) {

  for (const auto &file_name : mutated_kernel_file_names) {
    fs::path p(file_name);
    // Create a specific directory for this kernel's execution artifacts
    fs::path kernel_dir = output_dir / p.stem();

    // Define the expected output result file path
    // This is where Finch should write the resulting tensor
    // Switching to .ttx (Tensor Market) as requested
    vector<fs::path> result_files;
    result_files.push_back(kernel_dir / "results.ttx");

    // Generate the Finch-specific JSON configuration
    // We pass the input file path directly now, as the generator calls a python
    // script
    if (!generate_finch_kernel(file_name, kernel_dir, result_files)) {
      cerr << "Failed to generate Finch kernel for " << file_name << endl;
      return false;
    }
  }
  return true;
}

int FinchBackend::execute_kernel(const fs::path &kernelPath,
                                 const fs::path &outputDir) {
  // kernelPath might be passed as a file path (e.g. backend_kernel.cpp) by the
  // core fuzzer, even if that file doesn't exist. We want the directory.
  fs::path target_dir = kernelPath;
  if (target_dir.has_extension()) {
    target_dir = target_dir.parent_path();
  }

  int ret = execute_finch_kernel(target_dir);

  if (ret == 0) {
    // If this was the reference kernel execution (indicated by directory name
    // "kernel"), we need to copy the output to the standard reference output
    // location so that TenSure core can find it for comparison.
    // Core expects reference output in: iter_dir/data/ref_out/
    // target_dir is: iter_dir/backend_kernel/kernel/
    if (target_dir.stem() == "kernel") {
      try {
        fs::path src_file = target_dir / "results.ttx";
        // Go up: kernel -> backend_kernel -> iter_dir -> data -> ref_out
        fs::path ref_out_dir =
            target_dir.parent_path().parent_path() / "data" / "ref_out";
        fs::path dst_file = ref_out_dir / "results.ttx";

        if (fs::exists(src_file)) {
          fs::create_directories(ref_out_dir);
          fs::copy_file(src_file, dst_file,
                        fs::copy_options::overwrite_existing);
        }
      } catch (const std::exception &e) {
        std::cerr << "Warning: Failed to copy reference output: " << e.what()
                  << std::endl;
      }
    }
  }

  return ret;
}

bool FinchBackend::compare_results(const string &ref, const string &test) {
  // The core fuzzer passes full file paths (usually defaulting to .tns).
  // Since we switched to .ttx, we need to handle the mismatch if the file
  // passed doesn't exist but the .ttx version does.

  fs::path pRef(ref);
  fs::path pTest(test);

  auto resolve_path = [](fs::path p) -> fs::path {
    // If the path exists, use it.
    if (fs::exists(p))
      return p;

    // If it's a .tns file that doesn't exist, try .ttx
    if (p.extension() == ".tns") {
      fs::path ttx_path = p;
      ttx_path.replace_extension(".ttx");
      if (fs::exists(ttx_path))
        return ttx_path;
    }
    return p;
  };

  pRef = resolve_path(pRef);
  pTest = resolve_path(pTest);

  // Use the global compare_outputs from tensure/utils.hpp with a tolerance
  return ::compare_outputs(pRef.string(), pTest.string(), 1e-5);
}

// Plugin entry points required by TenSure's backend loader
extern "C" FuzzBackend *create_backend() { return new FinchBackend(); }

extern "C" void destroy_backend(FuzzBackend *backend) { delete backend; }
