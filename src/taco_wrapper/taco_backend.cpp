#include "taco_wrapper/taco_backend.hpp"
#include "taco_wrapper/generator.hpp"
#include "taco_wrapper/executor.hpp"
#include "taco_wrapper/comparator.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>


bool TacoBackend::generate_kernel(const vector<string>& mutated_kernel_file_names, const fs::path& output_dir) {
    // Call your existing executor.cpp function
    for (int i = 0; i < mutated_kernel_file_names.size(); i++) {
        auto &mutated_file_name = mutated_kernel_file_names[i];
        fs::path p(mutated_file_name);
        fs::path taco_kernel_file = output_dir / (p.stem());
        fs::create_directories(taco_kernel_file);
        tsKernel tskernel;
        tskernel.loadJson(mutated_file_name);
        if (i==0)
        {
            taco_wrapper::generate_taco_kernel(tskernel, taco_kernel_file, {(taco_kernel_file / "results.tns"), (p.parent_path() / "data" / "ref_out" / "results.tns")});
        } else {

            taco_wrapper::generate_taco_kernel(tskernel, taco_kernel_file, {(taco_kernel_file / "results.tns")});
        }
        fs::remove(p);
    }
    
    return true;
}

int TacoBackend::execute_kernel(const fs::path& kernelPath, const fs::path& outputDir) {
    // Call your existing executor.cpp function
    std::filesystem::path taco_path = std::filesystem::absolute(std::filesystem::current_path() / "../external/taco");
    std::filesystem::path abs_srcPath = std::filesystem::absolute(std::filesystem::current_path() / kernelPath);
    std::filesystem::path abs_outPath = std::filesystem::absolute(std::filesystem::current_path() / kernelPath.parent_path());

    std::filesystem::path exe_path = abs_outPath / abs_srcPath.stem();
    exe_path.replace_extension(".out");
    
    int ret = taco_wrapper::run_kernel(abs_srcPath.string(), exe_path.string(), taco_path.string());

    return ret;
}

bool TacoBackend::compare_results(const string& refDir, const string& testDir) {
    // Call your existing comparator.cpp function
    return taco_wrapper::compare_outputs(refDir, testDir);
}

// Plugin entry points
extern "C" FuzzBackend* create_backend() {
    return new TacoBackend();
}

extern "C" void destroy_backend(FuzzBackend* backend) {
    delete backend;
}
