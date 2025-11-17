#pragma once
#include <string>
#include <vector>
#include "tensure/formats.hpp"
#include <dlfcn.h>
#include <iostream>

using namespace std;
namespace fs = std::filesystem;

struct FuzzBackend {
    virtual ~FuzzBackend() = default;

    virtual bool generate_kernel(const vector<string>& mutated_kernel_file_names, const fs::path& output_dir) = 0;

    virtual int execute_kernel(const fs::path& kernelPath, const fs::path& outputDir) = 0;

    virtual bool compare_results(const string& refDir, const string& testDir) = 0;
};

// Utility to dynamically load/unload backend plugins
FuzzBackend* load_backend(const std::string& so_path);
void unload_backend(FuzzBackend* backend);
