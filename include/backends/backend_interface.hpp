#pragma once
#include <string>
#include <vector>

using namespace std;

struct FuzzBackend {
    virtual ~FuzzBackend() = default;

    virtual bool generate_kernel(const vector<string>& tskernel, const string& outFile) = 0;

    virtual bool execute_kernel(const string& kernelPath, const string& outputDir) = 0;

    virtual bool compare_results(const string& refDir, const string& testDir) = 0;
};

// Utility to dynamically load/unload backend plugins
FuzzBackend* load_backend(const std::string& so_path);
void unload_backend(FuzzBackend* backend);
