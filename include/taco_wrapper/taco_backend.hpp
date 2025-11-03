#pragma once
#include "backends/backend_interface.hpp"
#include "taco_wrapper/generator.hpp"
#include <string>
#include <vector>

using namespace std;

struct TacoBackend : public FuzzBackend {
    bool generate_kernel(const vector<string>& tskernel, const string& outFile) override;

    bool execute_kernel(const string& kernelPath,
                        const string& outputDir) override;

    bool compare_results(const string& refDir,
                         const string& testDir) override;
};

// Plugin entry points
extern "C" FuzzBackend* create_backend();
extern "C" void destroy_backend(FuzzBackend* backend);
