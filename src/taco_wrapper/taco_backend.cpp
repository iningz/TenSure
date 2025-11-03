#include "taco_wrapper/taco_backend.hpp"
#include "taco_wrapper/generator.hpp"
#include "taco_wrapper/executor.hpp"
#include "taco_wrapper/comparator.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

bool generate_kernel(const vector<tsTensor>& tensors, const vector<string>& computations, const vector<string>& dataFiles, const string& outFile) {
    if (tensors.size() != dataFiles.size()) return false;

    tsKernel kernel;
    for (size_t i = 0; i < tensors.size(); i++) {
        kernel.tensors.push_back(tensors[i]);
        kernel.dataFileNames.insert({std::string(1, tensors[i].name), dataFiles[i]});
    }

    for (auto& comp : computations) {
        tsComputation c;
        c.expressions = comp;
        kernel.computations.push_back(c);
    }

    try {
        // generate TACO program string
        string program_code = taco_wrapper::generate_program(kernel);

        // atomic write
        string tmp_name = outFile + ".tmp";
        ofstream ofs(tmp_name);
        ofs << program_code;
        ofs.close();
        fs::rename(tmp_name, outFile); // atomic replacement
    } catch (const exception& e) {
        cerr << "TacoBackend::generate_kernel failed: " << e.what() << endl;
        return false;
    }

    return true;
}

bool TacoBackend::generate_kernel(const vector<string>& tskernel, const string& outFile) {
    // Call your existing executor.cpp function
    return true;
}

bool TacoBackend::execute_kernel(const string& kernelPath, const string& outputDir) {
    // Call your existing executor.cpp function
    return taco_wrapper::run_kernel(kernelPath, outputDir);
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
