#pragma once

#include "tensure/formats.hpp"
#include "tensure/utils.hpp"
#include "taco.h"
#include <string>

namespace taco_wrapper {
using namespace std;

typedef struct TacoTensor {
    string name;
    vector<char> idxs;
    vector<int> shape;
    vector<TensorFormat> fmt;
    string dataFilename;

    TacoTensor() {}

    // generate initilization string
    string initilization_string(string tab_space)
    {   
        ostringstream oss;
        oss << tab_space << "Tensor<double> " << name << "(\"" << name << "\", {" << join(shape, ",") << "}, Format({";
        string dataFormat = "";
        for (int i = 0; i < fmt.size(); i++)
        {
            dataFormat += to_string(fmt[i]);
            if (i!=fmt.size()-1)
            {
                dataFormat += ",";
            }
        }
        oss << dataFormat << "}));\n";

        if (dataFilename != "-")
        {
            fs::path datafile(dataFilename);
            fs::path abs_datafile = std::filesystem::absolute(std::filesystem::current_path() / datafile);
            oss << tab_space << "read_taco_file(\"" << abs_datafile.string() << "\", " << name << ");";
            oss << tab_space << name << ".pack();\n\n";
        }
        
        return oss.str();
    }
} TacoTensor;

// bool generate_taco_kernel(const tsKernel& kernel, const fs::path& outFile);
bool generate_taco_kernel(const tsKernel& kernel, const fs::path& out_file, std::vector<fs::path> results_file);
string generate_program(const tsKernel &kernel_info, std::vector<fs::path> results_file);
// string generate_program(const tsKernel &kernel_info, const string& results_file);

}