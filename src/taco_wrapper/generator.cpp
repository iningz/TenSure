#include "taco_wrapper/generator.hpp"

namespace taco_wrapper {

TacoTensor toTacoTensor(const tsTensor& t, const string& dataFilename)
{
    TacoTensor tacoT;
    tacoT.name = std::string(1, t.name);
    tacoT.shape = t.shape;
    tacoT.idxs = t.idxs;
    tacoT.fmt = t.storageFormat;
    tacoT.dataFilename = dataFilename;

    return tacoT;
}

bool generate_taco_kernel(const tsKernel& kernel, const fs::path& out_file, std::vector<fs::path> results_file) {
    try {
        // generate TACO program string
        fs::create_directories(out_file);
        string program_code = generate_program(kernel, results_file);

        // atomic write
        string tmp_name = out_file / ((out_file.parent_path().stem().string()) + ".tmp");
        ofstream ofs(tmp_name);
        ofs << program_code;
        ofs.close();
        fs::rename(tmp_name, (out_file / ((out_file.parent_path().stem().string()) + ".cpp")).string()); // atomic replacement
    } catch (const exception& e) {
        cerr << "TacoBackend::generate_kernel failed: " << e.what() << endl;
        return false;
    }

    return true;
}

string generate_program(const tsKernel &kernel_info, std::vector<fs::path> results_file)
{
    int tab_space_count = 4;
    std::string space = "";
    for (size_t i = 0; i < tab_space_count; i++)
    {
            space += " ";
    }
    ostringstream oss;
    oss << "#include <iostream>\n#include <fstream>\n#include <sstream>\n#include <vector>\n#include <string>\n#include <stdexcept>\n#include \"taco.h\"\n\nusing namespace taco;\n\nint read_taco_file(std::string file_name, Tensor<double>& T)\n{\n\tstd::ifstream file(file_name);\n\tif (!file.is_open()) {\n\t\tthrow std::runtime_error(\"Failed to open file: \" + file_name);\n\t}\n\n\t std::string line;\n\twhile (std::getline(file, line)) {\n\t\tif (line.empty() || line[0] == '#') continue;\n\n\t\tstd::istringstream iss(line);\n\t\tstd::vector<double> tokens;\n\t\tdouble tmp;\n\n\twhile (iss >> tmp) {\n\t\t\ttokens.push_back(tmp);\n\t\t}\n\n\t\tif (tokens.size() < 2) {\n\t\t\tthrow std::runtime_error(\"Malformed line: \" + line);\n\t\t}\n\n\t\tstd::vector<int> coord;\n\t\tcoord.reserve(tokens.size() - 1);\n\n\t\tfor (size_t i =0; i < tokens.size() -  1; i++) {\n\t\t\tcoord.push_back(static_cast<int>(tokens[i]));\n\t\t}\n\t\tT.insert(coord, tokens.back());\n\t}\n\treturn 0;\n}\n\nint main() {\n";
    
    set<char> indexVar;
    vector<string> tensor_init = {};
    for(size_t i = 0; i < kernel_info.tensors.size(); i++)
    {
        const tsTensor &tensor = kernel_info.tensors[i];
        string tensorDataFilename = kernel_info.dataFileNames.at(std::string(1, tensor.name));
        for (auto &id : tensor.idxs)
            indexVar.insert(id);
        TacoTensor tacoTensor = toTacoTensor(tensor, tensorDataFilename);
        tensor_init.push_back(tacoTensor.initilization_string(space));
        // std::cout << tacoTensor.initilization_string(4) << std::endl;
    }
    oss << space << "IndexVar " << join(indexVar) << ";\n\n";

    for (auto &tensor_vals : tensor_init)
    {
        oss << tensor_vals;
    }

    for (auto &expression : kernel_info.computations)
    {
        oss << space << expression.expressions << ";\n\n";
    }

    oss << space << kernel_info.tensors[0].name << ".compile();\n";
    oss << space << kernel_info.tensors[0].name << ".assemble();\n";
    oss << space << kernel_info.tensors[0].name << ".compute();\n\n";

    for (auto &results_file_path : results_file) {
        fs::path abs_results_file_path = std::filesystem::absolute(std::filesystem::current_path() / results_file_path);
        oss << space << "write(\"" << abs_results_file_path.string() << "\", " << kernel_info.tensors[0].name << ");\n";
    }
    oss << "\n" << space << "return 0;\n";

    oss << "}";
    return oss.str();
}

}