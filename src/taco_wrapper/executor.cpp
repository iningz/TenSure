#include "taco_wrapper/executor.hpp"

namespace taco_wrapper {

int run_kernel(const string& kernelPath, const string& exe_file_name, const string& tool_path)
{
    namespace fs = std::filesystem;

    if (!fs::exists(kernelPath))
    {
        cerr << "Kernel file not found: " << kernelPath << "\n";
        return false;
    }

    // 1. Build the executable kernel
    string compileCmd = "g++ " + kernelPath + " -std=c++17"
                            " -I" + (tool_path + "/include") +
                            " -L" + (tool_path + "/build/lib") +
                            " -ltaco" +
                            " -Wl,-rpath," + (tool_path + "/build/lib") +
                            " -o " + exe_file_name;
    
    std::cout << "[INFO] Compiling kernel: " << compileCmd << std::endl;

    int ret = std::system(compileCmd.c_str());
    if (ret != 0)
    {
        std::cerr << "Compilation failed for " << kernelPath << std::endl;
        return WEXITSTATUS(ret);
    }

    // 2. Run the executable
    ret = std::system(exe_file_name.c_str());
    if (ret == -1)
    {
        std::cerr << "Failed to start process (system() error)\n";
    } else if (WIFEXITED(ret) && WEXITSTATUS(ret) == 0) {
        std::cout << "Kernel Execution Succeeded!\n";
        return 0;
    } else {
        std::cerr << "Kernel Execution failed with code: " << WEXITSTATUS(ret) << "\n";
    }

    return WEXITSTATUS(ret);
}

}