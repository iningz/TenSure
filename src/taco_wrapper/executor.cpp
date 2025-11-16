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

// bool run_kernel(const string &kernelPath)
// {
//     namespace fs = std::filesystem;
//     fs::path src(kernelPath);

//     if (!fs::exists(src))
//     {
//         cerr << "Kernel file not found: " << src << "\n";
//         return false;
//     }

//     // 1. Build output binary path
//     fs::path buildDir = src.parent_path();
//     fs::path exePath = buildDir / "kernel_exec";

//     // 2. Construct compile command
//     std::string compileCmd = "g++ -std=c++17 -O2 -fPIC " + src.string() +
//                              " -I" + std::string("./include") +
//                              " -L" + std::string("./build/lib") +
//                              " -ltaco -o " + exePath.string();

//     std::cout << "[INFO] Compiling kernel: " << compileCmd << std::endl;

//     int ret = std::system(compileCmd.c_str());
//     if (ret != 0)
//     {
//         std::cerr << "Compilation failed for " << kernelPath << std::endl;
//         return false;
//     }

//     // 3. Run with timeout (example: 10 seconds)
//     std::cout << "[INFO] Running kernel: " << exePath << std::endl;
//     auto start = std::chrono::steady_clock::now();

//     int runRet = std::system(exePath.c_str());

//     auto end = std::chrono::steady_clock::now();
//     double elapsed = std::chrono::duration<double>(end - start).count();
//     std::cout << "[INFO] Kernel run took " << elapsed << "s\n";

//     if (runRet != 0)
//     {
//         std::cerr << "Kernel execution failed (" << runRet << ")\n";
//         return false;
//     }

//     return true;
// }

}