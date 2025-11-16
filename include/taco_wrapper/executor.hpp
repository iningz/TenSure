#pragma once

#include <string>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <thread>

namespace taco_wrapper
{
using namespace std;

int run_kernel(const string& kernelPath, const string& exe_file_name, const string& tool_path);
}
