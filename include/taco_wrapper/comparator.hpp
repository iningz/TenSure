#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <cmath>

namespace taco_wrapper
{
using namespace std;

bool compare_outputs(const std::string& ref_output, const std::string& kernel_output, double total = 1e-8);
}