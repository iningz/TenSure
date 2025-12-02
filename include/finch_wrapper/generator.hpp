#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace finch_wrapper {

namespace fs = std::filesystem;

bool generate_finch_kernel(const std::string &input_json_path,
                           const fs::path &out_dir,
                           const std::vector<fs::path> &results_file);

} // namespace finch_wrapper
