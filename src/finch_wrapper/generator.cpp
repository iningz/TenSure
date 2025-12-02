#include "finch_wrapper/generator.hpp"
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace finch_wrapper {

namespace fs = std::filesystem;

bool generate_finch_kernel(const std::string &input_json_path,
                           const fs::path &out_dir,
                           const std::vector<fs::path> &results_file) {
  if (results_file.empty()) {
    std::cerr << "Error: No result file specified for Finch kernel."
              << std::endl;
    return false;
  }

  // Locate the python script
  // Assuming running from build directory, and script is in src/finch_wrapper
  fs::path script_path = fs::absolute(fs::current_path() /
                                      "../src/finch_wrapper/convert_kernel.py");

  if (!fs::exists(script_path)) {
    std::cerr << "Error: Python conversion script not found at " << script_path
              << std::endl;
    return false;
  }

  fs::path abs_input_path = fs::absolute(input_json_path);
  fs::path abs_out_dir = fs::absolute(out_dir);
  fs::path abs_result_file = fs::absolute(results_file[0]);

  // Construct command
  // python3 <script> <input> <out_dir> <result_file>
  std::string command = "python3 " + script_path.string() + " " +
                        abs_input_path.string() + " " + abs_out_dir.string() +
                        " " + abs_result_file.string();

  // std::cout << "Generating Finch kernel with command: " << command <<
  // std::endl;

  int ret = std::system(command.c_str());

  if (ret != 0) {
    std::cerr << "Error: Failed to generate Finch kernel (exit code " << ret
              << ")" << std::endl;
    return false;
  }

  return true;
}

} // namespace finch_wrapper
