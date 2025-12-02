#include "finch_wrapper/executor.hpp"
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace finch_wrapper {

namespace fs = std::filesystem;

int execute_finch_kernel(const fs::path &kernel_dir) {
  fs::path json_path = kernel_dir / "kernel.json";

  // Robustly locate Project.toml to define the project root.
  // We search in the current directory and up a few levels.
  fs::path cwd = fs::current_path();
  fs::path project_root;
  bool found_project = false;

  fs::path search_dir = cwd;
  // Limit search depth to avoid traversing too far up the filesystem
  for (int i = 0; i < 4; ++i) {
    if (fs::exists(search_dir / "Project.toml")) {
      // Use canonical path to resolve ".." and symlinks
      project_root = fs::canonical(search_dir);
      found_project = true;
      break;
    }

    if (search_dir.has_parent_path() &&
        search_dir != search_dir.parent_path()) {
      search_dir = search_dir.parent_path();
    } else {
      break;
    }
  }

  if (!found_project) {
    std::cerr << "Error: Could not locate Project.toml starting from " << cwd
              << ". Please ensure you are running from the build directory or "
                 "project root."
              << std::endl;
    return -1;
  }

  fs::path eval_script = project_root / "src/finch_wrapper/eval_finch.jl";

  if (!fs::exists(eval_script)) {
    std::cerr << "Error: eval_finch.jl not found at " << eval_script
              << std::endl;
    return -1;
  }

  if (!fs::exists(json_path)) {
    std::cerr << "Error: kernel.json not found at " << json_path << std::endl;
    return -1;
  }

  // Include the --project flag to use the Project.toml in the detected root
  // directory
  std::string command = "julia --project=" + project_root.string() + " " +
                        eval_script.string() + ' ' + json_path.string() + " --dump";

  // std::cout << "Executing Finch kernel: " << command << std::endl;
  int ret = std::system(command.c_str());

  if (ret != 0) {
    std::cerr << "Finch execution failed with code " << ret << std::endl;
  }

  return ret;
}

} // namespace finch_wrapper
