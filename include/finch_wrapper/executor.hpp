#pragma once

#include <filesystem>

namespace finch_wrapper {

namespace fs = std::filesystem;

int execute_finch_kernel(const fs::path &kernel_dir);

} // namespace finch_wrapper
