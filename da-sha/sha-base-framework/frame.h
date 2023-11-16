#pragma once

#include <string>
#include "declarations.h"
#include "logger.h"
#include "eigen.h"

namespace da {
void Terminate(const std::string &message);
void Assert(bool condition, const std::string &message = "");
auto WorkplacePath() -> fs_path;
auto WorkingAssetDirectoryPath() -> fs_path;
auto WorkingResultDirectoryPath() -> fs_path;
auto ProjectSourcePath() -> fs_path;
auto ProjectAssetDirectoryPath() -> fs_path;
}  // namespace da
