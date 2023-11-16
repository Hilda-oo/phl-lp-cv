#pragma once

#include <Eigen/Eigen>

#include "sha-base-framework/frame.h"

namespace da {
auto ReadPolyhedronEdgesFromPath(const fs_path &path) -> std::vector<Eigen::MatrixXd>;
}  // namespace da
