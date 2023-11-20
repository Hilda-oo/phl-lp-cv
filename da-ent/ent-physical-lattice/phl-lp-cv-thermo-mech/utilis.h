#pragma once

#include <Eigen/Eigen>

#include "sha-base-framework/frame.h"

namespace da {
Eigen::Vector3d ComputeTriangularCircumcenter(const Eigen::Vector3d &vertex_a,
                                              const Eigen::Vector3d &vertex_b,
                                              const Eigen::Vector3d &vertex_c);

double Heaviside(double value, double epsilon);

auto ReadPolyhedronEdgesFromPath(const fs_path &path) -> std::vector<Eigen::MatrixXd>;
}  // namespace da
