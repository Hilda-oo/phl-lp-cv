#pragma once

#include <Eigen/Eigen>

namespace da::sha {
void ComputeBarycentric(const Eigen::MatrixXd &polygon, const Eigen::RowVector3d &p, int rowI,
                        Eigen::MatrixXd &weight);
}  // namespace da::sha