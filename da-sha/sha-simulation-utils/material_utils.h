#pragma once

#include <Eigen/Eigen>

namespace da::sha {

void ComputeElasticMatrix(double YM, double PR, Eigen::Matrix<double, 6, 6> &D);

double ComputeVonStress(const Eigen::Vector<double, 6> &stress);

}  // namespace da::sha
