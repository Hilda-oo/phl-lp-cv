#pragma once

#include <Eigen/Eigen>

namespace da::sha {

// compute element stiffness matrix of 3D linear elastic material
// using 2-nd Gauss integration
// @input
//  a, b, c: half size of element
//  D      : constitutive matrix, (6, 6)
// @output
//  Ke     : element matrix, (24, 24)
void ComputeKe(double a, double b, double c, const Eigen::Matrix<double, 6, 6> &D,
               Eigen::Matrix<double, 24, 24> &Ke);

// compute element strain-displacement matrix B in 8 Gauss Points
// @input
//  a, b, c: half size of element
// @output
//  Be     : element strain-displacement matrix, (6, 24)
void ComputeBe(double a, double b, double c, std::vector<Eigen::Matrix<double, 6, 24>> &Be);

// compute element shape function matrix in a given local point P
// @input
//  P: given local point, in [-1, 1] * [-1, 1] * [-1, 1]
// @output
//  N: shape function matrix N, (3, 24)
void ComputeN(const Eigen::RowVector3d &P, Eigen::Matrix<double, 3, 24> &N);

// compute element strain-displacement matrix B in a given local point P
// @input
//  a, b, c: half size of element
//  P: given local point, in [-1, 1] * [-1, 1] * [-1, 1]
// @output
//  B: element strain-displacement matrix, (6, 24)
void ComputeB(double a, double b, double c, const Eigen::RowVector3d &P,
              Eigen::Matrix<double, 6, 24> &B);

void ComputeNForTet(const Eigen::RowVector3d &P, const Eigen::Matrix<double, 4, 3> &X,
                    Eigen::Matrix<double, 3, 12> &N);
void ComputeBForTet(const Eigen::Matrix<double, 4, 3> &X, Eigen::Matrix<double, 6, 12> &B);
void ComputeKeForTet(const Eigen::Matrix<double, 4, 3> &X, const Eigen::Matrix<double, 6, 6> &D,
                     Eigen::Matrix<double, 12, 12> &Ke, double &Vol);

void ComputeHeatKeForTet(const Eigen::Matrix<double, 4, 3> &X, Eigen::Matrix<double, 4, 4> &heatKe, double &Vol);         
void ComputeHeatBeForTet(const Eigen::Matrix<double, 4, 3> &X, Eigen::Matrix<double, 3, 4> &Be);            
}  // namespace da::sha
