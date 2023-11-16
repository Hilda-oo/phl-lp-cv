#pragma once

#include <Eigen/Eigen>
#include <memory>
#include "cpt-linear-solver/linear_solver.h"

namespace da::sha {

template <int dim>
void AddBlockToMatrix(
    const Eigen::MatrixXd& block, const Eigen::VectorXi& index, int rowIndI,
    std::shared_ptr<cpt::LinearSolver<Eigen::VectorXi, Eigen::VectorXd, dim>> linSysSolver) {
  int rowStart = index[rowIndI] * dim;
  if (rowStart < 0) {  // DBC node
    rowStart = -rowStart - dim;
    linSysSolver->SetCoeff(rowStart, rowStart, 1.0);
    linSysSolver->SetCoeff(rowStart + 1, rowStart + 1, 1.0);
    if constexpr (dim == 3) {
      linSysSolver->SetCoeff(rowStart + 2, rowStart + 2, 1.0);
    }
    return;
  }

  int eleNodeNum = static_cast<int>(index.size());
  for (int _i = 0; _i < eleNodeNum; ++_i) {
    if (index[_i] >= 0) {
      int _Idim      = _i * dim;
      int _dimIndexI = index[_i] * dim;
      linSysSolver->AddCoeff(rowStart, _dimIndexI, block(0, _Idim));
      linSysSolver->AddCoeff(rowStart, _dimIndexI + 1, block(0, _Idim + 1));
      linSysSolver->AddCoeff(rowStart + 1, _dimIndexI, block(1, _Idim));
      linSysSolver->AddCoeff(rowStart + 1, _dimIndexI + 1, block(1, _Idim + 1));
      if constexpr (dim == 3) {
        linSysSolver->AddCoeff(rowStart, _dimIndexI + 2, block(0, _Idim + 2));
        linSysSolver->AddCoeff(rowStart + 1, _dimIndexI + 2, block(1, _Idim + 2));
        linSysSolver->AddCoeff(rowStart + 2, _dimIndexI, block(2, _Idim));
        linSysSolver->AddCoeff(rowStart + 2, _dimIndexI + 1, block(2, _Idim + 1));
        linSysSolver->AddCoeff(rowStart + 2, _dimIndexI + 2, block(2, _Idim + 2));
      }
    }
  }
}

void GetGaussQuadratureCoordinates(int GO, Eigen::VectorXd &GP);

void GetGaussQuadratureWeights(int GO, Eigen::VectorXd &w);

}  // namespace da::sha
