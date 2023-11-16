#pragma once

#include <Eigen/Eigen>

#include <set>
#include <vector>

#include "linear_solver.h"

namespace da {
namespace cpt {
template <typename VectorTypeI, typename VectorTypeS, int Dimension = 3>
class EigenLibSolver : public LinearSolver<VectorTypeI, VectorTypeS, Dimension> {
  typedef LinearSolver<VectorTypeI, VectorTypeS, Dimension> Base;

 protected:
  Eigen::SparseMatrix<double> mat_coeff_;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> simplicial_LDLT_;

 public:
  void SetPattern(const std::vector<std::set<int>>& vNeighbor) override;
  void SetPattern(const Eigen::SparseMatrix<double>& mtr) override;  // NOTE: mtr must be SPD

  void AnalyzePattern() override;

  bool Factorize() override;

  void Solve(Eigen::VectorXd& rhs, Eigen::VectorXd& result) override;

  double CoeffMtr(int rowI, int colI) const override;

  void SetZero() override;

  void SetCoeff(int rowI, int colI, double val) override;

  void AddCoeff(int rowI, int colI, double val) override;

  void SetUnitRow(int rowI) override;

  void SetUnitCol(int colI, const std::set<int>& rowVIs) override;
};
}  // namespace cpt
}  // namespace da

#include "eigen_solver.cpp.impl"
