#pragma once

#include <Eigen/Eigen>

#include <set>
#include <string>
#include <vector>

#include "linear_solver.h"

#include "cholmod.h"

namespace da {
namespace cpt {
template <typename VectorTypeI, typename VectorTypeS, int Dimension = 3>
class CHOLMODSolver : public LinearSolver<VectorTypeI, VectorTypeS, Dimension> {
  typedef LinearSolver<VectorTypeI, VectorTypeS, Dimension> Base;

 protected:
  cholmod_common cm_;
  cholmod_sparse* A_;
  cholmod_factor* L_;
  cholmod_dense *b_, *solution_;
  cholmod_dense *x_cd_, *y_cd_;  // for multiply

  void *Ai_, *Ap_, *Ax_, *bx_, *solutionx_, *x_cdx_, *y_cdx_;

 public:
  CHOLMODSolver();
  ~CHOLMODSolver();

  void SetPattern(const std::vector<std::set<int>>& vNeighbor) override;
  void SetPattern(const Eigen::SparseMatrix<double>& mtr) override;  // NOTE: mtr must be SPD
  void Load(const char* filePath, Eigen::VectorXd& rhs) override;

  void AnalyzePattern() override;

  bool Factorize() override;

  void Solve(Eigen::VectorXd& rhs, Eigen::VectorXd& result) override;

  void Multiply(const Eigen::VectorXd& x, Eigen::VectorXd& Ax) override;

  void OutputFactorization(const std::string& filePath) override;
};
}  // namespace cpt
}  // namespace da

#include "cholmod_solver.cpp.impl"
