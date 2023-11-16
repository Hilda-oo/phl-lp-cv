#pragma once

#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace da {
namespace cpt {
template <typename VectorTypeI, typename VectorTypeS, int Dimension = 3>
class LinearSolver {
 protected:
  int num_rows_;
  Eigen::VectorXi ia_, ja_;
  std::vector<std::map<int, int>> IJ2aI_;
  Eigen::VectorXd a_;

 public:
  virtual ~LinearSolver(void) {}

 public:
  virtual void SetPattern(const std::vector<std::set<int>>& vNeighbor);

  virtual void Load(const char* filePath, Eigen::VectorXd& rhs);
  virtual void Write(const char* filePath, const Eigen::VectorXd& rhs);
  virtual void SetPattern(const Eigen::SparseMatrix<double>& mtr);

  virtual void AnalyzePattern() = 0;

  virtual bool Factorize() = 0;

  virtual void Solve(Eigen::VectorXd& rhs, Eigen::VectorXd& result) = 0;

  virtual void Multiply(const Eigen::VectorXd& x, Eigen::VectorXd& Ax);

 public:
  virtual void OutputFactorization(const std::string& filePath);
  virtual double CoeffMtr(int rowI, int colI) const;
  virtual void GetCoeffMtr(Eigen::SparseMatrix<double>& mtr) const;
  virtual void GetCoeffMtr_lower(Eigen::SparseMatrix<double>& mtr) const;
  virtual void GetTriplets(const Eigen::VectorXi& nodeList,
                           std::vector<Eigen::Triplet<double>>& triplet) const;
  virtual void SetCoeff(int rowI, int colI, double val);
  virtual void SetCoeff(const LinearSolver<VectorTypeI, VectorTypeS, Dimension>* other,
                        double multiplier);
  virtual void SetZero();
  virtual void SetUnitRow(int rowI);
  virtual void SetUnitRow(int rowI, std::unordered_map<int, double>& rowVec);
  virtual void SetZeroCol(int colI);
  virtual void SetUnitCol(int colI, const std::set<int>& rowVIs);
  virtual void SetUnitColDim1(int colI, const std::set<int>& rowVIs);

  virtual void AddCoeff(int rowI, int colI, double val);
  virtual void PreconditionDiag(const Eigen::VectorXd& input, Eigen::VectorXd& output);
  virtual void GetMaxDiag(double& maxDiag);
  virtual void AddCoeff(const LinearSolver<VectorTypeI, VectorTypeS, Dimension>* other,
                        double multiplier);

  virtual int GetNumRows(void) const { return num_rows_; }
  virtual int GetNumNonZeros(void) const { return a_.size(); }
  virtual const std::vector<std::map<int, int>>& GetIJ2aI(void) const { return IJ2aI_; }
  virtual Eigen::VectorXi& Get_ia(void) { return ia_; }
  virtual Eigen::VectorXi& Get_ja(void) { return ja_; }
  virtual Eigen::VectorXd& Get_a(void) { return a_; }
  virtual const Eigen::VectorXd& Get_a(void) const { return a_; }
};
}  // namespace cpt
}  // namespace da

#include "linear_solver.cpp.impl"
