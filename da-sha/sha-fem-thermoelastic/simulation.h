#pragma once

#include <Eigen/CholmodSupport>
#include <set>
#include <vector>
#include "sha-base-framework/frame.h"
#include "sha-simulation-utils/boundary_conditions.h"
#include "unsupported/Eigen/src/KroneckerProduct/KroneckerTensorProduct.h"



namespace da::sha {

template <int dim>
class Simulation {
 public:  // constructor
  Simulation(Eigen::MatrixXd p_TV, Eigen::MatrixXi p_TT, std::vector<DirichletBC> p_DirichletBCs,
             std::vector<NeumannBC> p_NeumannBCs)
      : TV_(std::move(p_TV)),
        TT_(std::move(p_TT)),
        DirichletBCs_(std::move(p_DirichletBCs)),
        NeumannBCs_(std::move(p_NeumannBCs)) {
    nN_   = (int)TV_.rows();
    nEle_ = (int)TT_.rows();

    // update absBBox of Boundary Conditions
    Eigen::Vector3d modelMinBBox = TV_.colwise().minCoeff();
    Eigen::Vector3d modelMaxBBox = TV_.colwise().maxCoeff();
    for (auto &DBC : DirichletBCs_) {
      DBC.calcAbsBBox(modelMinBBox, modelMaxBBox);
    }
    for (auto &NBC : NeumannBCs_) {
      NBC.calcAbsBBox(modelMinBBox, modelMaxBBox);
    }
  }

  ~Simulation() {}

  virtual void simulation()      = 0;
  virtual void computeFeatures() = 0;
  virtual void computeK()        = 0;
  virtual void solve()           = 0;
  virtual void setBC()           = 0;

  int GetNumEles() { return nEle_; }

  int GetNumDofs() { return nDof_; }

  int Get_DOFS_EACH_ELE() { return eleDofNum_; }

  auto GetMapEleId2DofsVec() -> std::vector<Eigen::Vector<int, 4 * dim>> { return eDof_; }

  auto GetMapEleId2DofsMat() -> Eigen::MatrixXi {
    Eigen::MatrixXi eDofMat;
    int vector_size = eDof_.size();
    Assert(vector_size > 0, "eDof_.size() = 0");
    int eigen_vector_size = eDof_[0].size();
    eDofMat.resize(vector_size, eigen_vector_size);
    for (int eI = 0; eI < vector_size; ++eI) {
      eDofMat.row(eI) = eDof_[eI];
    }
    return eDofMat;
  }

  auto GetElementK(int eI) -> Eigen::MatrixXd { return eleKe_[eI]; }
  auto GetV() -> Eigen::VectorXd { return vol_; }

  void computeGlobalK() {
    Eigen::MatrixXi mat_ele2dofs = GetMapEleId2DofsMat();
    int dofs_each_ele            = Get_DOFS_EACH_ELE();  // 12 for mathe; 4 for heat
    iK_.resize(0);
    iK_ = Eigen::KroneckerProduct(mat_ele2dofs, Eigen::VectorXi::Ones(dofs_each_ele))
              .transpose()
              .reshaped();
    jK_.resize(0);
    jK_ = Eigen::KroneckerProduct(mat_ele2dofs, Eigen::RowVectorXi::Ones(dofs_each_ele))
              .transpose()
              .reshaped();
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(eleDofNum_ * eleDofNum_, nEle_);
    for (int eI = 0; eI < nEle_; ++eI) {
      K.col(eI) = eleKe_[eI].reshaped();
    }
    Eigen::VectorXd K_val = K.reshaped();
    std::vector<Eigen::Triplet<double>> triplet;
    triplet.resize(0);
    for (int i = 0; i < iK_.size(); i++) {
      triplet.push_back({iK_(i), jK_(i), K_val(i)});
    }
    K_spMat_.resize(nDof_, nDof_);
    K_spMat_.setFromTriplets(triplet.begin(), triplet.end());
  }

  void addDBCForLoad() {
    spdlog::debug("addDBCForLoad");
    for (auto dof_value : v_dofs_to_set_) {
      auto [dof, value] = dof_value;
      K_spMat_.coeffRef(dof, dof) *= 1e7;
      F_.coeffRef(dof, 0) = K_spMat_.coeffRef(dof, dof) * value;
    }
  }

 public:
  std::vector<Eigen::MatrixXd> eleKe_;
  Eigen::VectorXd vol_;

 public:
  std::vector<DirichletBC> DirichletBCs_;
  std::vector<NeumannBC> NeumannBCs_;

 public:  // owned data
  int nN_, nEle_, nDof_;
  Eigen::MatrixXd TV_;  // vertices coordinates
  Eigen::MatrixXi TT_;  // vertice index of each tetrahedron

  int eleNodeNum_;
  int eleDofNum_;
  std::vector<Eigen::Vector<int, 4 * dim>> eDof_;

 public:                  // owned features
  Eigen::VectorXd load_;  // load of each dof
  Eigen::VectorXd U_;     // dofs' displacement to be computed

  Eigen::VectorXi DBC_nI_;  // vertex in DBC
  Eigen::VectorXi isDBC_;   // 0: not in DBC, 1: in DBC

  // indices for fast access
  std::vector<std::set<int>> vNeighbor_;  // records all vertices' indices adjacent to each vertice
  //保存每个顶点对应的所有四面体以及在四面体中是第几个点
  std::vector<std::set<std::pair<int, int>>> vFLoc_;

  Eigen::SparseMatrix<double> K_spMat_;
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver_;
  Eigen::SparseMatrix<double> F_;
  Eigen::VectorXi iK_, jK_;
  std::vector<std::pair<unsigned, double>> v_dofs_to_set_;
};
}  // namespace da::sha
