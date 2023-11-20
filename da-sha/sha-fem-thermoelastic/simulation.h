#pragma once

#include <Eigen/CholmodSupport>
#include <set>
#include <vector>
#include "sha-base-framework/frame.h"
#include "sha-simulation-utils/boundary_conditions.h"

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

  void ComputeGlobalK() {
    K_ = Eigen::MatrixXd::Zero(nDof_, nDof_);
    for (int eI = 0; eI < nEle_; ++eI) {
      auto ele_id_dof = eDof_[eI];
      K_(ele_id_dof, ele_id_dof) += eleKe_[eI];
    }
  }

  void SetRhos(Eigen::VectorXd p_rhos) {
    int size = p_rhos.size();
    rhos_.resize(size);
    rhos_ = p_rhos;
  }

 public:
  std::vector<Eigen::MatrixXd> eleKe_;
  Eigen::MatrixXd K_;
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

  Eigen::VectorXd rhos_;
  Eigen::SparseMatrix<double> K_spMat_;
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver_;
  Eigen::SparseMatrix<double> F_;
};
}  // namespace da::sha
