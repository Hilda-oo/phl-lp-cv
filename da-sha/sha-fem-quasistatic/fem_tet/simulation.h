#pragma once

#include <Eigen/Eigen>
#include <memory>
#include <set>
#include <vector>

#include "cpt-linear-solver/linear_solver.h"
#include "sha-simulation-utils/boundary_conditions.h"
#include "sha-surface-mesh/matmesh.h"

#define DIM_ 3

namespace da::sha {

class FEMTetQuasiSimulator {
 public:
  std::vector<DirichletBC> DirichletBCs_;
  std::vector<NeumannBC> NeumannBCs_;
  Eigen::Matrix<double, 6, 6> D_;  // constitutive matrix

 public:  // owned data
  int nN_, nEle_, nDof_;
  Eigen::MatrixXd TV_;   // vertices coordinates
  Eigen::MatrixXd TV1_;  // deformed vertices coordinates
  Eigen::MatrixXi TT_;   // vertice index of each tetrahedron
  Eigen::MatrixXi SF_;

  int eleNodeNum_;
  int eleDofNum_;
  std::vector<Eigen::Vector<int, 12>> eDof_;

 public:                  // owned features
  Eigen::VectorXd load_;  // load of each dof
  Eigen::VectorXd U_;     // dofs' displacement to be computed

  Eigen::VectorXi DBC_nI_;  // vertex in DBC
  Eigen::VectorXi isDBC_;   // 0: not in DBC, 1: in DBC

  Eigen::VectorXi SVI_;     // vertice indices of surface nodes
  Eigen::MatrixXi F_surf_;  // boundary vertice indices in surface triangles mesh

  // indices for fast access
  std::vector<std::set<int>> vNeighbor_;  // records all vertices' indices adjacent to each vertice
  std::vector<std::set<std::pair<int, int>>> vFLoc_;
  std::shared_ptr<cpt::LinearSolver<Eigen::VectorXi, Eigen::VectorXd, DIM_>> linSysSolver_;

 public:  // constructor
  FEMTetQuasiSimulator(Eigen::MatrixXd p_TV, Eigen::MatrixXi p_TT, Eigen::MatrixXi p_SF,
                       double p_YM, double p_PR, std::vector<DirichletBC> p_DirichletBCs,
                       std::vector<NeumannBC> p_NeumannBCs);

  ~FEMTetQuasiSimulator() {}

  void simulation();

  void output_surf_result();

  MatMesh3 GetSimulatedSurfaceMesh(Eigen::MatrixXd &mat_deformed_coordinates,
                                   Eigen::VectorXd &vtx_displacement,
                                   Eigen::VectorXd &vtx_stress);

  /**
   * Given query points Q, compute their displacements and stress
   * @param Q (nQ, 3), query points
   * @param QU return value, (nQ), norm of displacements
   * @param Qstress return value, (nQ, 6), stress size is (6) on each query point
   */
  void postprocess(Eigen::MatrixXd &Q, Eigen::VectorXd &QU, Eigen::MatrixXd &Qstress);

 private:
  void computeFeatures();
  void computeK();
  void solve();
  void setBC();
};
}  // namespace da::sha
