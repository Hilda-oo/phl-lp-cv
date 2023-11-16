#pragma once

#include <Eigen/Eigen>

#include <memory>
#include <vector>

#include "background_mesh.h"
#include "physical_domain.h"

#include "cpt-linear-solver/linear_solver.h"

namespace da::sha {
class CBNSimulator {
 public:
  explicit CBNSimulator(double p_YM1, double p_YM0, double p_PR, double p_penaltyYM,
                        std::shared_ptr<PhysicalDomain> p_physicalDomain,
                        const std::shared_ptr<NestedBackgroundMesh> &nested_background,
                        bool p_handlePhysicalDomain = true);

  ~CBNSimulator();

  void PreprocessBackgroundMesh();

  void ComputeFeatures();

  // compute Phi on one face
  void ComputePhiOnFace(const Eigen::MatrixXd &macV_f, const Eigen::MatrixXd &bV_f,
                        Eigen::MatrixXd &Phi_f);

  // compute Phi
  void ComputePhi();

  // preprocess NBC, DBC, vertex in physical domain
  void Preprocess();

  // prepare for micro integration: compute gp_ID, localGP_K, localGP_B, localGP_w
  void PrepareMicroIntegration();

  // simulation interface
  auto Simulate(const std::vector<Eigen::VectorXd> &rhos) -> Eigen::VectorXd;

  // compute M: transformation from micro surface dofs to micro internal dofs
  // compute and assemble global stiffness matrix, global load vector (Neumann boundary conditions)
  // apply Dirichlet boundary conditions
  void ComputeSystem();

  void Solve();  // solve the linear system

  /**
   * compute element displacement and element stress to each fine tet element
   * i.e. fine_tet_displacement, fine_tet_stress
   */
  void PostprocessFineMesh();

  /**
   * compute and return displacement, stress at query points.
   * Note: If query point not in background mesh, displacement is set to zero,
   * stress is set to identity matrix.
   * @param query_points query points
   * @param query_displacement return value, displacement of each query point
   * @param query_stress return value, stress of each query point
   */
  void QueryResults(const Eigen::MatrixXd &query_points,
                    std::vector<Eigen::Vector3d> &query_displacement,
                    std::vector<Eigen::Matrix3d> &query_stress);

  /**
   * compute and return displacement, stress at query points.
   * providing flag (0: outside mesh, 1: inside mesh), mac_index, mic_index
   * @param query_points
   * @param query_flag
   * @param query_mac_index
   * @param query_mic_index
   * @param query_displacement return value, displacement of each query point
   * @param query_stress return value, stress of each query point
   */
  void QueryResultsWithLocation(const Eigen::MatrixXd &query_points,
                                const Eigen::VectorXi &query_flag,
                                const Eigen::VectorXi &query_mac_index,
                                const Eigen::VectorXi &query_mic_index,
                                std::vector<Eigen::Vector3d> &query_displacement,
                                std::vector<Eigen::Matrix3d> &query_stress);

  // for point P with a global coordinate, get its responding element index
  // int getEleIDForPoint(const Eigen::RowVector3d &P) const;
  // update coordinates of points in triangle mesh (physical domain)
  void UpdateVInPD();

 public:
  double YM1, YM0;
  double penaltyYM;  // penalty parameter of Young's modulus
  std::shared_ptr<PhysicalDomain> physicalDomain;

  // elastic matrix for solid region (with 1.0 Young's modulus)
  Eigen::Matrix<double, 6, 6> D_;
  std::shared_ptr<NestedBackgroundMesh> nested_background_;

  cpt::LinearSolver<Eigen::VectorXi, Eigen::VectorXd, 3> *linSysSolver = nullptr;

  // density of each micro tet in each macro element
  std::vector<Eigen::VectorXd> rhos_;

 public:
  // all nodes in polygon mesh, (nNode, 3)
  Eigen::MatrixXd node;
  // macro dofs id in each element
  std::vector<Eigen::VectorXi> eDof;
  // nodes index of each face w.r.t each polygon
  std::vector<std::vector<Eigen::VectorXi>> nid;
  // duplicated index in S-Patches of each node in each face in each polygon
  std::vector<std::vector<std::vector<Eigen::VectorXi>>> dup;

  // micro nodes index of each face w.r.t tet in each polygon
  std::vector<std::vector<Eigen::VectorXi>> bnid;
  // de-duplicate micro nodes index of each face w.r.t tet in each polygon
  std::vector<std::vector<Eigen::VectorXi>> bnid_deDup;
  // bnid_deDup's index in bnid
  std::vector<std::vector<Eigen::VectorXi>> bnid_deDup_ID;
  // number of micro boundary nodes in each macro polygon
  std::vector<int> bnNode;
  // reorder vector (bnDof, inDof) of micro tet mesh in each macro polygon
  std::vector<Eigen::VectorXi> reorderVec;

 public:
  int nEle;   // number of macro elements (polygon)
  int nNode;  // number of vertex in this mesh
  int nDof;   // number of dofs in this mesh
  // number of face in each macro polygon
  std::vector<int> nFacePerPoly;
  // number of macro dofs in each macro polygon
  std::vector<int> eleDofNum;

  // number of micro nodes in each macro polygon
  std::vector<int> mic_nV;
  // number of micro tets in each macro polygon
  std::vector<int> mic_nT;

  // nodes id in each element
  std::vector<Eigen::VectorXi> eNode;
  // store neighbor nodes id of each node (neighbor: belong to one element)
  std::vector<std::set<int>> vNeighbor;

 public:
  // transformation matrix for each macro polygon element
  // transformation from micro surface dofs to macro dofs
  std::vector<Eigen::MatrixXd> Phi;

  // element ke of each micro tet in each macro polygon element
  std::vector<std::vector<Eigen::Matrix<double, 12, 12>>> mic_Ke;
  // volume of each micro tet in each macro polygon element
  std::vector<Eigen::VectorXd> mic_Vol;
  // model's initial volume
  double Vol0;

  // dofs id of all micro tets in each macro polygon element
  std::vector<Eigen::VectorXi> mic_eDof;
  // number of micro boundary dofs in each macro polygon element
  std::vector<int> nBDof;
  // number of micro internal dofs in each macro polygon element
  std::vector<int> nIDof;

  std::vector<Eigen::MatrixXd> elementM2;
  // transformation matrix of each macro element
  std::vector<Eigen::MatrixXd> elementM;
  // macro element stiffness matrix
  std::vector<Eigen::MatrixXd> elementKe;

  // macro element load vector
  std::vector<Eigen::VectorXd> elementLoad;
  // load vector
  Eigen::VectorXd load;

  // macro element solution(U) vector
  std::vector<Eigen::VectorXd> elementU;
  // solution(U) vector
  Eigen::VectorXd U;

  // integration order of Boundary Condition (BC) integration
  int integrationOrder_BC = 2;
  Eigen::VectorXd GP_BC;  // 1D Gauss Quadrature Points for BC
  Eigen::VectorXd GW_BC;  // 1D Gauss Quadrature Weights for BC

  // micro element index of Gauss Points in NBC
  std::vector<int> NBC_micI;
  // micro N of Gauss Points in NBC
  std::vector<Eigen::Matrix<double, 3, 12>> NBC_micN;
  // integration weight of Gauss Points in NBC
  // std::vector<double> NBC_w;
  // NBC value of Gauss Points in NBC
  std::vector<Eigen::Vector3d> NBC_val;
  // NBC's Gauss points index of each macro element
  std::vector<std::vector<int>> elementNBC;
  bool NBC_flag = false;

  // <<< variables for direct DBC
  // points for direct DBC
  Eigen::MatrixXd DBCV;
  // dof index for direct DBC
  std::vector<int> fixeddofs;
  // >>> variables for direct DBC

  // <<< variables for Nitsche method
  // macro element index of Gauss Points in DBC
  std::vector<int> DBC_macI;
  // micro element index of Gauss Points in DBC
  std::vector<int> DBC_micI;
  // micro N of Gauss Points in DBC
  std::vector<Eigen::Matrix<double, 3, 12>> DBC_micN;
  // micro B of Gauss Points in DBC
  std::vector<Eigen::Matrix<double, 6, 12>> DBC_micB;
  // normal vector Voigt of Gauss Points in DBC
  std::vector<Eigen::Matrix<double, 6, 3>> DBC_DT_mul_normal;
  // integration weight of Gauss Points in DBC
  std::vector<double> DBC_w;
  // DBC value of Gauss Points in DBC
  std::vector<Eigen::Vector3d> DBC_val;
  // DBC's Gauss points index of each macro element
  std::vector<std::vector<int>> elementDBC;
  // >>> variables for Nitsche method

  // displacements of fine nodes
  std::vector<std::vector<Eigen::Vector<double, 12>>> fine_tet_displacement;
  // stress of fine tets
  std::vector<std::vector<Eigen::Matrix3d>> fine_tet_stress;

  // whether to handle mesh of physical domain
  bool handlePhysicalDomain = true;
};
}  // namespace da::sha
