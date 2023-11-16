#pragma once

#include <igl/fast_winding_number.h>
#include <Eigen/Eigen>

#include "geometry.h"

#include "sha-surface-mesh/matmesh.h"

namespace da::sha {

class PhysicalDomain {
 private:
  igl::FastWindingNumberBVH fwn_bvh;

 public:
  // Boundary Conditions
  std::vector<Eigen::Matrix<double, 2, 3>> NBCRelBBox;
  std::vector<Eigen::Matrix<double, 2, 3>> NBCBBox;
  std::vector<Eigen::Vector3d> NBCVal;

  std::vector<Eigen::Matrix<double, 2, 3>> DBCRelBBox;
  std::vector<Eigen::Matrix<double, 2, 3>> DBCBBox;
  std::vector<Eigen::Vector3d> DBCVal;

  int nNBC;
  std::vector<std::pair<Eigen::RowVector3d, Eigen::Vector3d>> NBC;
  int nDBC;
  std::vector<std::pair<int, Eigen::Vector3d>> DBC;

  bool use_Nitsche = false;
  // penalty coefficient for Nitsche method
  double penaltyNitsche = 1e10;

 public:
  MatMesh3 mesh_;
  Eigen::MatrixXd V1;
  std::vector<Triangle> Tri;
  int numV, numF;

  Eigen::Matrix<double, 2, 3> bbox;  // bounding box of physical domain
  Eigen::RowVector3d lenBBox;        // length of bounding box in 3 dimensions

 public:
  explicit PhysicalDomain(const MatMesh3 &mesh,
                          const std::vector<Eigen::Matrix<double, 2, 3>> &p_NBCRelBBox,
                          const std::vector<Eigen::Vector3d> &p_NBCVal,
                          const std::vector<Eigen::Matrix<double, 2, 3>> &p_DBCRelBBox,
                          const std::vector<Eigen::Vector3d> &p_DBCVal);

 public:
  // after modification of V, update bbox, triangle and fwn_bvh
  void Update();

  // given query Q (global coordinates), return W (their domain index)
  // 0: fictitious domain
  // 1: physical domain
  // domain index corresponds to the material index
  void GetDomainID(const Eigen::MatrixXd &Q, Eigen::VectorXi &domainID) const;
  // initialize boundary conditions
  void InitializeBoundaryConditions();

  /**
   * Directly set boundary conditions, use Nitsche method for DBC
   * @param p_nNBC number of Neumann BC
   * @param p_NBC Neumann BC
   * @param p_nDBC number of Dirichlet BC
   * @param DBC Dirichlet BC
   */
  void SetBoundaryConditions(
      int p_nNBC, const std::vector<std::pair<Eigen::RowVector3d, Eigen::Vector3d>> &p_NBC,
      int p_nDBC, const std::vector<std::pair<int, Eigen::Vector3d>> &DBC, double p_penaltyNitsche);

  // debug
  void WriteNBCToVtk(const fs_path &path);
  void WriteDBCToObj(const fs_path &path);
  void WriteV1MeshToObj(const fs_path &path);
};

}  // namespace da::sha
