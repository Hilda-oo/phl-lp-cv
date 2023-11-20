#pragma once

#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include "sha-base-framework/declarations.h"

namespace da {
class AnisotropicMatWrapper {
 public:
  explicit AnisotropicMatWrapper();


  /**
   * Given query points Q, get their density
   * @param field_path field matrix file path
   * @param query_points (nQ, 3), query points
   * @param field_flag 1 for mech field, 0 for heat field
   */
  auto getAnisotropicMatByTopDensity(const fs_path &field_path, Eigen::MatrixXd &query_points,
                              bool field_flag = 1) -> std::vector<Eigen::Matrix3d>;

  /**
   * Given query points Q, get their stress
   * @param field_base_path field matrix file path
   * @param query_points (nQ, 3), query points
   * @param field_flag 1 for mech field, 0 for heat field
   */
  auto getAnisotropicMatByTopStress(const fs_path &field_base_path, Eigen::MatrixXd &query_points,
                              bool field_flag = 1) -> std::vector<Eigen::Matrix3d>;                                                      
  /**
   * Given mesh and the number of seeds, get sampled seeds
   * @param mesh_path mesh file path
   * @param seed_path seed file path to conserve
   * @param num_seed the number of seeds you need
   */
  void generateSampleSeedEntry(const fs_path &mesh_path, const fs_path &seed_path,
                               size_t num_seed = 100);

  /**
   * Given field matrix, get new field Matrix processed by laplace smoothing
   * @param field_maxtirx field matrix
   * @param lmd decline the speed of smoothing
   */
  auto fieldLaplaceSmooth(Eigen::MatrixXd &field_maxtrix, double lmd = 0.5) -> Eigen::MatrixXd;

  /**
   * Given a point, get its adjacent points
   * @param point coordinate of the point
   */
  auto getAdjPoints(Eigen::Vector3d &point) -> Eigen::MatrixXd;
  
  /**
   * Given stress s and points whose stress is s, output the stress orientation on all points in VTK file format 
   * @param out_path file path for outputing stress orientation
   * @param stress
   * @param points
   */
  void writeStressOrientation(std::vector<Eigen::Matrix3d> stress, Eigen::MatrixXd points, const fs_path &out_path);

  /**
   * Given txt file path, get a vector of the field value
   * @param field_mat_path field matrix file path
   */
  auto readDensityFromFile(const fs_path &field_mat_path) -> std::vector<double>;

  /**
   * Given VTK file path, get a vector of the cells'data
   * @param path VTK file path
   */
  auto readCellDataFromVTK(const fs_path &path) -> Eigen::VectorXd;

  /**
   * Given a nxn matrix, compute its eigen vector and eigen value;
   * @param A a nxn matrix
   */
  auto eigenDecomposition(const Eigen::MatrixXd &A) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd>;

  

 public:
  std::vector<double> top_density_;  // density by topopt for generating anisotropic matrix
  Eigen::MatrixXd TV_;   // vertices coordinates of the mesh (.vtk)
  Eigen::MatrixXi TT_;   // vertice index of each tetrahedron of the mesh (.vtk)
  Eigen::MatrixXd stress_on_point_; //x- y- z- stress on all points of the mesh
};
}  // namespace da