#pragma once

#include <LpCVT/common/types.h>
#include "sha-surface-mesh/matmesh.h"

namespace Geex {
class Mesh;

template <class T>
class Plane;
using plane3 = Plane<double>;
}  // namespace Geex

namespace da {
class LpNormCVTWrapper {
 public:
  explicit LpNormCVTWrapper(const MatMesh3 &mesh, const int P);
  ~LpNormCVTWrapper();

  double EvaluateLpCVT(
      const Eigen::MatrixXd &mat_seeds,
      const std::vector<Eigen::Matrix3d> &anisotropyMatrix,
      Eigen::VectorXd &dL);

  // return query points to get stress 
  auto GetQuerySeeds(const Eigen::MatrixXd &mat_seeds) -> Eigen::MatrixXd;

  //generate clipped voronoi obj with shrink
  void WriteShrinkCVD(const fs_path &path, const Eigen::MatrixXd &seeds, double shrink = 0.7);


  Geex::Mesh *boundary_mesh_;
  std::vector<Geex::plane3> Q;
  const int P_;
};
}  // namespace da