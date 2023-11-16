#pragma once

#include <Eigen/Eigen>

#include "mesh3.h"

namespace da {
namespace sha {
struct MatMesh3 {
  Eigen::MatrixXd mat_coordinates;
  Eigen::MatrixXi mat_faces;

  size_t NumVertices() const { return mat_coordinates.rows(); }
  size_t NumFaces() const { return mat_faces.rows(); }
  auto AlignedBox(double expand = 0) const {
    auto domain                  = Eigen::AlignedBox3d(mat_coordinates.colwise().minCoeff(),
                                                       mat_coordinates.colwise().maxCoeff());
    Eigen::Vector3d domain_sizes = domain.sizes();
    domain.min() -= domain_sizes * expand;
    domain.max() += domain_sizes * expand;
    return domain;
  }
};

struct MatMesh2 {
  Eigen::MatrixXd mat_coordinates;
  Eigen::MatrixXi mat_beams;

  size_t NumVertices() const { return mat_coordinates.rows(); }
  size_t NumBeams() const { return mat_beams.rows(); }
};

auto CreateMatMesh3FromSurfaceMesh3(const SurfaceMesh3 &mesh3, size_t num_face_vertices = 3)
    -> MatMesh3;
auto CreateSurfaceMesh3FromMatMesh3(const MatMesh3 &matmesh3) -> SurfaceMesh3;
auto CreateFaceMatrixFromSurfaceTopoMesh3(const SurfaceTopoMesh3 &mesh3,
                                          size_t num_face_vertices = 3) -> Eigen::MatrixXi;

auto CombineTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2) -> MatMesh3;
auto CombineTwoMatMesh2(const MatMesh2 &matmesh_1, const MatMesh2 &matmesh_2) -> MatMesh2;

auto BooleanIntersectTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2,
                                 Eigen::VectorXi &brith_face_indices) -> MatMesh3;
auto BooleanUnionTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2,
                             Eigen::VectorXi &brith_face_indices) -> MatMesh3;
auto BooleanMinusTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2,
                             Eigen::VectorXi &brith_face_indices) -> MatMesh3;

auto BooleanIntersectTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2) -> MatMesh3;
auto BooleanUnionTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2) -> MatMesh3;
auto BooleanMinusTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2) -> MatMesh3;

auto RemoveUnreferencedVertices(const MatMesh3 &matmesh) -> MatMesh3;
auto RemoveUnreferencedVertices(const MatMesh2 &matmesh) -> MatMesh2;

auto Materialize(const MatMesh2 &matmesh, const Eigen::VectorXd &beam_radiuses) -> MatMesh3;
}  // namespace sha
using sha::MatMesh2;
using sha::MatMesh3;
}  // namespace da
