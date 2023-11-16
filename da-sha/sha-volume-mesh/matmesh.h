#pragma once

#include <Eigen/Eigen>

#include "mesh.h"
#include "sha-base-framework/frame.h"

namespace da {
namespace sha {
struct TetrahedralMatMesh {
  Eigen::MatrixXd mat_coordinates;
  Eigen::MatrixXi mat_tetrahedrons;

  size_t NumVertices() const { return mat_coordinates.rows(); }
  size_t NumTetrahedrons() const { return mat_tetrahedrons.rows(); }
  bool IsTetrahedral() const {
    return mat_tetrahedrons.rows() == 0 || mat_tetrahedrons.cols() == 4;
  }
};

struct HexahedralMatMesh {
  Eigen::MatrixXd mat_coordinates;
  Eigen::MatrixXi mat_hexahedrons;

  size_t NumVertices() const { return mat_coordinates.rows(); }
  size_t NumHexahedrons() const { return mat_hexahedrons.rows(); }
  bool IsHexahedral() const { return mat_hexahedrons.rows() == 0 || mat_hexahedrons.cols() == 8; }
};

auto CreateMatMeshFromTetrahedralMesh(const TetrahedralMesh &mesh) -> TetrahedralMatMesh;
auto CreateMatMeshFromHexahedralMesh(const HexahedralMesh &mesh) -> HexahedralMatMesh;
auto CreateSurfaceMatrixFromTetrahedralTopoMesh(const TetrahedralTopoMesh &topomesh)
    -> Eigen::MatrixXi;
auto CreateSurfaceMatrixFromHexahedralTopoMesh(const HexahedralTopoMesh &topomesh)
    -> Eigen::MatrixXi;

auto CreateTetrahedralMeshFromMatMesh(const TetrahedralMatMesh &matmesh) -> TetrahedralMesh;
auto CreateHexahedralMeshFromMatMesh(const HexahedralMatMesh &matmesh) -> HexahedralMesh;

auto CreateTetrahedralTopoMeshFromMatrix(size_t num_vertices,
                                         const Eigen::MatrixXi &mat_tetrahedrons)
    -> TetrahedralTopoMesh;
auto CreateHexahedralTopoMeshFromMatrix(size_t num_vertices, const Eigen::MatrixXi &mat_hexahedrons)
    -> HexahedralTopoMesh;

auto CombineTwoTetrahedralMatMeshes(const TetrahedralMatMesh &matmesh_1,
                                    const TetrahedralMatMesh &matmesh_2) -> TetrahedralMatMesh;

auto CombineTwoHexahedralMatMeshes(const HexahedralMatMesh &matmesh_1,
                                   const HexahedralMatMesh &matmesh_2) -> HexahedralMatMesh;
}  // namespace sha
using HexahedralMatMesh  = sha::HexahedralMatMesh;
using TetrahedralMatMesh = sha::TetrahedralMatMesh;
}  // namespace da
