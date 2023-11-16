#pragma once

#include <Eigen/Eigen>

#include <deque>
#include <map>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#include "sha-base-framework/frame.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-volume-mesh/matmesh.h"

namespace da {
namespace sha {
auto GeneratePolycubeForTetrahedralMesh(const TetrahedralMatMesh &tetrahedral_matmesh,
                                        const double alpha, const double beta,
                                        const size_t num_iterations) -> Eigen::MatrixXd;

auto GenerateHexahedralMeshByPolycube(const TetrahedralMatMesh &tetrahedral_matmesh,
                                      const Eigen::MatrixXd &mat_tetrahedral_polycube_coordinates,
                                      const Eigen::Vector3i &num_cells) -> HexahedralMatMesh;
auto GenerateHexahedralMeshByPolycube(const TetrahedralMatMesh &tetrahedral_matmesh,
                                      const Eigen::MatrixXd &mat_tetrahedral_polycube_coordinates,
                                      const Eigen::Vector3i &num_cells,
                                      Eigen::MatrixXd &mat_polycube_coordinates)
    -> HexahedralMatMesh;
}  // namespace sha
}  // namespace da
