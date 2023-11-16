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
double approximate(const double x, const double eps);

auto ComputeGradientOperatorsOfTetrahedronMesh(const TetrahedralMatMesh &tetrahedral_matmesh)
    -> std::vector<Eigen::Matrix<double, 3, 4>>;

using VertexIndexEdge = std::pair<index_t, index_t>;
using Chain           = std::deque<VertexIndexEdge>;
auto ConvertEdgesSoupToChains(const std::vector<std::pair<index_t, index_t>> &edges_soup)
    -> std::vector<Chain>;

enum Orientation { X = 0, NegX, Y, NegY, Z, NegZ };

auto GetNondirectionalIndexByOrientation(Orientation orientation) -> index_t;

struct OrientedPatch {
  std::vector<SurfaceMesh3::Face_index> face_indices;
  std::set<SurfaceMesh3::Edge_index> boundary_edge_indices;
  std::map<index_t, double> neighbor_patches_with_boundary_length;
  std::set<Orientation> neighbor_orientations;
  Orientation orientation;
  double edge_length;
};

void DivideMeshIntoPatchesByOrientation(const SurfaceTopoMesh3 &mesh3,
                                        const Eigen::MatrixXd &mat_coordinates,
                                        const std::vector<Orientation> &surface_orientations,
                                        std::vector<OrientedPatch> &oriented_patches,
                                        std::vector<index_t> &map_face_to_patch,
                                        std::vector<VertexIndexEdge> &boundary_edges_soup);

auto MarkMeshFaceOrientations(const Eigen::MatrixXd &mat_coordinates,
                              const Eigen::MatrixXi &mat_faces) -> std::vector<Orientation>;

void PostProcessOptimalTetrahedronPolycube(const TetrahedralMatMesh &tetrahedral_matmesh,
                                           Eigen::MatrixXd &mat_tetrahedral_polycube_coordinates);

auto RemeshTetrahedronPolycubeToHexhahedron(
    const TetrahedralMatMesh &tetrahedral_matmesh,
    const Eigen::MatrixXd &mat_tetrahedral_polycube_coordinates, const Eigen::Vector3d &scale)
    -> HexahedralMatMesh;

auto DeformHexadralPolycubeMeshToOriginalDomain(
    const TetrahedralMatMesh &tetrahedral_matmesh,
    const Eigen::MatrixXd &mat_polycube_tetrahedral_coordinates,
    const HexahedralMatMesh &hexahedral_polycube_matmesh) -> Eigen::MatrixXd;
}  // namespace sha
}  // namespace da
