#pragma once

#include "sha-base-framework/declarations.h"
#include "sha-surface-mesh/matmesh.h"

#include "declarations.h"

namespace da {
namespace sha {
auto CreateVoronoiDiagramInDomain(const Eigen::AlignedBox3d &domain,
                                  const Eigen::MatrixXd &mat_seeds, size_t num_lloyd_iterations = 0)
    -> VoronoiDiagram;

auto CreateRestrictedVoronoiDiagramFromMesh(const MatMesh3 &mesh, const Eigen::MatrixXd &mat_seeds,
                                            size_t num_lloyd_iterations, double sharp_angle,
                                            bool keep_number_flag = true) -> VoronoiDiagram;

auto SeparatePolygonIfNotConnected(const PolygonFace &polygon, const VoronoiCell &voronoi_cell)
    -> std::vector<PolygonFace>;

auto SeparatePolygonIfEdgeIsSharp(const PolygonFace &polygon, const VoronoiCell &voronoi_cell,
                                  const SurfaceMesh3 &voronoi_mesh3, double angle)
    -> std::vector<PolygonFace>;

void RemoveEmptyPolygonsFromVoronoiCell(VoronoiCell &voronoi_cell);

void ComputeLoopVertexByItsTriangles(PolygonFace &polygon);

auto ComputeRelatedEdgesFromVoronoiDiagram(const VoronoiDiagram &voronoi,
                                           const Eigen::AlignedBox3d &domain) -> MatMesh2;

auto ComputeRelatedEdgesFromVoronoiDiagram(
    const VoronoiDiagram &voronoi, const Eigen::AlignedBox3d &domain,
    std::vector<std::set<index_t>> &map_beam_idx_to_cell_indices) -> MatMesh2;
}  // namespace sha
}  // namespace da