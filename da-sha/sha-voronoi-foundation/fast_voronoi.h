#pragma once

#include "declarations.h"

#include "sha-surface-mesh/matmesh.h"

#include <Eigen/Eigen>

#include <geogram/mesh/mesh.h>

namespace da {
namespace sha {
auto FastCreateRestrictedVoronoiDiagramFromMesh(const MatMesh3& mesh,
                                                const Eigen::MatrixXd& mat_seeds,
                                                double sharp_angle) -> VoronoiDiagram;

auto FastCreateRestrictedVoronoiDiagramFromMesh(GEO::Mesh& boundary_mesh,
                                                const Eigen::MatrixXd& mat_seeds,
                                                double sharp_angle) -> VoronoiDiagram;
}  // namespace sha
}  // namespace da