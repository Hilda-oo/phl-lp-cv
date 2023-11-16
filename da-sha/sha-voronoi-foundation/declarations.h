#pragma once

#include <set>
#include <vector>

#include "sha-base-framework/declarations.h"
#include "sha-surface-mesh/matmesh.h"

namespace da {
namespace sha {
using BoundaryVertexLoop = std::vector<index_t>;
struct PolygonFace {
  std::vector<BoundaryVertexLoop> boundary_vtx_loops;
  std::vector<index_t> triangle_face_indices_in_cell;
  Eigen::MatrixXi mat_triangle_faces;
};

struct Polyhedron {
  // Eigen::MatrixXd mat_coordinates;
  std::vector<PolygonFace> polygons;
};

struct VoronoiCell {
  MatMesh3 cell_triangle_mesh;
  Polyhedron polyhedron;
  Eigen::Vector3d seed;
  std::vector<index_t> map_triangle_to_polygon_idx;
  // std::vector<int> map_triangle_to_origi n_face_idx;       // Only for clipped voronoi
};

struct VoronoiDiagram {
  std::vector<VoronoiCell> cells;
  std::vector<std::set<index_t>> map_cell_to_neighbor_indices;
};
}  // namespace sha
}  // namespace da
