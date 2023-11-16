#pragma once

#include <utility>
#include <vector>
#include "sha-surface-mesh/matmesh.h"
#include "sha-volume-mesh/matmesh.h"

namespace da::sha {
struct NestedCell {
  sha::MatMesh3 macro_mesh;
  TetrahedralMatMesh tetrahedrons;
  std::vector<Eigen::MatrixXd> polyhedron_edges;
  Eigen::AlignedBox3d cell_box;
};

class NestedBackgroundMesh {
 public:
  explicit NestedBackgroundMesh(const std::vector<NestedCell> &nested_cells);

  auto GetPointLocation(const Eigen::Vector3d &p) -> std::pair<int, int> const;
  
  auto GetPointLocationAlternative(const Eigen::Vector3d &p, const double tol_macro,
  const double tol_micro)-> std::pair<int, int>;

  double ComputeAverageMicroTetEdgeLength() const;

  double ComputeMinMicroTetEdgeLength() const;

 public:
  std::vector<NestedCell> nested_cells_;
  // std::vector<TetrahedralMatMesh> micro_cells_;
  // std::vector<MatMesh3> macro_triangle_cells_;
  // std::vector<Eigen::AlignedBox3d> macro_cell_boxes_;
};
}  // namespace da:
