#include "background_mesh.h"

#include <igl/in_element.h>
#include <igl/signed_distance.h>
#include <oneapi/tbb.h>
#include <Eigen/Eigen>
#include <vector>

namespace da::sha {
NestedBackgroundMesh::NestedBackgroundMesh(const std::vector<NestedCell> &nested_cells)
    : nested_cells_(nested_cells) {}

auto NestedBackgroundMesh::GetPointLocation(const Eigen::Vector3d &p) -> std::pair<int, int> const {
  int found_cell_idx = -1;
  int found_tet_idx  = -1;

  const int num_macro_cells = nested_cells_.size();

  for (index_t cell_idx = 0; cell_idx < num_macro_cells; ++cell_idx) {
    auto &cell_surface = nested_cells_.at(cell_idx).macro_mesh;
    auto &cell         = nested_cells_.at(cell_idx).tetrahedrons;
    Eigen::MatrixXd P(1, 3);
    Eigen::VectorXd S;
    Eigen::MatrixXi I;
    Eigen::MatrixXd C;
    Eigen::MatrixXd N;
    P.row(0) = p.transpose();
    igl::signed_distance(P, cell_surface.mat_coordinates, cell_surface.mat_faces,
                         igl::SIGNED_DISTANCE_TYPE_DEFAULT, S, I, C, N);
    if (S(0, 0) > 0) continue;
    found_cell_idx = cell_idx;

    igl::AABB<Eigen::MatrixXd, 3> tree;
    tree.init(cell.mat_coordinates, cell.mat_tetrahedrons);
    Eigen::VectorXi tet_indices;
    igl::in_element(cell.mat_coordinates, cell.mat_tetrahedrons, P, tree, tet_indices);
    found_tet_idx = tet_indices(0);
    break;
  }
  return std::make_pair(found_cell_idx, found_tet_idx);
}

auto NestedBackgroundMesh::GetPointLocationAlternative(const Eigen::Vector3d &p, const double tol_macro, const double tol_micro)
    -> std::pair<int, int> {
  int found_cell_idx = -1;
  int found_tet_idx  = -1;

  const int num_macro_cells = nested_cells_.size();

  for (index_t cell_idx = 0; cell_idx < num_macro_cells; ++cell_idx) {
    auto &cell_surface = nested_cells_.at(cell_idx).macro_mesh;
    auto &cell         = nested_cells_.at(cell_idx).tetrahedrons;
    Eigen::MatrixXd P(1, 3);
    Eigen::VectorXd S;
    Eigen::MatrixXi I;
    Eigen::MatrixXd C;
    Eigen::MatrixXd N;
    P.row(0) = p.transpose();
    igl::signed_distance(P, cell_surface.mat_coordinates, cell_surface.mat_faces,
                         igl::SIGNED_DISTANCE_TYPE_DEFAULT, S, I, C, N);
    if (S(0, 0) > tol_macro) continue;
    found_cell_idx = cell_idx;

    const auto &tetVs    = cell.mat_coordinates;
    const auto &tetCells = cell.mat_tetrahedrons;

    for (int tid = 0; tid < tetCells.rows(); ++tid) {
      Eigen::Matrix4d D0;
      D0.leftCols<3>() = tetVs(tetCells.row(tid), Eigen::all);
      D0.col(3).setOnes();
      bool flag = true;
      for (int i = 0; i < 4; ++i) {
        Eigen::MatrixXd Di = D0;
        Di(i, {0, 1, 2})   = p;
        if (Di.determinant() > tol_micro) {
          flag = false;
          break;
        }
      }
      if (flag) {
        found_tet_idx = tid;
        break;
      }
    }
    break;
  }
  return std::make_pair(found_cell_idx, found_tet_idx);
}

double NestedBackgroundMesh::ComputeAverageMicroTetEdgeLength() const {
  double total_micro_tet_edge_length = 0.0;
  const int num_macro_cells          = nested_cells_.size();

  for (index_t macro_cell_idx = 0; macro_cell_idx < num_macro_cells; ++macro_cell_idx) {
    const auto &macro_cell       = nested_cells_.at(macro_cell_idx);
    const Eigen::MatrixXd &micV_ = macro_cell.tetrahedrons.mat_coordinates;
    const Eigen::MatrixXi &micT_ = macro_cell.tetrahedrons.mat_tetrahedrons;
    const int num_tetrahedrons   = macro_cell.tetrahedrons.NumTetrahedrons();

    Eigen::MatrixXd t = micV_(micT_.col(0), Eigen::all) - micV_(micT_.col(1), Eigen::all);
    Eigen::VectorXd len(num_tetrahedrons);
    oneapi::tbb::parallel_for(0, num_tetrahedrons, 1,
                              [&](int micI) { len(micI) = t.row(micI).norm(); });
    total_micro_tet_edge_length += len.mean();
  }
  return total_micro_tet_edge_length / num_macro_cells;
}

double NestedBackgroundMesh::ComputeMinMicroTetEdgeLength() const {
  const size_t num_macro_cells     = nested_cells_.size();
  double min_micro_tet_edge_length = std::numeric_limits<double>::max();
  for (index_t cell_idx = 0; cell_idx < num_macro_cells; ++cell_idx) {
    auto &TV = nested_cells_.at(cell_idx).tetrahedrons.mat_coordinates;
    auto &TT = nested_cells_.at(cell_idx).tetrahedrons.mat_tetrahedrons;
    for (int i = 0; i < TT.rows(); ++i) {
      for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
          min_micro_tet_edge_length =
              std::min((TV.row(TT(i, a)) - TV.row(TT(i, b))).norm(), min_micro_tet_edge_length);
        }
      }
    }
  }
  return min_micro_tet_edge_length;
}
}  // namespace da