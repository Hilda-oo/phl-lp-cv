#pragma once

#include <Eigen/Eigen>
#include <utility>
#include <vector>

#include "sha-simulation-3d/CBN/background_mesh.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-voronoi-foundation/voronoi.h"
#include "sha-voronoi-foundation/fast_voronoi.h"

namespace da {
class ModelAlgorithm {
 public:
  explicit ModelAlgorithm(const MatMesh3 &mesh,
                          const sha::NestedBackgroundMesh &nested_background_mesh, bool use_shell,
                          double shell_thickness);

  void InitBackgroundMesh();

  double ComputeMinDistance(const Eigen::Vector3d &x);

  void Update(const Eigen::MatrixXd &mat_variables);

  auto ComputeRhos() -> std::vector<Eigen::VectorXd>;

  auto ComputeRhos(const Eigen::MatrixXd &P) -> Eigen::VectorXd;

  void UpdateFD(const Eigen::MatrixXd &mat_variables);

  auto ComputeRhosFD() -> std::vector<Eigen::VectorXd>;

  auto ComputeRhosFD(Eigen::VectorXi &flag, const int seed_idx,
                     const Eigen::Vector3d &seed_position, double radius,
                     const Eigen::Vector3d &search_range) -> std::vector<Eigen::VectorXd>;

  auto ComputeRhosFDInSingleCell(int cell_idx, int updated_seed_id = -1) -> Eigen::VectorXd;

  void ComputeCellCentroids(Eigen::MatrixXd &centers, Eigen::VectorXi &flag) const;

 public:
  Eigen::MatrixXd current_seeds_;
  Eigen::VectorXd current_radiuses_;
  std::vector<Eigen::VectorXd> current_rhos;

  // speed up v
  std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> seeds_5_distance_temp;

  MatMesh3 mesh_;

  double sharp_angle_;

  sha::VoronoiDiagram voronoi_diagram_;
  MatMesh2 voronoi_beams_mesh_;
  std::vector<std::set<index_t>> map_voronoi_beam_idx_to_cell_indices;
  Eigen::VectorXd voronoi_beams_radiuses_;

  sha::NestedBackgroundMesh background_mesh_;

 private:
  std::vector<Eigen::MatrixXi> INIT_sorted_v_to_seeds_to_seeds_id;

 protected:
  bool use_shell_         = false;
  double shell_thickness_ = 0;
  std::vector<Eigen::VectorXd> signed_distances_of_micro_vertices_;
  double eps = 1e-3;
};
}  // namespace da