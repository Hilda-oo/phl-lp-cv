#pragma once

#include <Eigen/Eigen>

#include <vector>

namespace da {
namespace cpt {
auto SolveL1NormBasedPolycubeProblemByIfopt(
    const Eigen::MatrixXd &mat_coordinates, const Eigen::MatrixXi &mat_tetrahedrons,
    const Eigen::MatrixXi &mat_triangle_faces, const Eigen::MatrixXi &mat_adjacent_face_pairs,
    const std::vector<std::vector<int>> &map_face_idx_to_neighbor_indices,
    const std::vector<Eigen::Matrix<double, 3, 4>> &matrices_gradient_operator,
    const Eigen::VectorXd &tetrahedron_volumes, double alpha, double beta, size_t num_iterations)
    -> Eigen::MatrixXd;
}  // namespace cpt
}  // namespace da
