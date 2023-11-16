#include "sample.h"

#include <igl/random_points_on_mesh.h>
#include <igl/signed_distance.h>
#include <random>
#include "sha-base-framework/frame.h"

namespace da {
namespace sha {
auto SamplePointsOnMeshSurfaceUniformly(const MatMesh3 &matmesh, size_t num_samples)
    -> std::pair<Eigen::MatrixXd, Eigen::VectorXi> {
  Eigen::MatrixXd mat_sample_coordinates;
  Eigen::VectorXi mat_sample_face_indices;
  Eigen::MatrixXd mat_sample_barycenters;
  igl::random_points_on_mesh(num_samples, matmesh.mat_coordinates, matmesh.mat_faces,
                             mat_sample_barycenters, mat_sample_face_indices,
                             mat_sample_coordinates);
  return {mat_sample_coordinates, mat_sample_face_indices};
}

Eigen::MatrixXd SamplePointsInMeshVolumeUniformly(const MatMesh3 &matmesh, size_t num_samples) {
  constexpr int kNumberOfMaxTrials = 100;

  Eigen::MatrixXd mat_sample_coordinates(num_samples, 3);
  Eigen::AlignedBox<double, 3> aligned_box_of_mesh = matmesh.AlignedBox();
  std::uniform_real_distribution<> pos_samplers[3] = {
      std::uniform_real_distribution<>(aligned_box_of_mesh.min().x(),
                                       aligned_box_of_mesh.max().x()),
      std::uniform_real_distribution<>(aligned_box_of_mesh.min().y(),
                                       aligned_box_of_mesh.max().y()),
      std::uniform_real_distribution<>(aligned_box_of_mesh.min().z(),
                                       aligned_box_of_mesh.max().z())};
  std::mt19937 random_engine((std::random_device())());

  log::info("Sample Domain(x): {}-{}", aligned_box_of_mesh.min().x(),
            aligned_box_of_mesh.max().x());
  log::info("Sample Domain(y): {}-{}", aligned_box_of_mesh.min().y(),
            aligned_box_of_mesh.max().y());
  log::info("Sample Domain(z): {}-{}", aligned_box_of_mesh.min().z(),
            aligned_box_of_mesh.max().z());

  igl::FastWindingNumberBVH fwn_bvh;
  igl::fast_winding_number(matmesh.mat_coordinates.template cast<float>().eval(), matmesh.mat_faces,
                           2, fwn_bvh);
#pragma omp parallel for
  for (index_t sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
    bool loop_exit_flag      = false;
    int num_mismatch_samples = 0;
    Eigen::MatrixXd query_point(1, 3);
    Eigen::VectorXd signed_distance_values;
    Eigen::VectorXi cloest_triangle_indices;
    Eigen::MatrixXd cloest_points;
    Eigen::MatrixXd cloest_normals;
    do {
      query_point.resize(1, 3);
      query_point << pos_samplers[0](random_engine), pos_samplers[1](random_engine),
          pos_samplers[2](random_engine);
      double w = fast_winding_number(fwn_bvh, 2, query_point.template cast<float>().eval());
      double s = 1. - 2. * std::abs(w);
      // igl::signed_distance(query_point, matmesh.mat_coordinates, matmesh.mat_faces,
      //                      igl::SIGNED_DISTANCE_TYPE_DEFAULT, signed_distance_values,
      //                      cloest_triangle_indices, cloest_points, cloest_normals);
      // if (signed_distance_values(0, 0) <= 0) {
      //   loop_exit_flag = true;
      // } else {
      //   num_mismatch_samples++;
      // }
      if (s <= 0) {
        loop_exit_flag = true;
      } else {
        num_mismatch_samples++;
      }
    } while (!loop_exit_flag && num_mismatch_samples < kNumberOfMaxTrials);
    if (loop_exit_flag) {
      mat_sample_coordinates.row(sample_idx) = query_point.row(0);
    } else {
      break;
    }
  }
  return mat_sample_coordinates;
}

Eigen::MatrixXd SamplePointsInAlignedBox3dUniformly(const Eigen::AlignedBox3d &aligned_box,
                                                    size_t num_samples) {
  Eigen::MatrixXd mat_sample_coordinates(num_samples, 3);
  std::uniform_real_distribution<> uniform_samplers[3] = {
      std::uniform_real_distribution<>(aligned_box.min().x(), aligned_box.max().x()),
      std::uniform_real_distribution<>(aligned_box.min().y(), aligned_box.max().y()),
      std::uniform_real_distribution<>(aligned_box.min().z(), aligned_box.max().z())};
  std::mt19937 random_engine((std::random_device())());

  int num_mismatch_samples = 0;

  for (index_t sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
    mat_sample_coordinates.row(sample_idx) << uniform_samplers[0](random_engine),
        uniform_samplers[1](random_engine), uniform_samplers[2](random_engine);
  }
  return mat_sample_coordinates;
}
}  // namespace sha
}  // namespace da
