#include "sample.h"

#include <igl/barycentric_coordinates.h>
#include <igl/per_vertex_normals.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/principal_curvature.h>

#include <vector>

namespace da {
void SampleInBoundingSquare(const MatMesh3 &matmesh, const Eigen::MatrixXd &uv,
                            size_t num_on_long_axis, Eigen::MatrixXd &mat_sample_points,
                            Eigen::MatrixXd &mat_sample_normals) {
  Eigen::MatrixXd mat_vtx_normals;
  igl::per_vertex_normals(matmesh.mat_coordinates, matmesh.mat_faces, mat_vtx_normals);

  double left_u        = uv.col(0).minCoeff();
  double left_v        = uv.col(1).minCoeff();
  double length_u      = (uv.col(0).maxCoeff() - uv.col(0).minCoeff());
  double length_v      = (uv.col(1).maxCoeff() - uv.col(1).minCoeff());
  size_t num_samples_u = num_on_long_axis;
  size_t num_samples_v = num_on_long_axis;
  if (length_u > length_v) {
    num_samples_v = std::round(length_v / (length_u / num_on_long_axis));
  } else {
    num_samples_u = std::round(length_u / (length_v / num_on_long_axis));
  }
  Eigen::MatrixXd mat_all_samples(num_samples_u * num_samples_v, 3);
  for (index_t u_idx = 0, count = 0; u_idx < num_samples_u; u_idx++) {
    for (index_t v_idx = 0; v_idx < num_samples_v; v_idx++, count++) {
      mat_all_samples.row(count) << left_u + u_idx * length_u / (num_samples_u - 1),
          left_v + v_idx * length_v / (num_samples_v - 1), 0;
    }
  }
  Eigen::VectorXd sample_distances;
  Eigen::VectorXi sample_triangle_indices;
  Eigen::MatrixXd sample_projections;
  Eigen::MatrixXd uv0 = uv;
  uv0.conservativeResize(uv.rows(), 3);
  uv0.col(2).setZero();

  igl::point_mesh_squared_distance(mat_all_samples, uv0, matmesh.mat_faces, sample_distances,
                                   sample_triangle_indices, sample_projections);
  size_t num_samples = (sample_distances.array() < 1e-6).count();
  std::vector<Eigen::Vector3d> sample_point_list;
  std::vector<index_t> sample_point_triangle_indices;
  Eigen::MatrixXd sample_triangles_p, sample_triangles_q, sample_triangles_r;

  for (index_t idx = 0; idx < sample_distances.rows(); ++idx) {
    if (sample_distances(idx) < 1e-6) {
      sample_point_list.push_back(sample_projections.row(idx));
      sample_point_triangle_indices.push_back(sample_triangle_indices(idx));
    }
  }
  mat_sample_points.resize(sample_point_list.size(), 3);
  sample_triangles_p.resize(sample_point_list.size(), 3);
  sample_triangles_q.resize(sample_point_list.size(), 3);
  sample_triangles_r.resize(sample_point_list.size(), 3);

  for (index_t idx = 0; idx < sample_point_list.size(); ++idx) {
    index_t triangle_idx        = sample_point_triangle_indices[idx];
    mat_sample_points.row(idx)  = sample_point_list[idx];
    sample_triangles_p.row(idx) = uv0.row(matmesh.mat_faces(triangle_idx, 0));
    sample_triangles_q.row(idx) = uv0.row(matmesh.mat_faces(triangle_idx, 1));
    sample_triangles_r.row(idx) = uv0.row(matmesh.mat_faces(triangle_idx, 2));
  }
  Eigen::MatrixXd samples_barycentric_coordinates;
  igl::barycentric_coordinates(mat_sample_points, sample_triangles_p, sample_triangles_q,
                               sample_triangles_r, samples_barycentric_coordinates);
  mat_sample_normals.resize(mat_sample_points.rows(), 3);
  for (index_t idx = 0; idx < mat_sample_points.rows(); ++idx) {
    index_t triangle_idx       = sample_point_triangle_indices[idx];
    mat_sample_points.row(idx) = (matmesh.mat_coordinates.row(matmesh.mat_faces(triangle_idx, 0)) *
                                  samples_barycentric_coordinates(idx, 0)) +
                                 (matmesh.mat_coordinates.row(matmesh.mat_faces(triangle_idx, 1)) *
                                  samples_barycentric_coordinates(idx, 1)) +
                                 (matmesh.mat_coordinates.row(matmesh.mat_faces(triangle_idx, 2)) *
                                  samples_barycentric_coordinates(idx, 2));
    mat_sample_normals.row(idx) = (mat_vtx_normals.row(matmesh.mat_faces(triangle_idx, 0)) *
                                   samples_barycentric_coordinates(idx, 0)) +
                                  (mat_vtx_normals.row(matmesh.mat_faces(triangle_idx, 1)) *
                                   samples_barycentric_coordinates(idx, 1)) +
                                  (mat_vtx_normals.row(matmesh.mat_faces(triangle_idx, 2)) *
                                   samples_barycentric_coordinates(idx, 2));
    mat_sample_normals.row(idx).normalize();
  }
}

void SampleInBoundingSquareByCurvatureField(const MatMesh3 &matmesh, const Eigen::MatrixXd &uv,
                            size_t num_on_long_axis, Eigen::MatrixXd &mat_sample_points,
                            Eigen::MatrixXd &mat_sample_normals ,Eigen::MatrixXd &mat_sample_principal_curvatures) {
  //模型各顶点法线
  Eigen::MatrixXd mat_vtx_normals;
  //模型各顶点主曲率方向（最大曲率方向）
  Eigen::MatrixXd mat_vtx_principal_curvatures;
  //没用到，凑输出的
  Eigen::MatrixXd mat_vtx_min_pd;
  Eigen::MatrixXd mat_vtx_max_pv;
  Eigen::MatrixXd mat_vtx_min_pv;
  //计算每个顶点的法向量，赋给mat_vtx_normals
  igl::per_vertex_normals(matmesh.mat_coordinates, matmesh.mat_faces, mat_vtx_normals);
  igl::principal_curvature(matmesh.mat_coordinates, matmesh.mat_faces, mat_vtx_principal_curvatures,
                           mat_vtx_min_pd, mat_vtx_max_pv, mat_vtx_min_pv);

  double left_u        = uv.col(0).minCoeff();
  double left_v        = uv.col(1).minCoeff();
  double length_u      = (uv.col(0).maxCoeff() - uv.col(0).minCoeff());
  double length_v      = (uv.col(1).maxCoeff() - uv.col(1).minCoeff());
  size_t num_samples_u = num_on_long_axis;
  size_t num_samples_v = num_on_long_axis;
  if (length_u > length_v) {
    num_samples_v = std::round(length_v / (length_u / num_on_long_axis));
  } else {
    num_samples_u = std::round(length_u / (length_v / num_on_long_axis));
  }
  Eigen::MatrixXd mat_all_samples(num_samples_u * num_samples_v, 3);
  for (index_t u_idx = 0, count = 0; u_idx < num_samples_u; u_idx++) {
    for (index_t v_idx = 0; v_idx < num_samples_v; v_idx++, count++) {
      mat_all_samples.row(count) << left_u + u_idx * length_u / (num_samples_u - 1),
          left_v + v_idx * length_v / (num_samples_v - 1), 0;
    }
  }
  Eigen::VectorXd sample_distances;
  Eigen::VectorXi sample_triangle_indices;
  Eigen::MatrixXd sample_projections;
  Eigen::MatrixXd uv0 = uv;
  uv0.conservativeResize(uv.rows(), 3);
  uv0.col(2).setZero();

  igl::point_mesh_squared_distance(mat_all_samples, uv0, matmesh.mat_faces, sample_distances,
                                   sample_triangle_indices, sample_projections);
  size_t num_samples = (sample_distances.array() < 1e-6).count();
  std::vector<Eigen::Vector3d> sample_point_list;
  std::vector<index_t> sample_point_triangle_indices;
  Eigen::MatrixXd sample_triangles_p, sample_triangles_q, sample_triangles_r;

  for (index_t idx = 0; idx < sample_distances.rows(); ++idx) {
    if (sample_distances(idx) < 1e-6) {
      sample_point_list.push_back(sample_projections.row(idx));
      sample_point_triangle_indices.push_back(sample_triangle_indices(idx));
    }
  }
  mat_sample_points.resize(sample_point_list.size(), 3);
  sample_triangles_p.resize(sample_point_list.size(), 3);
  sample_triangles_q.resize(sample_point_list.size(), 3);
  sample_triangles_r.resize(sample_point_list.size(), 3);

  for (index_t idx = 0; idx < sample_point_list.size(); ++idx) {
    index_t triangle_idx        = sample_point_triangle_indices[idx];
    mat_sample_points.row(idx)  = sample_point_list[idx];
    sample_triangles_p.row(idx) = uv0.row(matmesh.mat_faces(triangle_idx, 0));
    sample_triangles_q.row(idx) = uv0.row(matmesh.mat_faces(triangle_idx, 1));
    sample_triangles_r.row(idx) = uv0.row(matmesh.mat_faces(triangle_idx, 2));
  }
  Eigen::MatrixXd samples_barycentric_coordinates;
  igl::barycentric_coordinates(mat_sample_points, sample_triangles_p, sample_triangles_q,
                               sample_triangles_r, samples_barycentric_coordinates);
  mat_sample_normals.resize(mat_sample_points.rows(), 3);
  mat_sample_principal_curvatures.resize(mat_sample_points.rows(), 3);
  //实现对points（点坐标）、normals（法向量）、principal_directions（主方向）的计算
  for (index_t idx = 0; idx < mat_sample_points.rows(); ++idx) {
    index_t triangle_idx       = sample_point_triangle_indices[idx];
    mat_sample_points.row(idx) = (matmesh.mat_coordinates.row(matmesh.mat_faces(triangle_idx, 0)) *
                                  samples_barycentric_coordinates(idx, 0)) +
                                 (matmesh.mat_coordinates.row(matmesh.mat_faces(triangle_idx, 1)) *
                                  samples_barycentric_coordinates(idx, 1)) +
                                 (matmesh.mat_coordinates.row(matmesh.mat_faces(triangle_idx, 2)) *
                                  samples_barycentric_coordinates(idx, 2));
    mat_sample_normals.row(idx) = (mat_vtx_normals.row(matmesh.mat_faces(triangle_idx, 0)) *
                                   samples_barycentric_coordinates(idx, 0)) +
                                  (mat_vtx_normals.row(matmesh.mat_faces(triangle_idx, 1)) *
                                   samples_barycentric_coordinates(idx, 1)) +
                                  (mat_vtx_normals.row(matmesh.mat_faces(triangle_idx, 2)) *
                                   samples_barycentric_coordinates(idx, 2));
    mat_sample_normals.row(idx).normalize();
    mat_sample_principal_curvatures.row(idx) = (mat_vtx_principal_curvatures.row(matmesh.mat_faces(triangle_idx, 0)) *
                                                samples_barycentric_coordinates(idx, 0)) +
                                               (mat_vtx_principal_curvatures.row(matmesh.mat_faces(triangle_idx, 1)) *
                                                samples_barycentric_coordinates(idx, 1)) +
                                               (mat_vtx_principal_curvatures.row(matmesh.mat_faces(triangle_idx, 2)) *
                                                samples_barycentric_coordinates(idx, 2));
    mat_sample_principal_curvatures.row(idx).normalize();
  }
}
}  // namespace da
