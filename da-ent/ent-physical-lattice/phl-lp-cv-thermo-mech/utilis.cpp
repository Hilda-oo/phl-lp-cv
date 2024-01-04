#include "utilis.h"

#include <cmath>
#include <iostream>
#include <fstream>


namespace da {
Eigen::Vector3d ComputeTriangularCircumcenter(const Eigen::Vector3d &vertex_a,
                                              const Eigen::Vector3d &vertex_b,
                                              const Eigen::Vector3d &vertex_c) {
  const double temp[3]{(vertex_b - vertex_c).squaredNorm(), (vertex_a - vertex_c).squaredNorm(),
                       (vertex_a - vertex_b).squaredNorm()};

  const double ba[3]{temp[0] * (temp[1] + temp[2] - temp[0]),
                     temp[1] * (temp[2] + temp[0] - temp[1]),
                     temp[2] * (temp[0] + temp[1] - temp[2])};
  const double sum = ba[0] + ba[1] + ba[2];
  Eigen::Vector3d circumcenter;
  circumcenter = (ba[0] / sum) * vertex_a + (ba[1] / sum) * vertex_b + (ba[2] / sum) * vertex_c;
  return circumcenter;
}

double Heaviside(double value, double epsilon) {
  const double a = 1e-9;
  if (value > epsilon) {
    return 1;
  } else if (value >= -epsilon) {
    return 3.0 * (1 - a) / 4.0 * (value / epsilon - std::pow(value / epsilon, 3) / 3.0) +
           (1 + a) / 2.0;
  } else {
    return a;
  }
}

std::vector<Eigen::MatrixXd> ReadPolyhedronEdgesFromPath(const fs_path &path) {
  std::vector<Eigen::MatrixXd> polyhedron_edges;
  Eigen::MatrixXd mat_coordinates;
  std::ifstream in_stream(path.string());

  int num_vertices;
  int num_faces;

  in_stream >> num_vertices;

  mat_coordinates.resize(num_vertices, 3);
  for (index_t vtx_idx = 0; vtx_idx < num_vertices; vtx_idx++) {
    in_stream >> mat_coordinates(vtx_idx, 0) >> mat_coordinates(vtx_idx, 1) >>
        mat_coordinates(vtx_idx, 2);
  }

  in_stream >> num_faces;
  polyhedron_edges.resize(num_faces);

  for (index_t face_idx = 0; face_idx < num_faces; face_idx++) {
    int num_edges, vertex_a_idx, vertex_b_idx;
    in_stream >> num_edges;
    polyhedron_edges[face_idx].resize(num_edges, 6);
    for (index_t edge_idx = 0; edge_idx < num_edges; edge_idx++) {
      in_stream >> vertex_a_idx >> vertex_b_idx;
      polyhedron_edges[face_idx].row(edge_idx).head(3) = mat_coordinates.row(vertex_a_idx);
      polyhedron_edges[face_idx].row(edge_idx).tail(3) = mat_coordinates.row(vertex_b_idx);
    }
  }
  return polyhedron_edges;
}

}  // namespace da
