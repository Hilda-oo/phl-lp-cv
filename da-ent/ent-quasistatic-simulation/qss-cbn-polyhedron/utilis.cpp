#include "utilis.h"

#include <cmath>
#include <iostream>
#include <fstream>

namespace da {

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
