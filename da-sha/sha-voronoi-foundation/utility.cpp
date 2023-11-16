#include "utility.h"

namespace da::sha {
void ConvertMatmesh3ToGeoMesh(const MatMesh3 &mesh, GEO::Mesh &geo_mesh) {
  geo_mesh.clear();
  geo_mesh.vertices.set_dimension(3);

  const size_t num_vertices = mesh.NumVertices();
  const size_t num_faces    = mesh.NumFaces();

  for (index_t vertex_idx = 0; vertex_idx < num_vertices; vertex_idx++) {
    Eigen::RowVector3d coords = mesh.mat_coordinates.row(vertex_idx);
    geo_mesh.vertices.create_vertex(coords.data());
  }

  for (index_t face_idx = 0; face_idx < num_faces; ++face_idx) {
    geo_mesh.facets.create_triangle(mesh.mat_faces(face_idx, 0), mesh.mat_faces(face_idx, 1),
                                    mesh.mat_faces(face_idx, 2));
  }
  geo_mesh.facets.connect();
}

void ConvertGeoMeshToMatmesh3(const GEO::Mesh &geo_mesh, MatMesh3 &mesh) {
  const size_t num_vertices = geo_mesh.vertices.nb();
  const size_t num_faces    = geo_mesh.facets.nb();
  mesh.mat_coordinates.resize(num_vertices, 3);
  mesh.mat_faces.resize(num_faces, 3);
  for (index_t vertex_idx = 0; vertex_idx < num_vertices; vertex_idx++) {
    mesh.mat_coordinates.row(vertex_idx) << geo_mesh.vertices.point(vertex_idx).x,
        geo_mesh.vertices.point(vertex_idx).y, geo_mesh.vertices.point(vertex_idx).z;
  }
  for (index_t face_idx = 0; face_idx < num_faces; ++face_idx) {
    mesh.mat_faces.row(face_idx) << geo_mesh.facets.vertex(face_idx, 0),
        geo_mesh.facets.vertex(face_idx, 1), geo_mesh.facets.vertex(face_idx, 2);
  }
}
}  // namespace da::sha