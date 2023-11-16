#include "matmesh.h"

#include <igl/combine.h>

#include <map>
#include <set>
#include <vector>

#include <boost/range/adaptor/indexed.hpp>

namespace da {
namespace sha {
auto CreateMatMeshFromTetrahedralMesh(const TetrahedralMesh &mesh) -> TetrahedralMatMesh {
  TetrahedralMatMesh mat_mesh;
  mat_mesh.mat_coordinates.resize(mesh.n_vertices(), 3);
  mat_mesh.mat_tetrahedrons.resize(mesh.n_cells(), 4);
  for (auto vertex : mesh.vertices()) {
    auto &point = mesh.vertex(vertex);
    mat_mesh.mat_coordinates.row(vertex.idx()) << point[0], point[1], point[2];
  }
  for (auto &tetrahedron : mesh.cells()) {
    for (auto &&[idx, vertex] : mesh.cell_vertices(tetrahedron) | boost::adaptors::indexed(0)) {
      mat_mesh.mat_tetrahedrons(tetrahedron.idx(), idx) = vertex.idx();
    }
  }
  return mat_mesh;
}

/**
 *
 *      5-------6
 *     /|      /|
 *    / |     / |
 *   3-------2  |
 *   |  4----|--7
 *   | /     | /
 *   |/      |/
 *   0-------1
 *
 */
auto CreateMatMeshFromHexahedralMesh(const HexahedralMesh &mesh) -> HexahedralMatMesh {
  std::map<index_t, index_t> map_openmesh_idx_to_vtk_idx = {{4, 0}, {0, 1}, {1, 2}, {7, 3},
                                                            {5, 4}, {3, 5}, {2, 6}, {6, 7}};
  std::map<index_t, index_t> map_vtk_idx_to_openmesh_idx = {{0, 4}, {1, 0}, {2, 1}, {3, 7},
                                                            {4, 5}, {5, 3}, {6, 2}, {7, 6}};
  std::array<index_t, 8> indices                         = {5, 6, 2, 1, 3, 7, 4, 0};
  HexahedralMatMesh mat_mesh;
  mat_mesh.mat_coordinates.resize(mesh.n_vertices(), 3);
  mat_mesh.mat_hexahedrons.resize(mesh.n_cells(), 8);
  for (auto vertex : mesh.vertices()) {
    auto &point = mesh.vertex(vertex);
    mat_mesh.mat_coordinates.row(vertex.idx()) << point[0], point[1], point[2];
  }
  for (auto &hexahedron : mesh.cells()) {
    int cnt = 0;
    for (auto he : mesh.cell_halfedges(hexahedron)) {
      mat_mesh.mat_hexahedrons(hexahedron.idx(), indices[cnt]) = mesh.from_vertex_handle(he).idx();
      if (++cnt == 8) break;
    }
  }
  return mat_mesh;
}

auto CreateTetrahedralMeshFromMatMesh(const TetrahedralMatMesh &mat_mesh) -> TetrahedralMesh {
  TetrahedralMesh mesh;
  for (index_t vertex_idx = 0; vertex_idx < mat_mesh.NumVertices(); ++vertex_idx) {
    mesh.add_vertex({mat_mesh.mat_coordinates(vertex_idx, 0),
                     mat_mesh.mat_coordinates(vertex_idx, 1),
                     mat_mesh.mat_coordinates(vertex_idx, 2)});
  }

  for (index_t cell_idx = 0; cell_idx < mat_mesh.NumTetrahedrons(); ++cell_idx) {
    mesh.add_cell(OpenVolumeMesh::VertexHandle(mat_mesh.mat_tetrahedrons(cell_idx, 0)),
                  OpenVolumeMesh::VertexHandle(mat_mesh.mat_tetrahedrons(cell_idx, 1)),
                  OpenVolumeMesh::VertexHandle(mat_mesh.mat_tetrahedrons(cell_idx, 2)),
                  OpenVolumeMesh::VertexHandle(mat_mesh.mat_tetrahedrons(cell_idx, 3)));
  }
  return mesh;
}

auto CreateTetrahedralTopoMeshFromMatrix(size_t num_vertices,
                                         const Eigen::MatrixXi &mat_tetrahedrons)
    -> TetrahedralTopoMesh {
  const size_t num_tetrahedrons = mat_tetrahedrons.rows();
  TetrahedralTopoMesh tetrahedral_topomesh;
  for (index_t vertex_idx = 0; vertex_idx < num_vertices; ++vertex_idx) {
    tetrahedral_topomesh.add_vertex();
  }
  for (index_t cell_idx = 0; cell_idx < num_tetrahedrons; ++cell_idx) {
    tetrahedral_topomesh.add_cell(OpenVolumeMesh::VertexHandle(mat_tetrahedrons(cell_idx, 0)),
                                  OpenVolumeMesh::VertexHandle(mat_tetrahedrons(cell_idx, 1)),
                                  OpenVolumeMesh::VertexHandle(mat_tetrahedrons(cell_idx, 2)),
                                  OpenVolumeMesh::VertexHandle(mat_tetrahedrons(cell_idx, 3)));
  }
  return tetrahedral_topomesh;
}

/**
 *
 *      5-------6
 *     /|      /|
 *    / |     / |
 *   3-------2  |
 *   |  4----|--7
 *   | /     | /
 *   |/      |/
 *   0-------1
 *
 */
auto CreateHexahedralMeshFromMatMesh(const HexahedralMatMesh &mat_mesh) -> HexahedralMesh {
  const size_t num_vertices                                           = mat_mesh.NumVertices();
  const size_t num_hexahedrons                                        = mat_mesh.NumHexahedrons();
  static const std::map<index_t, index_t> map_openmesh_idx_to_vtk_idx = {
      {4, 0}, {0, 1}, {1, 2}, {7, 3}, {5, 4}, {3, 5}, {2, 6}, {6, 7}};

  HexahedralMesh mesh;
  for (index_t vertex_idx = 0; vertex_idx < num_vertices; ++vertex_idx) {
    mesh.add_vertex({mat_mesh.mat_coordinates(vertex_idx, 0),
                     mat_mesh.mat_coordinates(vertex_idx, 1),
                     mat_mesh.mat_coordinates(vertex_idx, 2)});
  }
  for (index_t cell_idx = 0; cell_idx < num_hexahedrons; ++cell_idx) {
    std::vector<OpenVolumeMesh::VertexHandle> cell_vertices(8);
    for (size_t idx = 0; idx < 8; idx++) {
      cell_vertices[idx] = OpenVolumeMesh::VertexHandle(
          mat_mesh.mat_hexahedrons(cell_idx, map_openmesh_idx_to_vtk_idx.at(idx)));
    }
    mesh.add_cell(cell_vertices);
  }
  return mesh;
}
auto CreateHexahedralTopoMeshFromMatrix(size_t num_vertices, const Eigen::MatrixXi &mat_hexahedrons)
    -> HexahedralTopoMesh {
  static const std::map<index_t, index_t> map_openmesh_idx_to_vtk_idx = {
      {4, 0}, {0, 1}, {1, 2}, {7, 3}, {5, 4}, {3, 5}, {2, 6}, {6, 7}};

  const size_t num_hexahedrons = mat_hexahedrons.rows();

  HexahedralTopoMesh hexahedral_topomesh;

  for (index_t vertex_idx = 0; vertex_idx < num_vertices; ++vertex_idx) {
    hexahedral_topomesh.add_vertex();
  }
  for (index_t cell_idx = 0; cell_idx < num_hexahedrons; ++cell_idx) {
    std::vector<OpenVolumeMesh::VertexHandle> cell_vertices(8);
    for (size_t idx = 0; idx < 8; idx++) {
      cell_vertices[idx] = OpenVolumeMesh::VertexHandle(
          mat_hexahedrons(cell_idx, map_openmesh_idx_to_vtk_idx.at(idx)));
    }
    hexahedral_topomesh.add_cell(cell_vertices);
  }
  return hexahedral_topomesh;
}

auto CreateSurfaceMatrixFromTetrahedralTopoMesh(const TetrahedralTopoMesh &topomesh)
    -> Eigen::MatrixXi {
  Eigen::MatrixXi mat_surface_triangles;
  std::vector<OpenVolumeMesh::FaceHandle> boundary_faces;
  std::vector<OpenVolumeMesh::CellHandle> map_boundary_face_to_cell;
  for (auto boundary_face = topomesh.bf_iter(); boundary_face.is_valid(); ++boundary_face) {
    auto face = *boundary_face;
    boundary_faces.push_back(face);
  }
  const size_t num_surface_faces = boundary_faces.size();
  mat_surface_triangles.resize(num_surface_faces, 3);

  for (index_t face_idx = 0; face_idx < num_surface_faces; ++face_idx) {
    auto face = boundary_faces[face_idx];

    for (auto &&[idx, face_vertex] : topomesh.face_vertices(face) | boost::adaptors::indexed(0)) {
      mat_surface_triangles(face_idx, idx) = face_vertex.idx();
    }
  }
  return mat_surface_triangles;
}

auto CreateSurfaceMatFromHexahedralTopoMesh(const HexahedralTopoMesh &topomesh) -> Eigen::MatrixXi {
  Eigen::MatrixXi mat_surface_quadrangles;
  std::vector<OpenVolumeMesh::FaceHandle> boundary_faces;
  std::vector<OpenVolumeMesh::CellHandle> map_boundary_face_to_cell;
  for (auto boundary_face = topomesh.bf_iter(); boundary_face.is_valid(); ++boundary_face) {
    auto face = *boundary_face;
    boundary_faces.push_back(face);
  }
  const size_t num_surface_faces = boundary_faces.size();
  mat_surface_quadrangles.resize(num_surface_faces, 4);

  for (index_t face_idx = 0; face_idx < num_surface_faces; ++face_idx) {
    auto face = boundary_faces[face_idx];

    for (auto &&[idx, face_vertex] : topomesh.face_vertices(face) | boost::adaptors::indexed(0)) {
      mat_surface_quadrangles(face_idx, idx) = face_vertex.idx();
    }
  }
  return mat_surface_quadrangles;
}

auto CombineTwoTetrahedralMatMeshes(const TetrahedralMatMesh &matmesh_1,
                                    const TetrahedralMatMesh &matmesh_2) -> TetrahedralMatMesh {
  if (matmesh_1.NumVertices() == 0) return matmesh_2;
  if (matmesh_2.NumVertices() == 0) return matmesh_1;
  TetrahedralMatMesh result_mesh;
  igl::combine<Eigen::MatrixXd, Eigen::MatrixXi>(
      {matmesh_1.mat_coordinates, matmesh_2.mat_coordinates},
      {matmesh_1.mat_tetrahedrons, matmesh_2.mat_tetrahedrons}, result_mesh.mat_coordinates,
      result_mesh.mat_tetrahedrons);
  return result_mesh;
}

auto CombineTwoHexahedralMatMeshes(const HexahedralMatMesh &matmesh_1,
                                   const HexahedralMatMesh &matmesh_2) -> HexahedralMatMesh {
  if (matmesh_1.NumVertices() == 0) return matmesh_2;
  if (matmesh_2.NumVertices() == 0) return matmesh_1;
  HexahedralMatMesh result_mesh;
  igl::combine<Eigen::MatrixXd, Eigen::MatrixXi>(
      {matmesh_1.mat_coordinates, matmesh_2.mat_coordinates},
      {matmesh_1.mat_hexahedrons, matmesh_2.mat_hexahedrons}, result_mesh.mat_coordinates,
      result_mesh.mat_hexahedrons);
  return result_mesh;
}

}  // namespace sha
}  // namespace da
