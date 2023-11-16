#include "mesh.h"

namespace da {
namespace sha {
SurfaceMesh3 CreateSurfaceMesh3FromTetrahedralMesh(const TetrahedralMesh &tetrahedral_mesh) {
  SurfaceMesh3 mesh3;
  for (auto vertex : tetrahedral_mesh.vertices()) {
    auto &&ovm_point = tetrahedral_mesh.vertex(vertex);
    mesh3.add_vertex(Point(ovm_point[0], ovm_point[1], ovm_point[2]));
  }
  for (auto boundary_face = tetrahedral_mesh.bf_iter(); boundary_face.is_valid(); ++boundary_face) {
    auto face                                    = *boundary_face;
    auto &&[face_vertices_it, face_vertices_end] = tetrahedral_mesh.face_vertices(face);
    mesh3.add_face(CGAL::SM_Vertex_index((face_vertices_it++)->idx()),
                   CGAL::SM_Vertex_index((face_vertices_it++)->idx()),
                   CGAL::SM_Vertex_index((face_vertices_it++)->idx()));
  }
  return mesh3;
}

SurfaceMesh3 CreateSurfaceMesh3FromHexahedralMesh(const HexahedralMesh &hexahedral_mesh) {
  SurfaceMesh3 mesh3;
  for (auto vertex : hexahedral_mesh.vertices()) {
    auto &&ovm_point = hexahedral_mesh.vertex(vertex);
    mesh3.add_vertex(Point(ovm_point[0], ovm_point[1], ovm_point[2]));
  }
  for (auto boundary_face = hexahedral_mesh.bf_iter(); boundary_face.is_valid(); ++boundary_face) {
    auto face                                    = *boundary_face;
    auto &&[face_vertices_it, face_vertices_end] = hexahedral_mesh.face_vertices(face);
    mesh3.add_face(CGAL::SM_Vertex_index((face_vertices_it)->idx()),
                   CGAL::SM_Vertex_index((++face_vertices_it)->idx()),
                   CGAL::SM_Vertex_index((++face_vertices_it)->idx()),
                   CGAL::SM_Vertex_index((++face_vertices_it)->idx()));
  }

  return mesh3;
}

SurfaceTopoMesh3 CreateSurfaceTopoMesh3FromTetrahedralTopoMesh(
    const TetrahedralTopoMesh &tetrahedral_topomesh) {
  SurfaceTopoMesh3 topomesh3;
  for (auto vertex : tetrahedral_topomesh.vertices()) {
    topomesh3.add_vertex();
  }
  for (auto boundary_face = tetrahedral_topomesh.bf_iter(); boundary_face.is_valid();
       ++boundary_face) {
    auto face                                    = *boundary_face;
    auto &&[face_vertices_it, face_vertices_end] = tetrahedral_topomesh.face_vertices(face);
    topomesh3.add_face(CGAL::SM_Vertex_index((face_vertices_it++)->idx()),
                       CGAL::SM_Vertex_index((face_vertices_it++)->idx()),
                       CGAL::SM_Vertex_index((face_vertices_it++)->idx()));
  }
  return topomesh3;
}

SurfaceTopoMesh3 CreateSurfaceTopoMesh3FromHexahedralTopoMesh(
    const HexahedralTopoMesh &hexahedral_topomesh) {
  SurfaceTopoMesh3 topomesh3;
  for (auto vertex : hexahedral_topomesh.vertices()) {
    topomesh3.add_vertex();
  }
  for (auto boundary_face = hexahedral_topomesh.bf_iter(); boundary_face.is_valid();
       ++boundary_face) {
    auto face                                    = *boundary_face;
    auto &&[face_vertices_it, face_vertices_end] = hexahedral_topomesh.face_vertices(face);
    topomesh3.add_face(CGAL::SM_Vertex_index((face_vertices_it)->idx()),
                       CGAL::SM_Vertex_index((++face_vertices_it)->idx()),
                       CGAL::SM_Vertex_index((++face_vertices_it)->idx()),
                       CGAL::SM_Vertex_index((++face_vertices_it)->idx()));
  }
  return topomesh3;
}
}  // namespace sha
}  // namespace da
