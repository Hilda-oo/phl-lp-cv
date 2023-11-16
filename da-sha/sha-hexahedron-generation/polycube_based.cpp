#include "polycube_based.h"

#include <igl/adjacency_list.h>
#include <igl/boundary_facets.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/edges.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/volume.h>
#include <igl/writeOBJ.h>

#include <algorithm>

#include <boost/range/adaptor/indexed.hpp>

#include "cpt-l1-norm-polycube/polycube.h"
#include "polycube_based_utility.h"

namespace da {
namespace sha {
auto GeneratePolycubeForTetrahedralMesh(const TetrahedralMatMesh &tetrahedral_matmesh,
                                        const double alpha, const double beta,
                                        const size_t num_iterations) -> Eigen::MatrixXd {
  Eigen::MatrixXi mat_surface_faces;
  Eigen::VectorXi map_surface_to_cell, face_across_vertices;
  Eigen::MatrixXi mat_face_edges_of_tetrahedral_mesh;

  igl::boundary_facets(tetrahedral_matmesh.mat_tetrahedrons, mat_surface_faces, map_surface_to_cell,
                       face_across_vertices);

  log::info("tetrahedron: V: {}, T: {}, F: {}", tetrahedral_matmesh.NumVertices(),
            tetrahedral_matmesh.NumTetrahedrons(), mat_surface_faces.rows());
  igl::edges(mat_surface_faces, mat_face_edges_of_tetrahedral_mesh);

  auto matrices_gradient_operator = ComputeGradientOperatorsOfTetrahedronMesh(tetrahedral_matmesh);

  Eigen::VectorXd tetrahedron_volumes;
  igl::volume(tetrahedral_matmesh.mat_coordinates, tetrahedral_matmesh.mat_tetrahedrons,
              tetrahedron_volumes);

  SurfaceMesh3 mesh3 = sha::CreateSurfaceMesh3FromMatMesh3(MatMesh3{
      .mat_coordinates = tetrahedral_matmesh.mat_coordinates, .mat_faces = mat_surface_faces});

  Eigen::MatrixXi mat_adjacent_face_pairs(mesh3.num_edges(), 2);
  std::vector<std::vector<int>> map_face_idx_to_neighbor_indices(mesh3.num_faces());

  for (auto edge : mesh3.edges()) {
    auto face_0 = mesh3.face(mesh3.halfedge(edge));
    auto face_1 = mesh3.face(mesh3.opposite(mesh3.halfedge(edge)));
    mat_adjacent_face_pairs.row(edge.idx()) << face_0.idx(), face_1.idx();
  }

  for (auto face : mesh3.faces()) {
    for (auto adjacent_face : mesh3.faces_around_face(mesh3.halfedge(face))) {
      map_face_idx_to_neighbor_indices[face.idx()].push_back(adjacent_face.idx());
    }
  }

  auto mat_tetrahedral_polycube_coordinates = cpt::SolveL1NormBasedPolycubeProblemByIfopt(
      tetrahedral_matmesh.mat_coordinates, tetrahedral_matmesh.mat_tetrahedrons, mat_surface_faces,
      mat_adjacent_face_pairs, map_face_idx_to_neighbor_indices, matrices_gradient_operator,
      tetrahedron_volumes, alpha, beta, num_iterations);

  PostProcessOptimalTetrahedronPolycube(tetrahedral_matmesh, mat_tetrahedral_polycube_coordinates);

  return mat_tetrahedral_polycube_coordinates;
}

auto GenerateHexahedralMeshByPolycube(const TetrahedralMatMesh &tetrahedral_matmesh,
                                      const Eigen::MatrixXd &mat_tetrahedral_polycube_coordinates,
                                      const Eigen::Vector3i &num_cells) -> HexahedralMatMesh {
  Eigen::MatrixXd mat_polycube_coordinates;
  return GenerateHexahedralMeshByPolycube(tetrahedral_matmesh, mat_tetrahedral_polycube_coordinates,
                                          num_cells, mat_polycube_coordinates);
}

auto GenerateHexahedralMeshByPolycube(const TetrahedralMatMesh &tetrahedral_matmesh,
                                      const Eigen::MatrixXd &mat_tetrahedral_polycube_coordinates,
                                      const Eigen::Vector3i &num_cells,
                                      Eigen::MatrixXd &mat_polycube_coordinates)
    -> HexahedralMatMesh {
  const Eigen::AlignedBox3d aligned_box(mat_tetrahedral_polycube_coordinates.colwise().minCoeff(),
                                        mat_tetrahedral_polycube_coordinates.colwise().maxCoeff());
  Eigen::Vector3d scale_factors(1, 1, 1);
  scale_factors = aligned_box.sizes().cwiseInverse().cwiseProduct(num_cells.cast<double>());
  HexahedralMatMesh hexahedral_polycube_matmesh = RemeshTetrahedronPolycubeToHexhahedron(
      tetrahedral_matmesh, mat_tetrahedral_polycube_coordinates, scale_factors);
  auto mat_hexahedral_coordinates = DeformHexadralPolycubeMeshToOriginalDomain(
      tetrahedral_matmesh, mat_tetrahedral_polycube_coordinates, hexahedral_polycube_matmesh);
  HexahedralMatMesh hexahedral_matmesh{
      .mat_coordinates = mat_hexahedral_coordinates,
      .mat_hexahedrons = hexahedral_polycube_matmesh.mat_hexahedrons};
  mat_polycube_coordinates = hexahedral_polycube_matmesh.mat_coordinates;
  return hexahedral_matmesh;
}
}  // namespace sha
}  // namespace da
