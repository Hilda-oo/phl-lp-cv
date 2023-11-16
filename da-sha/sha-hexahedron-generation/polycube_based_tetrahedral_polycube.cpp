#include "polycube_based_utility.h"

#include <igl/per_face_normals.h>

#include <CGAL/polygon_mesh_processing.h>

#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-surface-mesh/mesh3.h"

#include <boost/range/adaptor/indexed.hpp>

namespace da {
namespace sha {
namespace tet_polycube {
void LaplacianSmoothForSurfacePatchBoundaryEdges(
    const SurfaceMesh3 &surface_mesh, const std::vector<VertexIndexEdge> &boundary_edges_soup,
    Eigen::MatrixXd &mat_coordinates, size_t num_iterations) {
  log::info("Tet polycube Smoothing Patch Boundary for #{}", num_iterations);
  auto boundary_chains = ConvertEdgesSoupToChains(boundary_edges_soup);

  std::vector<std::vector<double>> chain_edge_weights(boundary_chains.size());
  for (auto &&[chain_idx, boundary_chain] : boundary_chains | boost::adaptors::indexed(0)) {
    auto &edge_weights = chain_edge_weights[chain_idx];
    edge_weights.resize(boundary_chain.size() + 1, 0);
    for (int edge_idx = 0; edge_idx < boundary_chain.size(); ++edge_idx) {
      edge_weights[edge_idx + 1] =
          edge_weights[edge_idx] + (mat_coordinates.row(boundary_chain[edge_idx].first) -
                                    mat_coordinates.row(boundary_chain[edge_idx].second))
                                       .norm();
    }
  }

  for (index_t iteration_idx = 0; iteration_idx < num_iterations; ++iteration_idx) {
    for (auto &&[chain_idx, boundary_chain] : boundary_chains | boost::adaptors::indexed(0)) {
      const auto &edge_weights      = chain_edge_weights[chain_idx];
      Eigen::Vector3d corner_vector = mat_coordinates.row(boundary_chain.back().second) -
                                      mat_coordinates.row(boundary_chain.front().first);
      Eigen::Vector3d corner = mat_coordinates.row(boundary_chain.front().first);
      for (int edge_idx = 0; edge_idx < boundary_chain.size(); ++edge_idx) {
        const auto &edge = boundary_chain[edge_idx];
        mat_coordinates.row(edge.second) =
            corner + corner_vector * edge_weights[edge_idx + 1] / edge_weights.back();
      }
    }
  }
}

void LaplacianSmoothForSurfacePatches(const SurfaceMesh3 &surface_mesh,
                                      const std::vector<VertexIndexEdge> &boundary_edges_soup,
                                      Eigen::MatrixXd &mat_coordinates, size_t num_iterations) {
  log::info("Tet polycube Smoothing Surface for #{}", num_iterations);
  std::set<index_t> boundary_vertices;
  boost::for_each(boundary_edges_soup, [&](const VertexIndexEdge &edge) {
    boundary_vertices.insert(edge.first);
    boundary_vertices.insert(edge.second);
  });

  for (index_t iteration_idx = 0; iteration_idx < num_iterations; ++iteration_idx) {
    std::map<SurfaceMesh3::Vertex_index, bool> vertex_visited_flags;
    for (auto halfedge : surface_mesh.halfedges()) {
      auto vertex = surface_mesh.target(halfedge);
      if (vertex_visited_flags[vertex]) continue;
      vertex_visited_flags[vertex] = true;
      if (boundary_vertices.count(vertex.idx()) != 0) continue;
      Eigen::Vector3d new_coordinate = mat_coordinates.row(vertex.idx());
      size_t num_neighbor            = 0;
      for (auto neighbor_vtx : surface_mesh.vertices_around_target(halfedge)) {
        new_coordinate += mat_coordinates.row(neighbor_vtx.idx());
        num_neighbor++;
      }
      new_coordinate /= (num_neighbor + 1);
      mat_coordinates.row(vertex.idx()) = new_coordinate;
    }
  }
}

void LaplacianSmoothForVolume(const TetrahedralMesh &tetrahedral_mesh,
                              Eigen::MatrixXd &mat_coordinates, size_t num_iterations) {
  log::info("Tet polycube Smoothing Volume for #{}", num_iterations);
  for (index_t iteration_idx = 0; iteration_idx < num_iterations; ++iteration_idx) {
    for (auto vertex : tetrahedral_mesh.vertices()) {
      if (tetrahedral_mesh.is_boundary(vertex)) continue;
      Eigen::Vector3d new_coordinate = mat_coordinates.row(vertex.idx());
      size_t num_neighbor            = 0;
      for (auto neighbor_vtx : tetrahedral_mesh.vertex_vertices(vertex)) {
        new_coordinate += mat_coordinates.row(neighbor_vtx.idx());
        num_neighbor++;
      }
      new_coordinate /= (num_neighbor + 1);
      mat_coordinates.row(vertex.idx()) = new_coordinate;
    }
  }
}
}  // namespace tet_polycube

void PostProcessOptimalTetrahedronPolycube(const TetrahedralMatMesh &tetrahedral_matmesh,
                                           Eigen::MatrixXd &mat_tetrahedral_polycube_coordinates) {
  auto &mat_coordinates            = mat_tetrahedral_polycube_coordinates;
  TetrahedralMesh tetrahedral_mesh = sha::CreateTetrahedralMeshFromMatMesh(tetrahedral_matmesh);

  SurfaceMesh3 surface_mesh = sha::CreateSurfaceMesh3FromTetrahedralMesh(tetrahedral_mesh);
  MatMesh3 surface_matmesh  = sha::CreateMatMesh3FromSurfaceMesh3(surface_mesh);

  const size_t num_faces = surface_mesh.num_faces();

  std::vector<Orientation> surface_orientations =
      MarkMeshFaceOrientations(surface_matmesh.mat_coordinates, surface_matmesh.mat_faces);

  auto CleanUpSurfaceOfTetrahedronPolycube = [&](size_t num_relabeling_for_faces,
                                                 size_t num_relabeling_for_patches) {
    auto RelabelOritationsForFaces = [&](const SurfaceMesh3 &mesh3,
                                         std::vector<Orientation> &face_orientations) {
      for (auto face : mesh3.faces()) {
        std::array<int, 3> vote_for_orientation = {0, 0, 0};
        for (auto neighbor_face : mesh3.faces_around_face(mesh3.halfedge(face))) {
          index_t idx = GetNondirectionalIndexByOrientation(face_orientations[neighbor_face.idx()]);
          if ((++vote_for_orientation[idx]) >= 2) {
            face_orientations[face.idx()] = face_orientations[neighbor_face.idx()];
            break;
          }
        }  // end for iterating neighbors
      }    // end for every face
    };     // lambda RelabelOritations

    auto RelabelOritationsForPatches = [](const SurfaceMesh3 &mesh3,
                                          const Eigen::MatrixXd &mat_coordinates,
                                          std::vector<Orientation> &face_orientations) {
      std::vector<OrientedPatch> oriented_patches;
      std::vector<index_t> map_face_to_patch;
      std::vector<VertexIndexEdge> boundary_edges_soup;
      DivideMeshIntoPatchesByOrientation(mesh3, mat_coordinates, face_orientations,
                                         oriented_patches, map_face_to_patch, boundary_edges_soup);
      for (auto &&[patch_idx, oriented_patch] : oriented_patches | boost::adaptors::indexed(0)) {
        size_t num_neighbors = oriented_patch.neighbor_patches_with_boundary_length.size();
        if (num_neighbors == 1) {
          Orientation orientation = *oriented_patch.neighbor_orientations.begin();
          for (auto face : oriented_patch.face_indices) {
            face_orientations[face.idx()] = orientation;
          }
        } else if (num_neighbors == 2) {
          auto longest_boundary_neighbor =
              boost::max_element(oriented_patch.neighbor_patches_with_boundary_length,
                                 [&](const std::pair<index_t, double> &neighbor_patch_0,
                                     const std::pair<index_t, double> &neighbor_patch_1) {
                                   return neighbor_patch_0.second < neighbor_patch_1.second;
                                 });
          Orientation orientation = oriented_patches[longest_boundary_neighbor->first].orientation;
          for (auto face : oriented_patch.face_indices) {
            face_orientations[face.idx()] = orientation;
          }
        }
      }
    };

    for (index_t idx = 0; idx < num_relabeling_for_faces; ++idx) {
      RelabelOritationsForFaces(surface_mesh, surface_orientations);
    }
    for (index_t idx = 0; idx < num_relabeling_for_patches; ++idx) {
      RelabelOritationsForPatches(surface_mesh, mat_coordinates, surface_orientations);
    }
  };

  auto MoveCoordinatesByOrientation = [](const SurfaceTopoMesh3 &mesh3,
                                         const std::vector<Orientation> &surface_orientations,
                                         Eigen::MatrixXd &mat_coordinates) {
    std::vector<OrientedPatch> oriented_patches;
    std::vector<index_t> map_face_to_patch;
    std::vector<VertexIndexEdge> boundary_edges_soup;
    DivideMeshIntoPatchesByOrientation(mesh3, mat_coordinates, surface_orientations,
                                       oriented_patches, map_face_to_patch, boundary_edges_soup);
    for (auto &oriented_patch : oriented_patches) {
      index_t component_idx     = GetNondirectionalIndexByOrientation(oriented_patch.orientation);
      double component_position = 0;
      for (auto face : oriented_patch.face_indices) {
        auto [vertices_it, vertices_end] = mesh3.vertices_around_face(mesh3.halfedge(face));
        component_position += (mat_coordinates(vertices_it->idx(), component_idx) +
                               mat_coordinates((++vertices_it)->idx(), component_idx) +
                               mat_coordinates((++vertices_it)->idx(), component_idx)) /
                              3;
      }
      component_position /= oriented_patch.face_indices.size();
      for (auto face : oriented_patch.face_indices) {
        for (auto vertex : mesh3.vertices_around_face(mesh3.halfedge(face))) {
          mat_coordinates(vertex.idx(), component_idx) = component_position;
        }
      }
    }  // for every patch
  };   // lambda MoveCoordinatesByOrientation

  for (index_t idx = 0; idx < 10; ++idx) {
    CleanUpSurfaceOfTetrahedronPolycube(10, 10);
  }
  for (index_t idx = 0; idx < 10; ++idx) {
    MoveCoordinatesByOrientation(surface_mesh, surface_orientations, mat_coordinates);
  }

  std::vector<OrientedPatch> oriented_patches;
  std::vector<index_t> map_face_to_patch;
  std::vector<VertexIndexEdge> boundary_edges_soup;
  DivideMeshIntoPatchesByOrientation(surface_mesh, mat_coordinates, surface_orientations,
                                     oriented_patches, map_face_to_patch, boundary_edges_soup);
  log::info("Divided polycube into {} patches.", oriented_patches.size());

  tet_polycube::LaplacianSmoothForSurfacePatchBoundaryEdges(surface_mesh, boundary_edges_soup,
                                                            mat_coordinates, 10);
  tet_polycube::LaplacianSmoothForSurfacePatches(surface_mesh, boundary_edges_soup, mat_coordinates,
                                                 10);
  tet_polycube::LaplacianSmoothForVolume(tetrahedral_mesh, mat_coordinates, 10);

  std::vector<double> cell_data;
  boost::transform(map_face_to_patch, std::back_inserter(cell_data),
                   [](const index_t &value) -> double { return value; });
  surface_matmesh.mat_coordinates = mat_coordinates;
}
}  // namespace sha
}  // namespace da
