#include "polycube_based_utility.h"

#include <igl/barycentric_coordinates.h>
#include <igl/barycentric_interpolation.h>
#include <igl/in_element.h>
#include <igl/point_mesh_squared_distance.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>

#include <set>
#include <vector>

#include <boost/range/adaptors.hpp>

#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-surface-mesh/mesh3.h"
#include "sha-volume-mesh/matmesh.h"
#include "sha-volume-mesh/mesh.h"

namespace da {
namespace sha {
namespace hex {
void LaplacianSmoothForSurfacePatchBoundaryEdges(
    const std::vector<VertexIndexEdge> &triangular_boundary_edges_soup,
    const std::vector<VertexIndexEdge> &quadrilateral_boundary_edges_soup,
    const Eigen::MatrixXd &mat_hexahedral_polycube_coordinates,
    const Eigen::MatrixXd &mat_tetrahedral_polycube_coordinates,
    const Eigen::MatrixXd &mat_tetrahedral_coordinates, Eigen::MatrixXd &mat_hexahedral_coordinates,
    size_t num_iterations) {
  log::info("Hex Smoothing Patch Boundary for #{}", num_iterations);
  auto triangular_boundary_chains    = ConvertEdgesSoupToChains(triangular_boundary_edges_soup);
  auto quadrilateral_boundary_chains = ConvertEdgesSoupToChains(quadrilateral_boundary_edges_soup);
  if (triangular_boundary_chains.size() != quadrilateral_boundary_chains.size()) {
    Terminate(
        fmt::format("Size of triangular boundary chains #{} is not equal to quadrilateral #{}",
                    triangular_boundary_chains.size(), quadrilateral_boundary_chains.size()));
  }
  constexpr double kEpsilon = 1e-6;
  const size_t num_chains   = triangular_boundary_chains.size();
  for (index_t chain_idx = 0; chain_idx < num_chains; ++chain_idx) {
    const auto &triangular_chain = triangular_boundary_chains.at(chain_idx);
    bool reverse_flag            = false;
    auto quadrilateral_chain_it  = std::find_if(
        quadrilateral_boundary_chains.begin() + chain_idx, quadrilateral_boundary_chains.end(),
        [&](const Chain &chain) -> bool {
          if ((mat_tetrahedral_polycube_coordinates.row(triangular_chain.front().first) -
               mat_hexahedral_polycube_coordinates.row(chain.front().first))
                      .norm() < kEpsilon &&
              (mat_tetrahedral_polycube_coordinates.row(triangular_chain.back().second) -
               mat_hexahedral_polycube_coordinates.row(chain.back().second))
                      .norm() < kEpsilon) {
            return true;
          }
          if ((mat_tetrahedral_polycube_coordinates.row(triangular_chain.front().first) -
               mat_hexahedral_polycube_coordinates.row(chain.back().second))
                      .norm() < kEpsilon &&
              (mat_tetrahedral_polycube_coordinates.row(triangular_chain.back().second) -
               mat_hexahedral_polycube_coordinates.row(chain.front().first))
                      .norm() < kEpsilon) {
            reverse_flag = true;
            return true;
          }
          return false;
        });
    if (quadrilateral_chain_it == quadrilateral_boundary_chains.end()) {
      Terminate("Can not match quadrilateral chain and triangular chain");
    }
    if (reverse_flag) {
      boost::reverse(*quadrilateral_chain_it);
      for (auto &edge : *quadrilateral_chain_it) {
        std::swap(edge.first, edge.second);
      }
    }
    std::swap(*quadrilateral_chain_it, *(quadrilateral_boundary_chains.begin() + chain_idx));
  }

  for (index_t iteration_idx = 0; iteration_idx < num_iterations; ++iteration_idx) {
    for (index_t chain_idx = 0; chain_idx < num_chains; ++chain_idx) {
      const auto &triangular_chain    = triangular_boundary_chains.at(chain_idx);
      const auto &quadrilateral_chain = quadrilateral_boundary_chains.at(chain_idx);
      if (quadrilateral_chain.size() == 1) continue;

      Eigen::MatrixXi mat_triangular_chain(triangular_chain.size(), 2);
      for (auto &&[edge_idx, edge] : triangular_chain | boost::adaptors::indexed(0)) {
        mat_triangular_chain(edge_idx, 0) = edge.first;
        mat_triangular_chain(edge_idx, 1) = edge.second;
      }
      Eigen::MatrixXd mat_quadrilateral_chain_points(quadrilateral_chain.size() - 1, 3);
      for (index_t edge_idx = 0; edge_idx < quadrilateral_chain.size() - 1; ++edge_idx) {
        Eigen::Vector3d coordinate =
            (mat_hexahedral_coordinates.row(quadrilateral_chain[edge_idx].first) +
             mat_hexahedral_coordinates.row(quadrilateral_chain[edge_idx].second) +
             mat_hexahedral_coordinates.row(quadrilateral_chain[edge_idx + 1].second)) /
            3.0;
        mat_quadrilateral_chain_points.row(edge_idx) = coordinate;
      }
      Eigen::VectorXd chain_point_min_distances;
      Eigen::VectorXi chain_point_closet_edges;
      Eigen::MatrixXd chain_point_closet_point;
      igl::point_mesh_squared_distance(mat_quadrilateral_chain_points, mat_tetrahedral_coordinates,
                                       mat_triangular_chain, chain_point_min_distances,
                                       chain_point_closet_edges, chain_point_closet_point);
      for (index_t edge_idx = 0; edge_idx < quadrilateral_chain.size() - 1; ++edge_idx) {
        mat_hexahedral_coordinates.row(quadrilateral_chain[edge_idx].second) =
            chain_point_closet_point.row(edge_idx);
      }
      mat_hexahedral_coordinates.row(quadrilateral_chain.front().first) =
          mat_tetrahedral_coordinates.row(triangular_chain.front().first);
      mat_hexahedral_coordinates.row(quadrilateral_chain.back().second) =
          mat_tetrahedral_coordinates.row(triangular_chain.back().second);
    }
  }
}

void LaplacianSmoothForSurfacePatches(
    const SurfaceTopoMesh3 &quadrilateral_topomesh, const Eigen::MatrixXi &mat_surface_triangles,
    const std::vector<OrientedPatch> &triangular_oriented_patches,
    const std::vector<OrientedPatch> &quadrilateral_oriented_patches,
    const std::vector<std::pair<index_t, index_t>> &matching_patch_pairs,
    const Eigen::MatrixXd &mat_tetrahedral_coordinates, Eigen::MatrixXd &mat_hexahedral_coordinates,
    size_t num_iterations) {
  log::info("Hex Smoothing Surface for #{}", num_iterations);
  size_t num_patches = matching_patch_pairs.size();
  std::vector<Eigen::MatrixXi> triangular_patch_triangles(num_patches);
  std::vector<SurfaceTopoMesh3> quadrilateral_patch_topomesh(num_patches);
  std::vector<std::set<SurfaceTopoMesh3::Vertex_index>> quadrilateral_patch_vertices(num_patches);

  for (index_t pair_idx = 0; pair_idx < num_patches; ++pair_idx) {
    index_t quadrilateral_patch_idx = matching_patch_pairs.at(pair_idx).second;
    auto &topomesh                  = quadrilateral_patch_topomesh.at(quadrilateral_patch_idx);
    auto &quadrilateral_patch       = quadrilateral_oriented_patches.at(quadrilateral_patch_idx);
    auto &quadrilateral_vertices    = quadrilateral_patch_vertices.at(quadrilateral_patch_idx);

    std::set<SurfaceTopoMesh3::Vertex_index> vertices;
    for (auto vertex : quadrilateral_topomesh.vertices()) {
      topomesh.add_vertex();
    }
    for (auto face_idx : quadrilateral_patch.face_indices) {
      auto vertices_range =
          quadrilateral_topomesh.vertices_around_face(quadrilateral_topomesh.halfedge(face_idx));
      topomesh.add_face(vertices_range);
      for (auto vtx : vertices_range) {
        vertices.insert(vtx);
      }
    }
    for (auto vertex : vertices) {
      if (!topomesh.is_border(vertex)) {
        quadrilateral_vertices.insert(vertex);
      }
    }
  }

  for (index_t pair_idx = 0; pair_idx < num_patches; ++pair_idx) {
    index_t triangular_patch_idx = matching_patch_pairs.at(pair_idx).first;
    auto &mat_patch_triangles    = triangular_patch_triangles.at(triangular_patch_idx);
    auto &triangular_patch       = triangular_oriented_patches.at(triangular_patch_idx);
    mat_patch_triangles.resize(triangular_patch.face_indices.size(), 3);
    for (auto &&[idx, face_idx] : triangular_patch.face_indices | boost::adaptors::indexed(0)) {
      mat_patch_triangles.row(idx) = mat_surface_triangles.row(face_idx.idx());
    }
  }

  for (index_t iteration_idx = 0; iteration_idx < num_iterations; ++iteration_idx) {
    for (auto [triangular_patch_idx, quadrilateral_patch_idx] : matching_patch_pairs) {
      auto &topomesh               = quadrilateral_patch_topomesh.at(quadrilateral_patch_idx);
      auto &triangular_patch       = triangular_oriented_patches.at(triangular_patch_idx);
      auto &quadrilateral_patch    = quadrilateral_oriented_patches.at(quadrilateral_patch_idx);
      auto &mat_patch_triangles    = triangular_patch_triangles.at(triangular_patch_idx);
      auto &quadrilateral_vertices = quadrilateral_patch_vertices.at(quadrilateral_patch_idx);

      Eigen::MatrixXd mat_coordinates(quadrilateral_vertices.size(), 3);
      for (auto &&[idx, vertex] : quadrilateral_vertices | boost::adaptors::indexed(0)) {
        Eigen::Vector3d coordinate   = mat_hexahedral_coordinates.row(vertex.idx());
        size_t num_one_ring_vertices = 0;
        for (auto neighbor : topomesh.vertices_around_target(topomesh.halfedge(vertex))) {
          coordinate += mat_hexahedral_coordinates.row(neighbor.idx());
          num_one_ring_vertices++;
        }
        coordinate /= (1 + num_one_ring_vertices);
        mat_coordinates.row(idx) = coordinate;
      }
      Eigen::VectorXd patch_point_min_distances;
      Eigen::VectorXi patch_point_closet_triangles;
      Eigen::MatrixXd patch_point_closet_points;

      igl::point_mesh_squared_distance(mat_coordinates, mat_tetrahedral_coordinates,
                                       mat_patch_triangles, patch_point_min_distances,
                                       patch_point_closet_triangles, patch_point_closet_points);
      for (auto &&[idx, vertex] : quadrilateral_vertices | boost::adaptors::indexed(0)) {
        mat_hexahedral_coordinates.row(vertex.idx()) = patch_point_closet_points.row(idx);
      }
    }  // pair of patches
  }    // for iteration
}

void LaplacianSmoothForVolume(const HexahedralTopoMesh &hexahedral_topomesh,
                              Eigen::MatrixXd &mat_coordinates, size_t num_iterations) {
  log::info("Hex Smoothing Volume for #{}", num_iterations);
  for (index_t iteration_idx = 0; iteration_idx < num_iterations; ++iteration_idx) {
    for (auto vertex : hexahedral_topomesh.vertices()) {
      if (hexahedral_topomesh.is_boundary(vertex)) continue;
      Eigen::Vector3d new_coordinate = mat_coordinates.row(vertex.idx());
      size_t num_neighbor            = 0;
      for (auto neighbor_vtx : hexahedral_topomesh.vertex_vertices(vertex)) {
        new_coordinate += mat_coordinates.row(neighbor_vtx.idx());
        num_neighbor++;
      }
      new_coordinate /= (num_neighbor + 1);
      mat_coordinates.row(vertex.idx()) = new_coordinate;
    }
  }
}
}  // namespace hex

auto DeformHexadralPolycubeMeshToOriginalDomain(
    const TetrahedralMatMesh &tetrahedral_matmesh,
    const Eigen::MatrixXd &mat_tetrahedral_polycube_coordinates,
    const HexahedralMatMesh &hexahedral_polycube_matmesh) -> Eigen::MatrixXd {
  const size_t num_hexahedral_vertices       = hexahedral_polycube_matmesh.NumVertices();
  const size_t num_tetrahedral_vertices      = tetrahedral_matmesh.NumVertices();
  Eigen::MatrixXd mat_hexahedral_coordinates = hexahedral_polycube_matmesh.mat_coordinates;

  TetrahedralTopoMesh tetrahedral_topomesh = sha::CreateTetrahedralTopoMeshFromMatrix(
      num_tetrahedral_vertices, tetrahedral_matmesh.mat_tetrahedrons);

  SurfaceTopoMesh3 triangle_topomesh =
      sha::CreateSurfaceTopoMesh3FromTetrahedralTopoMesh(tetrahedral_topomesh);

  Eigen::MatrixXi mat_surface_triangles =
      sha::CreateFaceMatrixFromSurfaceTopoMesh3(triangle_topomesh);

  HexahedralTopoMesh hexahedral_topomesh = sha::CreateHexahedralTopoMeshFromMatrix(
      num_hexahedral_vertices, hexahedral_polycube_matmesh.mat_hexahedrons);

  SurfaceTopoMesh3 quadrilateral_topomesh =
      sha::CreateSurfaceTopoMesh3FromHexahedralTopoMesh(hexahedral_topomesh);

  Eigen::VectorXd quadrilateral_vtx_min_distances_to_triangles;
  Eigen::VectorXi quadrilateral_vtx_cloest_triangle_indices;
  Eigen::MatrixXd quadrilateral_vtx_cloest_points_to_triangles;
  Eigen::MatrixXd quadrilateral_vtx_barycentric_coord_to_triangles;

  igl::point_mesh_squared_distance(
      hexahedral_polycube_matmesh.mat_coordinates, mat_tetrahedral_polycube_coordinates,
      mat_surface_triangles, quadrilateral_vtx_min_distances_to_triangles,
      quadrilateral_vtx_cloest_triangle_indices, quadrilateral_vtx_cloest_points_to_triangles);

  igl::barycentric_coordinates(
      hexahedral_polycube_matmesh.mat_coordinates,
      mat_tetrahedral_polycube_coordinates(
          mat_surface_triangles(quadrilateral_vtx_cloest_triangle_indices, 0), Eigen::all),
      mat_tetrahedral_polycube_coordinates(
          mat_surface_triangles(quadrilateral_vtx_cloest_triangle_indices, 1), Eigen::all),
      mat_tetrahedral_polycube_coordinates(
          mat_surface_triangles(quadrilateral_vtx_cloest_triangle_indices, 2), Eigen::all),
      quadrilateral_vtx_barycentric_coord_to_triangles);

  auto ComputeBarycentricCoordinateInTetrahedron =
      [&](const Eigen::Vector3d &vertex_a, const Eigen::Vector3d &vertex_b,
          const Eigen::Vector3d &vertex_c, const Eigen::Vector3d &vertex_d,
          const Eigen::Vector3d &point) -> Eigen::Vector4d {
    auto ComputeTripleProduct = [](const Eigen::Vector3d &a, const Eigen::Vector3d &b,
                                   const Eigen::Vector3d &c) { return b.cross(c).dot(a); };

    Eigen::Vector3d vec_ap = point - vertex_a;
    Eigen::Vector3d vec_bp = point - vertex_b;

    Eigen::Vector3d vec_ab = vertex_b - vertex_a;
    Eigen::Vector3d vec_ac = vertex_c - vertex_a;
    Eigen::Vector3d vec_ad = vertex_d - vertex_a;

    Eigen::Vector3d vec_bc = vertex_c - vertex_b;
    Eigen::Vector3d vec_bd = vertex_d - vertex_b;

    double va6 = ComputeTripleProduct(vec_bp, vec_bd, vec_bc);
    double vb6 = ComputeTripleProduct(vec_ap, vec_ac, vec_ad);
    double vc6 = ComputeTripleProduct(vec_ap, vec_ad, vec_ab);
    double vd6 = ComputeTripleProduct(vec_ap, vec_ab, vec_ac);
    double v6  = 1 / ComputeTripleProduct(vec_ab, vec_ac, vec_ad);
    return Eigen::Vector4d(va6 * v6, vb6 * v6, vc6 * v6, vd6 * v6);
  };
  Eigen::VectorXi hexahedral_vtx_cloest_tetrahedron_indices;
  igl::AABB<Eigen::MatrixXd, 3> tetrahedral_polycube_tree;
  tetrahedral_polycube_tree.init(mat_tetrahedral_polycube_coordinates,
                                 tetrahedral_matmesh.mat_tetrahedrons);
  igl::in_element(mat_tetrahedral_polycube_coordinates, tetrahedral_matmesh.mat_tetrahedrons,
                  hexahedral_polycube_matmesh.mat_coordinates, tetrahedral_polycube_tree,
                  hexahedral_vtx_cloest_tetrahedron_indices);

  for (auto vertex_idx = 0; vertex_idx < num_hexahedral_vertices; ++vertex_idx) {
    auto &mat_original_coords = tetrahedral_matmesh.mat_coordinates;
    if (hexahedral_topomesh.is_boundary(OpenVolumeMesh::VertexHandle(vertex_idx))) {
      index_t triangle_idx             = quadrilateral_vtx_cloest_triangle_indices(vertex_idx);
      Eigen::Vector3i triangle_indices = mat_surface_triangles.row(triangle_idx);
      Eigen::Vector3d barycentric_coord =
          quadrilateral_vtx_barycentric_coord_to_triangles.row(vertex_idx);
      Eigen::Vector3d restore_coord =
          mat_original_coords.row(triangle_indices(0)) * barycentric_coord.x() +
          mat_original_coords.row(triangle_indices(1)) * barycentric_coord.y() +
          mat_original_coords.row(triangle_indices(2)) * barycentric_coord.z();
      mat_hexahedral_coordinates.row(vertex_idx) = restore_coord;
    } else {
      index_t tetrahedron_idx = hexahedral_vtx_cloest_tetrahedron_indices(vertex_idx);
      if (tetrahedron_idx == index_t(-1)) {
        Terminate(
            fmt::format("Internal vertex #{} of hexahedral polycube can not match any tetrahedron.",
                        vertex_idx));
      }
      Eigen::RowVector4i tetrahedron_vertices =
          tetrahedral_matmesh.mat_tetrahedrons.row(tetrahedron_idx);
      Eigen::Vector4d barycentric_coord = ComputeBarycentricCoordinateInTetrahedron(
          mat_tetrahedral_polycube_coordinates.row(tetrahedron_vertices(0)),
          mat_tetrahedral_polycube_coordinates.row(tetrahedron_vertices(1)),
          mat_tetrahedral_polycube_coordinates.row(tetrahedron_vertices(2)),
          mat_tetrahedral_polycube_coordinates.row(tetrahedron_vertices(3)),
          hexahedral_polycube_matmesh.mat_coordinates.row(vertex_idx));
      mat_hexahedral_coordinates.row(vertex_idx) =
          mat_original_coords.row(tetrahedron_vertices(0)) * barycentric_coord(0) +
          mat_original_coords.row(tetrahedron_vertices(1)) * barycentric_coord(1) +
          mat_original_coords.row(tetrahedron_vertices(2)) * barycentric_coord(2) +
          mat_original_coords.row(tetrahedron_vertices(3)) * barycentric_coord(3);
    }
  }

  std::vector<Orientation> triangular_orientations =
      MarkMeshFaceOrientations(mat_tetrahedral_polycube_coordinates, mat_surface_triangles);
  std::vector<OrientedPatch> triangular_oriented_patches;
  std::vector<index_t> map_triangle_to_patch;
  std::vector<VertexIndexEdge> triangular_boundary_edges_soup;
  std::vector<double> triangular_patch_values;
  DivideMeshIntoPatchesByOrientation(triangle_topomesh, mat_tetrahedral_polycube_coordinates,
                                     triangular_orientations, triangular_oriented_patches,
                                     map_triangle_to_patch, triangular_boundary_edges_soup);
  log::info("Divided tet polycube into {} patches.", triangular_oriented_patches.size());

  Eigen::MatrixXi mat_quadrangles =
      sha::CreateFaceMatrixFromSurfaceTopoMesh3(quadrilateral_topomesh, 4);
  std::vector<Orientation> quadrilateral_orientations = MarkMeshFaceOrientations(
      hexahedral_polycube_matmesh.mat_coordinates, mat_quadrangles.leftCols(3));

  std::vector<OrientedPatch> quadrilateral_oriented_patches;
  std::vector<index_t> map_quadrangle_to_patch;
  std::vector<VertexIndexEdge> quadrilateral_boundary_edges_soup;
  std::vector<double> quadrilateral_patch_values;
  DivideMeshIntoPatchesByOrientation(quadrilateral_topomesh, mat_hexahedral_coordinates,
                                     quadrilateral_orientations, quadrilateral_oriented_patches,
                                     map_quadrangle_to_patch, quadrilateral_boundary_edges_soup);
  log::info("Divided hex polycube into {} patches.", quadrilateral_oriented_patches.size());

  for (auto &oriented_patch : triangular_oriented_patches) {
    index_t component_idx     = GetNondirectionalIndexByOrientation(oriented_patch.orientation);
    double component_position = 0;
    for (auto face : oriented_patch.face_indices) {
      auto [vertices_it, vertices_end] =
          triangle_topomesh.vertices_around_face(triangle_topomesh.halfedge(face));
      component_position +=
          (mat_tetrahedral_polycube_coordinates(vertices_it->idx(), component_idx) +
           mat_tetrahedral_polycube_coordinates((++vertices_it)->idx(), component_idx) +
           mat_tetrahedral_polycube_coordinates((++vertices_it)->idx(), component_idx)) /
          3;
    }
    component_position /= oriented_patch.face_indices.size();
    triangular_patch_values.push_back(component_position);
  }

  for (auto &oriented_patch : quadrilateral_oriented_patches) {
    index_t component_idx     = GetNondirectionalIndexByOrientation(oriented_patch.orientation);
    double component_position = 0;
    for (auto face : oriented_patch.face_indices) {
      auto [vertices_it, vertices_end] =
          quadrilateral_topomesh.vertices_around_face(quadrilateral_topomesh.halfedge(face));
      component_position +=
          (hexahedral_polycube_matmesh.mat_coordinates(vertices_it->idx(), component_idx) +
           hexahedral_polycube_matmesh.mat_coordinates((++vertices_it)->idx(), component_idx) +
           hexahedral_polycube_matmesh.mat_coordinates((++vertices_it)->idx(), component_idx) +
           hexahedral_polycube_matmesh.mat_coordinates((++vertices_it)->idx(), component_idx)) /
          4;
    }
    component_position /= oriented_patch.face_indices.size();
    quadrilateral_patch_values.push_back(component_position);
  }

  size_t num_patches = quadrilateral_oriented_patches.size();
  if (num_patches != triangular_oriented_patches.size()) {
    Terminate("Number of triangular patches is not same as one of quadrilateral patches");
  }

  // match triangular_oriented_patches and quadrilateral_oriented_patches
  std::vector<index_t> triangular_indices, quadrilateral_indices;
  std::vector<std::pair<index_t, index_t>> matching_patch_pairs;
  boost::copy(boost::counting_range<index_t>(0, num_patches),
              std::back_inserter(triangular_indices));
  boost::copy(boost::counting_range<index_t>(0, num_patches),
              std::back_inserter(quadrilateral_indices));
  boost::sort(triangular_indices, [&](index_t patch_0_idx, index_t patch_1_idx) {
    const auto &patch_0             = triangular_oriented_patches.at(patch_0_idx);
    const auto &patch_1             = triangular_oriented_patches.at(patch_1_idx);
    const index_t orientation_0_idx = GetNondirectionalIndexByOrientation(patch_0.orientation);
    const index_t orientation_1_idx = GetNondirectionalIndexByOrientation(patch_1.orientation);

    if (orientation_0_idx != orientation_1_idx) {
      return orientation_0_idx < orientation_1_idx;
    } else {
      return triangular_patch_values.at(patch_0_idx) < triangular_patch_values.at(patch_1_idx);
    }
  });
  boost::sort(quadrilateral_indices, [&](index_t patch_0_idx, index_t patch_1_idx) {
    const auto &patch_0             = quadrilateral_oriented_patches.at(patch_0_idx);
    const auto &patch_1             = quadrilateral_oriented_patches.at(patch_1_idx);
    const index_t orientation_0_idx = GetNondirectionalIndexByOrientation(patch_0.orientation);
    const index_t orientation_1_idx = GetNondirectionalIndexByOrientation(patch_1.orientation);

    if (orientation_0_idx != orientation_1_idx) {
      return orientation_0_idx < orientation_1_idx;
    } else {
      return quadrilateral_patch_values.at(patch_0_idx) <
             quadrilateral_patch_values.at(patch_1_idx);
    }
  });

  for (index_t patch_idx = 0; patch_idx < num_patches; ++patch_idx) {
    constexpr double kEpsilon             = 1e-6;
    const index_t triangular_patch_idx    = triangular_indices.at(patch_idx);
    const index_t quadrilateral_patch_idx = quadrilateral_indices.at(patch_idx);
    const auto &triangular_patch          = triangular_oriented_patches.at(triangular_patch_idx);
    const auto &quadrilateral_patch = quadrilateral_oriented_patches.at(quadrilateral_patch_idx);

    if (GetNondirectionalIndexByOrientation(triangular_patch.orientation) !=
            GetNondirectionalIndexByOrientation(quadrilateral_patch.orientation) ||
        std::abs(quadrilateral_patch_values.at(quadrilateral_patch_idx) -
                 triangular_patch_values.at(triangular_patch_idx)) > kEpsilon) {
      Terminate(
          fmt::format("Triangular patch #{} and quadrilateral patch #{} don't match.\n"
                      "Triangular patch: Orientation #{} P: #{}.\n"
                      "Quadrilateral patch: Orientation #{} P: #{}.\n",
                      triangular_patch_idx, quadrilateral_patch_idx,
                      GetNondirectionalIndexByOrientation(triangular_patch.orientation),
                      triangular_patch_values.at(triangular_patch_idx),
                      GetNondirectionalIndexByOrientation(quadrilateral_patch.orientation),
                      quadrilateral_patch_values.at(quadrilateral_patch_idx)));
    }
    matching_patch_pairs.push_back({triangular_patch_idx, quadrilateral_patch_idx});
  }

  hex::LaplacianSmoothForSurfacePatchBoundaryEdges(
      triangular_boundary_edges_soup, quadrilateral_boundary_edges_soup,
      hexahedral_polycube_matmesh.mat_coordinates, mat_tetrahedral_polycube_coordinates,
      tetrahedral_matmesh.mat_coordinates, mat_hexahedral_coordinates, 10);

  hex::LaplacianSmoothForSurfacePatches(quadrilateral_topomesh, mat_surface_triangles,
                                        triangular_oriented_patches, quadrilateral_oriented_patches,
                                        matching_patch_pairs, tetrahedral_matmesh.mat_coordinates,
                                        mat_hexahedral_coordinates, 10);
  hex::LaplacianSmoothForVolume(hexahedral_topomesh, mat_hexahedral_coordinates, 10);
  return mat_hexahedral_coordinates;
}
}  // namespace sha
}  // namespace da
