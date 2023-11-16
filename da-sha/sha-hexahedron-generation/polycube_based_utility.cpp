#include "polycube_based_utility.h"

#include <igl/adjacency_list.h>
#include <igl/centroid.h>
#include <igl/face_areas.h>
#include <igl/per_face_normals.h>

#include <CGAL/polygon_mesh_processing.h>

#include <algorithm>
#include <map>
#include <utility>

#include "sha-base-framework/declarations.h"
#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-surface-mesh/mesh3.h"
#include "sha-volume-mesh/matmesh.h"
#include "sha-volume-mesh/mesh.h"

#include <boost/range/adaptor/indexed.hpp>

namespace da {
namespace sha {
using Matrix43d = Eigen::Matrix<double, 4, 3>;
using Matrix34d = Eigen::Matrix<double, 3, 4>;

namespace PMP = CGAL::Polygon_mesh_processing;

double approximate(const double x, const double eps) {
  return std::abs(x - std::round(x)) < eps ? (std::round(x) == -0.0 ? 0 : std::round(x)) : x;
}

auto GetNondirectionalIndexByOrientation(Orientation orientation) -> index_t {
  index_t idx = -1;
  switch (orientation) {
    case Orientation::X:
    case Orientation::NegX:
      idx = 0;
      break;
    case Orientation::Y:
    case Orientation::NegY:
      idx = 1;
      break;
    case Orientation::Z:
    case Orientation::NegZ:
      idx = 2;
      break;
  }
  return idx;
};

auto ComputeGradientOperatorsOfTetrahedronMesh(const TetrahedralMatMesh &tetrahedral_matmesh)
    -> std::vector<Matrix34d> {
  std::vector<Matrix34d> gradient_matices(tetrahedral_matmesh.NumTetrahedrons());

  Matrix43d mat_indices;
  mat_indices << 1, 0, 0,  //
      0, 1, 0,             //
      0, 0, 1,             //
      -1, -1, -1;
  for (index_t tetrahedron_idx = 0; tetrahedron_idx < tetrahedral_matmesh.NumTetrahedrons();
       ++tetrahedron_idx) {
    Matrix34d &mat_gradient             = gradient_matices.at(tetrahedron_idx);
    const auto &vertices_of_tetrahedron = tetrahedral_matmesh.mat_tetrahedrons.row(tetrahedron_idx);
    Matrix34d mat_tet_coordinates       = Matrix34d::Ones();
    for (index_t idx = 0; idx < 4; ++idx) {
      mat_tet_coordinates.block(0, idx, 3, 1) =
          tetrahedral_matmesh.mat_coordinates.row(vertices_of_tetrahedron(idx)).transpose();
    }
    Eigen::Matrix3d mat_A = mat_tet_coordinates * mat_indices;
    mat_gradient          = (mat_indices * mat_A.inverse()).transpose();
  }

  return gradient_matices;
}

auto ConvertEdgesSoupToChains(const std::vector<std::pair<index_t, index_t>> &edges_soup)
    -> std::vector<Chain> {
  std::vector<Chain> chains;
  std::map<index_t, std::vector<index_t>> map_vtx_to_edges;
  for (int edge_idx = 0; edge_idx < edges_soup.size(); ++edge_idx) {
    map_vtx_to_edges[edges_soup[edge_idx].first].push_back(edge_idx);
    map_vtx_to_edges[edges_soup[edge_idx].second].push_back(edge_idx);
  }
  std::set<VertexIndexEdge> edges_set(edges_soup.begin(), edges_soup.end());
  while (!edges_set.empty()) {
    Chain chain;
    chain.push_back(*edges_set.begin());
    edges_set.erase(edges_set.begin());
    while (!edges_set.empty() && map_vtx_to_edges[chain.back().second].size() == 2) {
      int num_edges_set_before = edges_set.size();
      for (auto edge = edges_set.begin(); edge != edges_set.end(); ++edge) {
        if (edge->first == chain.back().second) {
          chain.push_back(*edge);
          edges_set.erase(edge);
          break;
        }
        if (edge->second == chain.back().second) {
          chain.emplace_back(edge->second, edge->first);
          edges_set.erase(edge);
          break;
        }
      }
      if (num_edges_set_before == edges_set.size()) break;
    }

    while (!edges_set.empty() && map_vtx_to_edges[chain.front().first].size() == 2) {
      int num_edges_set_before = edges_set.size();
      for (auto edge = edges_set.begin(); edge != edges_set.end(); ++edge) {
        if (edge->first == chain.front().first) {
          chain.push_front(std::make_pair(edge->second, edge->first));
          edges_set.erase(edge);
          break;
        }
        if (edge->second == chain.front().first) {
          chain.push_front(*edge);
          edges_set.erase(edge);
          break;
        }
      }
      if (num_edges_set_before == edges_set.size()) break;
    }
    chains.push_back(chain);
  }
  return chains;
}

auto MarkMeshFaceOrientations(const Eigen::MatrixXd &mat_coordinates,
                              const Eigen::MatrixXi &mat_faces) -> std::vector<Orientation> {
  std::vector<Orientation> face_orientations(mat_faces.rows());
  Eigen::MatrixXd mat_normals_for_surface;
  igl::per_face_normals(mat_coordinates, mat_faces, mat_normals_for_surface);
  for (index_t face_idx = 0; face_idx < mat_faces.rows(); ++face_idx) {
    auto &&face_normal = mat_normals_for_surface.row(face_idx);
    auto &&abs_normal  = face_normal.cwiseAbs();
    face_orientations[face_idx] =
        (abs_normal.x() > abs_normal.y() && abs_normal.x() > abs_normal.z())
            ? (face_normal.x() >= 0 ? Orientation::X : Orientation::NegX)
            : ((abs_normal.y() > abs_normal.x() && abs_normal.y() > abs_normal.z())
                   ? (face_normal.y() >= 0 ? Orientation::Y : Orientation::NegY)
                   : (face_normal.z() >= 0 ? Orientation::Z : Orientation::NegZ));
  }
  return face_orientations;
};

void DivideMeshIntoPatchesByOrientation(const SurfaceTopoMesh3 &mesh3,
                                        const Eigen::MatrixXd &mat_coordinates,
                                        const std::vector<Orientation> &surface_orientations,
                                        std::vector<OrientedPatch> &oriented_patches,
                                        std::vector<index_t> &map_face_to_patch,
                                        std::vector<VertexIndexEdge> &boundary_edges_soup) {
  const size_t num_faces = mesh3.num_faces();
  oriented_patches.clear();
  map_face_to_patch.clear();
  boundary_edges_soup.clear();
  map_face_to_patch.resize(num_faces);
  std::set<SurfaceMesh3::Edge_index> patch_boundary_edges;
  std::vector<bool> face_visited_flags(num_faces, false);
  for (auto face : mesh3.faces()) {
    if (face_visited_flags[face.idx()]) continue;
    std::queue<SurfaceMesh3::Face_index> face_queue;
    face_queue.push(face);
    face_visited_flags[face.idx()] = true;

    index_t patch_idx = oriented_patches.size();
    OrientedPatch patch;
    patch.orientation = surface_orientations[face.idx()];
    patch.edge_length = 0;
    while (!face_queue.empty()) {
      auto current_face = face_queue.front();
      face_queue.pop();
      map_face_to_patch[current_face.idx()] = patch_idx;
      patch.face_indices.push_back(current_face);

      for (auto half_edge : mesh3.halfedges_around_face(mesh3.halfedge(current_face))) {
        auto neighbor_face = mesh3.face(mesh3.opposite(half_edge));
        if (!neighbor_face.is_valid()) continue;
        if (surface_orientations[neighbor_face.idx()] == patch.orientation) {
          if (face_visited_flags[neighbor_face.idx()]) continue;
          face_queue.push(neighbor_face);
          face_visited_flags[neighbor_face.idx()] = true;
        } else {
          patch_boundary_edges.insert(mesh3.edge(half_edge));
          patch.boundary_edge_indices.insert(mesh3.edge(half_edge));
          patch.edge_length += (mat_coordinates.row(mesh3.source(half_edge).idx()) -
                                mat_coordinates.row(mesh3.target(half_edge).idx()))
                                   .norm();
        }
      }  // end for neighbors
    }    // end while
    oriented_patches.push_back(patch);
  }  // end for every face

  for (auto patch_edge : patch_boundary_edges) {
    auto face_0              = mesh3.face(mesh3.halfedge(patch_edge));
    auto face_1              = mesh3.face(mesh3.opposite(mesh3.halfedge(patch_edge)));
    auto vertex_0            = mesh3.vertex(patch_edge, 0);
    auto vertex_1            = mesh3.vertex(patch_edge, 1);
    double edge_length       = PMP::edge_length(mesh3.halfedge(patch_edge), mesh3);
    index_t face_0_patch_idx = map_face_to_patch[face_0.idx()];
    index_t face_1_patch_idx = map_face_to_patch[face_1.idx()];
    if (face_0_patch_idx == face_1_patch_idx) continue;
    oriented_patches[face_0_patch_idx].neighbor_patches_with_boundary_length[face_1_patch_idx] +=
        edge_length;
    oriented_patches[face_0_patch_idx].neighbor_orientations.insert(
        oriented_patches[face_1_patch_idx].orientation);

    oriented_patches[face_1_patch_idx].neighbor_patches_with_boundary_length[face_0_patch_idx] +=
        edge_length;
    oriented_patches[face_1_patch_idx].neighbor_orientations.insert(
        oriented_patches[face_0_patch_idx].orientation);
    boundary_edges_soup.push_back(VertexIndexEdge(vertex_0.idx(), vertex_1.idx()));
  }
}
}  // namespace sha
}  // namespace da
