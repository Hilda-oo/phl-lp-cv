#include "polycube_based_utility.h"

#include <Eigen/Eigen>

#include <igl/remove_duplicate_vertices.h>

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <OpenVolumeMesh/FileManager/FileManager.hh>

#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-surface-mesh/mesh3.h"

#include <boost/range/adaptor/indexed.hpp>

namespace da {
namespace sha {
using VertexCoordinateEdge = std::pair<Eigen::Vector3d, Eigen::Vector3d>;

class DirectionalPlane {
 public:
  Orientation orientation_;
  double intercept_;
  std::pair<int, int> indices_;
  std::vector<VertexCoordinateEdge> boundary_edges_;

 public:
  explicit DirectionalPlane(Orientation orientation, double intercept,
                            const std::vector<VertexCoordinateEdge> &boundary_edges)
      : orientation_(orientation), intercept_(intercept), boundary_edges_(boundary_edges) {
    indices_ =
        (orientation == Orientation::X)
            ? std::make_pair(1, 2)
            : ((orientation == Orientation::Y) ? std::make_pair(0, 2) : std::make_pair(0, 1));
  }

  bool operator<(DirectionalPlane const &other) const {
    if (this->orientation_ != other.orientation_) return this->orientation_ < other.orientation_;
    return this->intercept_ < other.intercept_;
  }

  bool IsParallelWith(const DirectionalPlane &other) const {
    return this->orientation_ == other.orientation_;
  }

  bool ContainsPoint(const Eigen::Vector3d &point) const {
    int n_cross                  = 0;
    std::pair<double, double> pt = std::make_pair(point[indices_.first], point[indices_.second]);
    for (const std::pair<Eigen::Vector3d, Eigen::Vector3d> &boundary : this->boundary_edges_) {
      std::pair<double, double> p1 =
          std::make_pair(boundary.first[indices_.first], boundary.first[indices_.second]);
      std::pair<double, double> p2 =
          std::make_pair(boundary.second[indices_.first], boundary.second[indices_.second]);
      if (p1.second == p2.second) continue;
      if (pt.second < std::min(p1.second, p2.second)) continue;
      if (pt.second >= std::max(p1.second, p2.second)) continue;
      double x =
          (pt.second - p1.second) * (p2.first - p1.first) / (p2.second - p1.second) + p1.first;
      if (approximate(x - pt.first, 1e-6) > 0) {
        n_cross++;
      }
    }
    return n_cross % 2 == 1;
  }
};

bool CompareDirectionalPlanePointer(const std::shared_ptr<DirectionalPlane> &plane1,
                                    const std::shared_ptr<DirectionalPlane> &plane2) {
  if (plane1->orientation_ != plane2->orientation_)
    return plane1->orientation_ < plane2->orientation_;
  return plane1->intercept_ < plane2->intercept_;
}

struct ComponentBox {
  std::shared_ptr<DirectionalPlane> plane_min_x, plane_max_x;
  std::shared_ptr<DirectionalPlane> plane_min_y, plane_max_y;
  std::shared_ptr<DirectionalPlane> plane_min_z, plane_max_z;
  std::pair<size_t, size_t> num_planes_along_x, num_planes_along_y, num_planes_along_z;
  Eigen::AlignedBox3d range;

 public:
  using PlanePtr = std::shared_ptr<DirectionalPlane>;

  explicit ComponentBox(const std::pair<PlanePtr, PlanePtr> &plane_pair_x,
                        const std::pair<size_t, size_t> &num_planes_along_x,
                        const std::pair<PlanePtr, PlanePtr> &plane_pair_y,
                        const std::pair<size_t, size_t> &num_planes_along_y,
                        const std::pair<PlanePtr, PlanePtr> &plane_pair_z,
                        const std::pair<size_t, size_t> &num_planes_along_z)
      : plane_min_x(plane_pair_x.first),
        plane_max_x(plane_pair_x.second),
        plane_min_y(plane_pair_y.first),
        plane_max_y(plane_pair_y.second),
        plane_min_z(plane_pair_z.first),
        plane_max_z(plane_pair_z.second),
        num_planes_along_x(num_planes_along_x),
        num_planes_along_y(num_planes_along_y),
        num_planes_along_z(num_planes_along_z),
        range(Eigen::Vector3d(plane_min_x->intercept_, plane_min_y->intercept_,
                              plane_min_z->intercept_),
              Eigen::Vector3d(plane_max_x->intercept_, plane_max_y->intercept_,
                              plane_max_z->intercept_)) {
    Assert(range.max().x() > range.min().x());
    Assert(range.max().y() > range.min().y());
    Assert(range.max().z() > range.min().z());
  }

  Eigen::Vector3d Center() const { return range.center(); }
};

class PolycubeBoundaryModel {
 public:
  std::vector<std::shared_ptr<DirectionalPlane>> planes_x, planes_y, planes_z;
  std::vector<std::shared_ptr<ComponentBox>> boundary_boxes;

 public:
  bool ContainsBox(const ComponentBox &box) const {
    Eigen::Vector3d center = box.Center();
    decltype(&planes_x) planes;
    decltype(&box.num_planes_along_x) num_planes_along_axis;
    size_t num_min_planes_x = std::min(box.num_planes_along_x.first, box.num_planes_along_x.second);
    size_t num_min_planes_y = std::min(box.num_planes_along_y.first, box.num_planes_along_y.second);
    size_t num_min_planes_z = std::min(box.num_planes_along_z.first, box.num_planes_along_z.second);

    if (num_min_planes_x <= num_min_planes_y && num_min_planes_x <= num_min_planes_z) {
      planes                = &planes_x;
      num_planes_along_axis = &box.num_planes_along_x;
    } else if (num_min_planes_y <= num_min_planes_x && num_min_planes_y <= num_min_planes_z) {
      planes                = &planes_y;
      num_planes_along_axis = &box.num_planes_along_y;
    } else {
      planes                = &planes_z;
      num_planes_along_axis = &box.num_planes_along_z;
    }
    index_t begin_idx, end_idx;
    if (num_planes_along_axis->first < num_planes_along_axis->second) {
      begin_idx = 0;
      end_idx   = num_planes_along_axis->first + 1;
    } else {
      begin_idx = planes->size() - num_planes_along_axis->second;
      end_idx   = planes->size();
    }
    size_t num_across_planes = 0;
    for (index_t idx = begin_idx; idx < end_idx; ++idx) {
      if (planes->at(idx)->ContainsPoint(center)) {
        num_across_planes++;
      }
    }
    return num_across_planes % 2 == 1;
  }
};

auto GenerateHexahedralMeshFromPolycubeDomain(const PolycubeBoundaryModel &polycube_domain)
    -> HexahedralMatMesh {
  Assert(polycube_domain.planes_x.size() >= 2, "x planes size >= 2");
  Assert(polycube_domain.planes_y.size() >= 2, "y planes size >= 2");
  Assert(polycube_domain.planes_z.size() >= 2, "z planes size >= 2");
  auto planes_x = polycube_domain.planes_x;
  auto planes_y = polycube_domain.planes_y;
  auto planes_z = polycube_domain.planes_z;

  auto CompareDirectionalPlane = [](const std::shared_ptr<DirectionalPlane> &plane_1,
                                    const std::shared_ptr<DirectionalPlane> &plane_2) {
    return plane_1->intercept_ < plane_2->intercept_;
  };
  std::sort(planes_x.begin(), planes_x.end(), CompareDirectionalPlane);
  std::sort(planes_y.begin(), planes_y.end(), CompareDirectionalPlane);
  std::sort(planes_z.begin(), planes_z.end(), CompareDirectionalPlane);
  std::vector<std::shared_ptr<ComponentBox>> boundary_boxes;

  log::info("plane X: #{}, {}~{}", planes_x.size(), planes_x.front()->intercept_,
            planes_x.back()->intercept_);
  log::info("plane Y: #{}, {}~{}", planes_y.size(), planes_y.front()->intercept_,
            planes_y.back()->intercept_);
  log::info("plane Z: #{}, {}~{}", planes_z.size(), planes_z.front()->intercept_,
            planes_z.back()->intercept_);

  for (index_t plane_min_z_idx = 0; plane_min_z_idx + 1 < planes_z.size(); ++plane_min_z_idx) {
    index_t plane_max_z_idx = plane_min_z_idx + 1;
    if (planes_z[plane_min_z_idx]->intercept_ == planes_z[plane_max_z_idx]->intercept_) continue;
    for (index_t plane_min_y_idx = 0; plane_min_y_idx + 1 < planes_y.size(); ++plane_min_y_idx) {
      index_t plane_max_y_idx = plane_min_y_idx + 1;
      if (planes_y[plane_min_y_idx]->intercept_ == planes_y[plane_max_y_idx]->intercept_) continue;
      for (index_t plane_min_x_idx = 0; plane_min_x_idx + 1 < planes_x.size(); ++plane_min_x_idx) {
        index_t plane_max_x_idx = plane_min_x_idx + 1;
        if (planes_x[plane_min_x_idx]->intercept_ == planes_x[plane_max_x_idx]->intercept_)
          continue;

        std::shared_ptr<ComponentBox> box = std::make_shared<ComponentBox>(
            std::make_pair(planes_x[plane_min_x_idx], planes_x[plane_max_x_idx]),
            std::make_pair(plane_min_x_idx, planes_x.size() - plane_max_x_idx),
            std::make_pair(planes_y[plane_min_y_idx], planes_y[plane_max_y_idx]),
            std::make_pair(plane_min_y_idx, planes_y.size() - plane_max_y_idx),
            std::make_pair(planes_z[plane_min_z_idx], planes_z[plane_max_z_idx]),
            std::make_pair(plane_min_z_idx, planes_z.size() - plane_max_z_idx));
        if (!polycube_domain.ContainsBox(*box)) {
          continue;
        }
        boundary_boxes.push_back(box);
      }
    }
  }
  log::info("Generated {} rough boxes", boundary_boxes.size());

  std::vector<Eigen::Vector3d> hexahedral_points;
  std::vector<std::array<index_t, 8>> hexahedral_cells;
  auto MakeHexahedralCubeCell = [&hexahedral_points, &hexahedral_cells](double last_x, double x,
                                                                        double last_y, double y,
                                                                        double last_z, double z) {
    std::array<Eigen::Vector3d, 8> cube_points;
    std::array<index_t, 8> cube_cell;
    cube_points[0] << last_x, last_y, last_z;
    cube_points[1] << x, last_y, last_z;
    cube_points[2] << x, y, last_z;
    cube_points[3] << last_x, y, last_z;
    cube_points[4] << last_x, last_y, z;
    cube_points[5] << x, last_y, z;
    cube_points[6] << x, y, z;
    cube_points[7] << last_x, y, z;

    const size_t base_num_points = hexahedral_points.size();
    for (index_t idx = 0; idx < 8; ++idx) {
      hexahedral_points.push_back(cube_points[idx]);
      cube_cell[idx] = base_num_points + idx;
    }
    hexahedral_cells.push_back(cube_cell);
  };

  constexpr double kMergeThreshold = 0.5;
  constexpr double KTolerance      = 1e-6;
  for (auto rough_box : boundary_boxes) {
    const auto &box_range = rough_box->range;
    int min_int_x         = std::ceil(box_range.min().x());
    int min_int_y         = std::ceil(box_range.min().y());
    int min_int_z         = std::ceil(box_range.min().z());

    double last_x, last_y, last_z;
    double x, y, z;

    std::vector<double> values_x = {box_range.min().x()}, values_y = {box_range.min().y()},
                        values_z = {box_range.min().z()};

    for (z = (min_int_z == box_range.min().z() ? min_int_z + 1 : min_int_z);
         z < box_range.max().z() + 1; z += 1) {
      z = std::min(z, box_range.max().z());
      values_z.push_back(z);
    }

    for (y = (min_int_y == box_range.min().y() ? min_int_y + 1 : min_int_y);
         y < box_range.max().y() + 1; y += 1) {
      y = std::min(y, box_range.max().y());
      values_y.push_back(y);
    }

    for (x = (min_int_x == box_range.min().x() ? min_int_x + 1 : min_int_x);
         x < box_range.max().x() + 1; x += 1) {
      x = std::min(x, box_range.max().x());
      values_x.push_back(x);
    }

    Assert(values_x.size() >= 2);
    Assert(values_y.size() >= 2);
    Assert(values_z.size() >= 2);

    // merge point that dis less than 0.5
    auto MergeClosetValues = [&](std::vector<double> &value) {
      if (value.size() > 2) {
        if (value[1] - value.front() <= kMergeThreshold) value.erase(value.begin() + 1);
      }
      if (value.size() > 2) {
        if (value.back() - value[value.size() - 2] <= kMergeThreshold) value.erase(value.end() - 2);
      }
    };
    MergeClosetValues(values_x);
    MergeClosetValues(values_y);
    MergeClosetValues(values_z);

    Assert(values_x.size() >= 2);
    Assert(values_y.size() >= 2);
    Assert(values_z.size() >= 2);

    for (index_t z_idx = 1; z_idx < values_z.size(); ++z_idx) {
      last_z = values_z[z_idx - 1];
      z      = values_z[z_idx];
      for (index_t y_idx = 1; y_idx < values_y.size(); ++y_idx) {
        last_y = values_y[y_idx - 1];
        y      = values_y[y_idx];
        for (index_t x_idx = 1; x_idx < values_x.size(); ++x_idx) {
          last_x = values_x[x_idx - 1];
          x      = values_x[x_idx];
          MakeHexahedralCubeCell(last_x, x, last_y, y, last_z, z);
        }  // for x
      }    // for y
    }      // for z
  }        // for every box
  HexahedralMatMesh duplicate_hexahedral_matmesh;
  HexahedralMatMesh hexahedral_matmesh;
  duplicate_hexahedral_matmesh.mat_coordinates.resize(hexahedral_points.size(), 3);
  duplicate_hexahedral_matmesh.mat_hexahedrons.resize(hexahedral_cells.size(), 8);
  for (index_t point_idx = 0; point_idx < hexahedral_points.size(); ++point_idx) {
    duplicate_hexahedral_matmesh.mat_coordinates.row(point_idx) = hexahedral_points[point_idx];
  }
  for (index_t cell_idx = 0; cell_idx < hexahedral_cells.size(); ++cell_idx) {
    for (index_t idx = 0; idx < 8; ++idx) {
      duplicate_hexahedral_matmesh.mat_hexahedrons(cell_idx, idx) = hexahedral_cells[cell_idx][idx];
    }
  }
  Eigen::VectorXi map_old_vtx_to_new_vtx, map_new_vtx_to_old_vtx;

  igl::remove_duplicate_vertices(duplicate_hexahedral_matmesh.mat_coordinates,
                                 duplicate_hexahedral_matmesh.mat_hexahedrons, KTolerance,
                                 hexahedral_matmesh.mat_coordinates, map_old_vtx_to_new_vtx,
                                 map_new_vtx_to_old_vtx, hexahedral_matmesh.mat_hexahedrons);
  return hexahedral_matmesh;
}

namespace hex_polycube {
void LaplacianSmoothForSurfacePatchBoundaryEdges(
    const SurfaceMesh3 &surface_mesh, const std::vector<VertexIndexEdge> &boundary_edges_soup,
    Eigen::MatrixXd &mat_coordinates, size_t num_iterations) {
  log::info("Hex polycube Smoothing Patch Boundary for #{}", num_iterations);
  auto boundary_chains = ConvertEdgesSoupToChains(boundary_edges_soup);

  for (index_t iteration_idx = 0; iteration_idx < num_iterations; ++iteration_idx) {
    for (auto &&[chain_idx, boundary_chain] : boundary_chains | boost::adaptors::indexed(0)) {
      Eigen::Vector3d corner_vector = mat_coordinates.row(boundary_chain.back().second) -
                                      mat_coordinates.row(boundary_chain.front().first);
      Eigen::RowVector3d step = corner_vector / boundary_chain.size();
      for (int edge_idx = 0; edge_idx < boundary_chain.size(); ++edge_idx) {
        const auto &edge                 = boundary_chain[edge_idx];
        mat_coordinates.row(edge.second) = mat_coordinates.row(edge.first) + step;
      }
    }
  }
}

void LaplacianSmoothForSurfacePatches(const SurfaceMesh3 &surface_mesh,
                                      const std::vector<VertexIndexEdge> &boundary_edges_soup,
                                      Eigen::MatrixXd &mat_coordinates, size_t num_iterations) {
  log::info("Hex polycube Smoothing Surface for #{}", num_iterations);
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

void LaplacianSmoothForVolume(const HexahedralMesh &hexahedral_mesh,
                              Eigen::MatrixXd &mat_coordinates, size_t num_iterations) {
  log::info("Hex polycube Smoothing Volume for #{}", num_iterations);
  for (index_t iteration_idx = 0; iteration_idx < num_iterations; ++iteration_idx) {
    for (auto vertex : hexahedral_mesh.vertices()) {
      if (hexahedral_mesh.is_boundary(vertex)) continue;
      Eigen::Vector3d new_coordinate = mat_coordinates.row(vertex.idx());
      size_t num_neighbors           = 0;
      for (auto neighbor_vtx : hexahedral_mesh.vertex_vertices(vertex)) {
        new_coordinate += mat_coordinates.row(neighbor_vtx.idx());
        num_neighbors++;
      }
      new_coordinate /= (num_neighbors + 1);
      mat_coordinates.row(vertex.idx()) = new_coordinate;
    }
  }
}
}  // namespace hex_polycube

auto RemeshTetrahedronPolycubeToHexhahedron(
    const TetrahedralMatMesh &tetrahedral_matmesh,
    const Eigen::MatrixXd &mat_tetrahedral_polycube_coordinates, const Eigen::Vector3d &scale)
    -> HexahedralMatMesh {
  Eigen::MatrixXd mat_coordinates = mat_tetrahedral_polycube_coordinates;

  Eigen::AlignedBox3d boundbox(mat_coordinates.colwise().minCoeff(),
                               mat_coordinates.colwise().maxCoeff());
  auto ScaleCoordinateMatrix = [&boundbox](Eigen::MatrixXd &mat_coordinates,
                                           const Eigen::Vector3d &scale) {
    for (index_t vertex_idx = 0; vertex_idx < mat_coordinates.rows(); ++vertex_idx) {
      Eigen::Vector3d coordinate = mat_coordinates.row(vertex_idx);
      mat_coordinates.row(vertex_idx) =
          boundbox.min() + (coordinate - boundbox.min()).cwiseProduct(scale);
    }
  };

  ScaleCoordinateMatrix(mat_coordinates, scale);

  TetrahedralTopoMesh tetrahedral_topomesh = sha::CreateTetrahedralTopoMeshFromMatrix(
      tetrahedral_matmesh.NumVertices(), tetrahedral_matmesh.mat_tetrahedrons);
  SurfaceTopoMesh3 surface_topomesh =
      sha::CreateSurfaceTopoMesh3FromTetrahedralTopoMesh(tetrahedral_topomesh);
  Eigen::MatrixXi mat_surface_triangles =
      sha::CreateSurfaceMatrixFromTetrahedralTopoMesh(tetrahedral_topomesh);
  std::vector<Orientation> surface_face_orientations =
      MarkMeshFaceOrientations(mat_tetrahedral_polycube_coordinates, mat_surface_triangles);

  auto ExtractPolycubeDomainPlanes = [&mat_coordinates, &surface_topomesh,
                                      &surface_face_orientations]() -> PolycubeBoundaryModel {
    PolycubeBoundaryModel polycube_boundary_domain;

    std::multiset<std::shared_ptr<DirectionalPlane>, decltype(&CompareDirectionalPlanePointer)>
        x_planes(&CompareDirectionalPlanePointer), y_planes(&CompareDirectionalPlanePointer),
        z_planes(&CompareDirectionalPlanePointer);

    std::vector<OrientedPatch> oriented_patches;
    std::vector<index_t> map_face_to_patch;
    std::vector<VertexIndexEdge> boundary_edges_soup;
    DivideMeshIntoPatchesByOrientation(surface_topomesh, mat_coordinates, surface_face_orientations,
                                       oriented_patches, map_face_to_patch, boundary_edges_soup);

    for (auto oriented_patch : oriented_patches) {
      index_t component_idx     = GetNondirectionalIndexByOrientation(oriented_patch.orientation);
      double component_position = 0;
      for (auto face : oriented_patch.face_indices) {
        auto [vertices_it, vertices_end] =
            surface_topomesh.vertices_around_face(surface_topomesh.halfedge(face));
        component_position += (mat_coordinates(vertices_it->idx(), component_idx) +
                               mat_coordinates((++vertices_it)->idx(), component_idx) +
                               mat_coordinates((++vertices_it)->idx(), component_idx)) /
                              3;
      }
      component_position /= oriented_patch.face_indices.size();

      log::info("oriented_patch.boundary_edge_indices: {}",
                oriented_patch.boundary_edge_indices.size());
      std::vector<VertexCoordinateEdge> boundary_vec;
      boost::transform(oriented_patch.boundary_edge_indices, std::back_inserter(boundary_vec),
                       [&](const SurfaceMesh3::Edge_index &edge) {
                         return VertexCoordinateEdge(
                             mat_coordinates.row(surface_topomesh.vertex(edge, 0).idx()),
                             mat_coordinates.row(surface_topomesh.vertex(edge, 1).idx()));
                       });
      Orientation undirected_orientation = oriented_patch.orientation;
      if (undirected_orientation == Orientation::NegX) undirected_orientation = Orientation::X;
      if (undirected_orientation == Orientation::NegY) undirected_orientation = Orientation::Y;
      if (undirected_orientation == Orientation::NegZ) undirected_orientation = Orientation::Z;

      std::shared_ptr<DirectionalPlane> plane_ptr = std::make_shared<DirectionalPlane>(
          undirected_orientation, component_position, boundary_vec);
      switch (oriented_patch.orientation) {
        case Orientation::X:
        case Orientation::NegX:
          x_planes.insert(plane_ptr);
          break;
        case Orientation::Y:
        case Orientation::NegY:
          y_planes.insert(plane_ptr);
          break;
        case Orientation::Z:
        case Orientation::NegZ:
          z_planes.insert(plane_ptr);
          break;
      }
    }  // for every patch
    boost::copy(x_planes, std::inserter(polycube_boundary_domain.planes_x,
                                        polycube_boundary_domain.planes_x.begin()));
    boost::copy(y_planes, std::inserter(polycube_boundary_domain.planes_y,
                                        polycube_boundary_domain.planes_y.begin()));
    boost::copy(z_planes, std::inserter(polycube_boundary_domain.planes_z,
                                        polycube_boundary_domain.planes_z.begin()));
    return polycube_boundary_domain;
  };

  PolycubeBoundaryModel polycube_domain = ExtractPolycubeDomainPlanes();

  HexahedralMatMesh hexahedral_polycube_matmesh =
      GenerateHexahedralMeshFromPolycubeDomain(polycube_domain);

  Eigen::AlignedBox3d tetbox(mat_tetrahedral_polycube_coordinates.colwise().minCoeff(),
                             mat_tetrahedral_polycube_coordinates.colwise().maxCoeff());
  Eigen::AlignedBox3d hexbox(hexahedral_polycube_matmesh.mat_coordinates.colwise().minCoeff(),
                             hexahedral_polycube_matmesh.mat_coordinates.colwise().maxCoeff());

  ScaleCoordinateMatrix(hexahedral_polycube_matmesh.mat_coordinates, scale.cwiseInverse());

  HexahedralMesh hexahedral_polycube_mesh =
      sha::CreateHexahedralMeshFromMatMesh(hexahedral_polycube_matmesh);

  log::info("Hex: V #{}", hexahedral_polycube_mesh.n_vertices());
  log::info("Hex: C #{}", hexahedral_polycube_mesh.n_cells());
  log::info("Hex: F #{}", hexahedral_polycube_mesh.n_faces());
  log::info("Hex: E #{}", hexahedral_polycube_mesh.n_edges());

  SurfaceMesh3 quadrilateral_mesh =
      sha::CreateSurfaceMesh3FromHexahedralMesh(hexahedral_polycube_mesh);

  MatMesh3 quadrilateral_matmesh = sha::CreateMatMesh3FromSurfaceMesh3(quadrilateral_mesh, 4);

  std::vector<Orientation> quadrilateral_orientations = MarkMeshFaceOrientations(
      hexahedral_polycube_matmesh.mat_coordinates, quadrilateral_matmesh.mat_faces.leftCols(3));

  std::vector<OrientedPatch> oriented_patches;
  std::vector<index_t> map_face_to_patch;
  std::vector<VertexIndexEdge> boundary_edges_soup;
  DivideMeshIntoPatchesByOrientation(
      quadrilateral_mesh, hexahedral_polycube_matmesh.mat_coordinates, quadrilateral_orientations,
      oriented_patches, map_face_to_patch, boundary_edges_soup);

  log::info("Quadrilateral Face: #{}", quadrilateral_mesh.num_faces());
  log::info("Quadrilateral Patch: #{}", oriented_patches.size());

  hex_polycube::LaplacianSmoothForSurfacePatchBoundaryEdges(
      quadrilateral_mesh, boundary_edges_soup, hexahedral_polycube_matmesh.mat_coordinates, 10);

  hex_polycube::LaplacianSmoothForSurfacePatches(quadrilateral_mesh, boundary_edges_soup,
                                                 hexahedral_polycube_matmesh.mat_coordinates, 10);

  hex_polycube::LaplacianSmoothForVolume(hexahedral_polycube_mesh,
                                         hexahedral_polycube_matmesh.mat_coordinates, 10);

  // TODO(wpkong): Pillowing
  return hexahedral_polycube_matmesh;
}
}  // namespace sha
}  // namespace da
