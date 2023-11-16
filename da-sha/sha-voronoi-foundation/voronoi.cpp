#include "voronoi.h"

#include <geogram/delaunay/delaunay_3d.h>
#include <geogram/delaunay/periodic_delaunay_3d.h>
#include <geogram/voronoi/convex_cell.h>

#include <oneapi/tbb.h>

#include <igl/adjacency_list.h>
#include <igl/boundary_loop.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/facet_components.h>
#include <igl/octree.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/remove_unreferenced.h>
#include <igl/sharp_edges.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/combine.hpp>
#include <boost/range/numeric.hpp>

#include "sha-base-framework/declarations.h"
#include "sha-base-framework/frame.h"
#include "sha-base-framework/logger.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-surface-mesh/mesh3.h"

#include "utility.h"

namespace da {
namespace sha {
VoronoiDiagram CreateVoronoiDiagramInDomain(const Eigen::AlignedBox3d &domain,
                                            const Eigen::MatrixXd &mat_seeds,
                                            size_t num_lloyd_iterations) {
  const size_t num_seeds = mat_seeds.rows();
  VoronoiDiagram voronoi_diagram;
  Eigen::Vector3d transform_factor = domain.min();
  double scale_factor              = domain.sizes().maxCoeff();

  std::vector<Eigen::Vector4d> regulated_domain_box_planes = {
      {1, 0, 0, 0}, {-1, 0, 0, domain.sizes().x() / scale_factor},
      {0, 1, 0, 0}, {0, -1, 0, domain.sizes().y() / scale_factor},
      {0, 0, 1, 0}, {0, 0, -1, domain.sizes().z() / scale_factor}};

  Eigen::MatrixXd mat_regulated_seeds;
  mat_regulated_seeds = (mat_seeds.rowwise() - transform_factor.transpose()).array() / scale_factor;

  auto MapUnitCoordinateToOriginSpace = [&](const Eigen::Vector3d &unit_coord) -> Eigen::Vector3d {
    return unit_coord * scale_factor + transform_factor;
  };

  std::vector<double> flatten_seeds(num_seeds * 3);
  boost::for_each(boost::counting_range<size_t>(0, num_seeds), [&](index_t seed_idx) {
    boost::for_each(boost::counting_range<size_t>(0, 3), [&](index_t idx) {
      flatten_seeds[seed_idx * 3 + idx] = mat_regulated_seeds(seed_idx, idx);
    });
  });
  GEO::initialize();
  auto delaunay = std::make_shared<GEO::PeriodicDelaunay3d>(false);
  //  GEO::PeriodicDelaunay3d delaunay(false);

  delaunay->set_keeps_infinite(true);
  delaunay->set_vertices(num_seeds, flatten_seeds.data());
  delaunay->set_stores_neighbors(true);
  delaunay->compute();

  // if using CVT
  if (num_lloyd_iterations > 0) {
    for (index_t iter_idx = 0; iter_idx < num_lloyd_iterations; ++iter_idx) {
      log::info("lloyd iteration {} / {}", iter_idx + 1, num_lloyd_iterations);
      std::vector<double> new_points(mat_seeds.rows() * 3);
      for (index_t v = 0; v < mat_seeds.rows(); ++v) {
        GEO::ConvexCell convex_cell;
        delaunay->copy_Laguerre_cell_from_Delaunay(v, convex_cell);
        for (auto &plane : regulated_domain_box_planes) {
          convex_cell.clip_by_plane(GEO::vec4(plane.x(), plane.y(), plane.z(), plane.w()));
        }
        convex_cell.compute_geometry();
        GEO::vec3 g           = convex_cell.barycenter();
        new_points[3 * v]     = g.x;
        new_points[3 * v + 1] = g.y;
        new_points[3 * v + 2] = g.z;
      }
      // In periodic mode, points may escape out of
      // the domain. Relocate them in [0,1]^3
      for (index_t i = 0; i < new_points.size(); ++i) {
        if (new_points[i] < 0.0) {
          new_points[i] += 1.0;
        }
        if (new_points[i] > 1.0) {
          new_points[i] -= 1.0;
        }
      }
      std::copy(new_points.begin(), new_points.end(), flatten_seeds.begin());
      delaunay->set_vertices(flatten_seeds.size() / 3, flatten_seeds.data());
      delaunay->compute();
    }
  }

  auto CreateVoronoiCellFromConvexCell = [&](GEO::ConvexCell &convex_cell) -> VoronoiCell {
    VoronoiCell voronoi_cell;
    std::vector<GEO::index_t> map_vertex_to_triangle_idx(convex_cell.nb_v(), GEO::index_t(-1));
    std::vector<GEO::index_t> triangle_indices(convex_cell.nb_t(), GEO::index_t(-1));
    GEO::index_t curr_num_triangles = 0;
    GEO::index_t triangle_idx       = convex_cell.first_triangle();

    std::vector<Eigen::Vector3d> cell_vertex_vector;
    std::vector<Eigen::Vector3i> cell_triangle_vector;

    while (triangle_idx != VBW::END_OF_LIST) {
      auto triangle_with_flag = convex_cell.get_triangle_and_flags(triangle_idx);
      auto point              = convex_cell.stored_triangle_point(triangle_idx);
      cell_vertex_vector.emplace_back(point.x, point.y, point.z);
      triangle_indices[triangle_idx] = curr_num_triangles;
      ++curr_num_triangles;
      map_vertex_to_triangle_idx[triangle_with_flag.i] = triangle_idx;
      map_vertex_to_triangle_idx[triangle_with_flag.j] = triangle_idx;
      map_vertex_to_triangle_idx[triangle_with_flag.k] = triangle_idx;
      triangle_idx                                     = GEO::index_t(triangle_with_flag.flags);
    }

    for (GEO::index_t vertex_idx = 1; vertex_idx < convex_cell.nb_v(); ++vertex_idx) {
      if (map_vertex_to_triangle_idx[vertex_idx] != GEO::index_t(-1)) {
        GEO::index_t t = map_vertex_to_triangle_idx[vertex_idx];
        PolygonFace face;
        face.boundary_vtx_loops.resize(1);
        auto &boundary_vtx_loop = face.boundary_vtx_loops.front();
        do {
          boundary_vtx_loop.push_back(triangle_indices[t]);
          index_t lv = convex_cell.triangle_find_vertex(t, vertex_idx);
          t          = convex_cell.triangle_adjacent(t, (lv + 1) % 3);
        } while (t != map_vertex_to_triangle_idx[vertex_idx]);

        for (int i = 2; i < boundary_vtx_loop.size(); ++i) {
          cell_triangle_vector.push_back(
              {boundary_vtx_loop[0], boundary_vtx_loop[i - 1], boundary_vtx_loop[i]});
          face.triangle_face_indices_in_cell.push_back(cell_triangle_vector.size() - 1);
          voronoi_cell.map_triangle_to_polygon_idx.push_back(
              voronoi_cell.polyhedron.polygons.size());
        }
        face.mat_triangle_faces = voronoi_cell.cell_triangle_mesh.mat_faces(
            face.triangle_face_indices_in_cell, Eigen::all);
        voronoi_cell.polyhedron.polygons.push_back(face);
      }
    }

    auto &mat_coordinates_of_cell     = voronoi_cell.cell_triangle_mesh.mat_coordinates;
    auto &mat_faces_of_cell           = voronoi_cell.cell_triangle_mesh.mat_faces;
    const size_t num_vertices_of_cell = cell_vertex_vector.size();
    const size_t num_faces_of_cell    = cell_triangle_vector.size();
    mat_coordinates_of_cell.resize(num_vertices_of_cell, 3);
    mat_faces_of_cell.resize(num_faces_of_cell, 3);

    boost::for_each(boost::counting_range<index_t>(0, num_vertices_of_cell),
                    [&](index_t vertex_idx) {
                      mat_coordinates_of_cell.row(vertex_idx) =
                          MapUnitCoordinateToOriginSpace(cell_vertex_vector[vertex_idx]);
                    });
    boost::for_each(boost::counting_range<index_t>(0, num_faces_of_cell), [&](index_t face_idx) {
      mat_faces_of_cell.row(face_idx) << cell_triangle_vector[face_idx].x(),
          cell_triangle_vector[face_idx].y(), cell_triangle_vector[face_idx].z();
    });
    return voronoi_cell;
  };

  auto BuildVoronoiCell = [&](index_t seed_idx) -> VoronoiCell {
    GEO::ConvexCell convex_cell;
    delaunay->copy_Laguerre_cell_from_Delaunay(seed_idx, convex_cell);
    for (auto &plane : regulated_domain_box_planes) {
      convex_cell.clip_by_plane(GEO::vec4(plane.x(), plane.y(), plane.z(), plane.w()));
    }
    convex_cell.compute_geometry();
    return CreateVoronoiCellFromConvexCell(convex_cell);
  };

  voronoi_diagram.map_cell_to_neighbor_indices.resize(num_seeds);

  boost::for_each(boost::counting_range<size_t>(0, num_seeds), [&](index_t seed_idx) {
    const index_t &cell_idx  = seed_idx;
    VoronoiCell voronoi_cell = BuildVoronoiCell(seed_idx);
    voronoi_cell.seed        = mat_seeds.row(seed_idx);
    // voronoi_cell.map_triangle_to_origin_face_idx.resize(voronoi_cell.cell_triangle_mesh.NumFaces(),
    //                                                     -1);
    voronoi_diagram.cells.push_back(voronoi_cell);
    VBW::vector<GEO::index_t> neighbor_cell_indices_of_current_cell;
    delaunay->get_neighbors(seed_idx, neighbor_cell_indices_of_current_cell);
    for (auto neighbor_cell_idx : neighbor_cell_indices_of_current_cell) {
      if (neighbor_cell_idx == -1) continue;
      voronoi_diagram.map_cell_to_neighbor_indices[cell_idx].insert(neighbor_cell_idx);
    }
  });
  return voronoi_diagram;
}

VoronoiDiagram CreateRestrictedVoronoiDiagramFromMesh(const MatMesh3 &shell_mesh,
                                                      const Eigen::MatrixXd &mat_seeds,
                                                      size_t num_lloyd_iterations,
                                                      double sharp_angle, bool keep_number_flag) {
  auto shell_domain = shell_mesh.AlignedBox();
  auto unrestricted_voronoi_diagram =
      CreateVoronoiDiagramInDomain(shell_domain, mat_seeds, num_lloyd_iterations);
  log::info("Created Voronoi Diagram In Domain");
  std::map<index_t, std::set<index_t>> map_unrestricted_cell_idx_to_restricted_cell_idx;

  const int num_cells = unrestricted_voronoi_diagram.cells.size();

  VoronoiDiagram restricted_voronoi_diagram;

  std::map<index_t, bool> map_restricted_cell_idx_to_is_totally_in_shell;

  // oneapi::tbb::parallel_for(0, num_cells, 1, [&](index_t unrestricted_cell_idx) {
  // boost::range::for_each(boost::irange(0, num_cells), [&](index_t unrestricted_cell_idx) {

  // log::info("Building octree");
  // DynamicBoundingBoxOctree<int> shell_boundingbox_octree(shell_domain, 6);
  // for (index_t face_idx = 0; face_idx < shell_mesh.NumFaces(); ++face_idx) {
  //   Eigen::MatrixXd triangle_coords(3, 3);
  //   triangle_coords.row(0) = shell_mesh.mat_coordinates.row(shell_mesh.mat_faces(face_idx, 0));
  //   triangle_coords.row(1) = shell_mesh.mat_coordinates.row(shell_mesh.mat_faces(face_idx, 1));
  //   triangle_coords.row(2) = shell_mesh.mat_coordinates.row(shell_mesh.mat_faces(face_idx, 2));
  //   auto triangle_box      = Eigen::AlignedBox3d(triangle_coords.colwise().minCoeff(),
  //                                           triangle_coords.colwise().maxCoeff());
  //   shell_boundingbox_octree.InsertData(triangle_box, 1);
  // }

  double total_mmsecs = 0;
  log::info("Start Boolean");
#pragma omp parallel for
  for (index_t unrestricted_cell_idx = 0; unrestricted_cell_idx < num_cells;
       ++unrestricted_cell_idx) {
    auto &unrestricted_cell = unrestricted_voronoi_diagram.cells.at(unrestricted_cell_idx);
    VoronoiCell new_voronoi_cell;
    new_voronoi_cell.seed      = unrestricted_cell.seed;
    MatMesh3 &new_voronoi_mesh = new_voronoi_cell.cell_triangle_mesh;
    Eigen::VectorXi birth_face_indices;

    auto t1          = std::chrono::steady_clock::now();
    new_voronoi_mesh = sha::BooleanIntersectTwoMatMesh3(
        shell_mesh, unrestricted_cell.cell_triangle_mesh, birth_face_indices);
    auto t2 = std::chrono::steady_clock::now();

    total_mmsecs += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    if (new_voronoi_mesh.NumVertices() == 0) {
      if (keep_number_flag) {
        index_t restricted_cell_idx = restricted_voronoi_diagram.cells.size();
        restricted_voronoi_diagram.cells.push_back(new_voronoi_cell);
        restricted_voronoi_diagram.map_cell_to_neighbor_indices.push_back(
            unrestricted_voronoi_diagram.map_cell_to_neighbor_indices[unrestricted_cell_idx]);
        map_unrestricted_cell_idx_to_restricted_cell_idx[unrestricted_cell_idx].insert(
            restricted_cell_idx);
      }
    } else {
      bool cell_totally_in_shell = (birth_face_indices.minCoeff() >= shell_mesh.NumFaces());
      if (keep_number_flag) {
        map_restricted_cell_idx_to_is_totally_in_shell[unrestricted_cell_idx] =
            cell_totally_in_shell;

        if (cell_totally_in_shell) {
          new_voronoi_cell = unrestricted_cell;
        } else {
          new_voronoi_cell.map_triangle_to_polygon_idx.resize(
              new_voronoi_cell.cell_triangle_mesh.NumFaces());
          auto &new_polygons = new_voronoi_cell.polyhedron.polygons;

          new_polygons.resize(unrestricted_cell.polyhedron.polygons.size() +
                              (cell_totally_in_shell ? 0 : 1));

          // auto t1 = std::chrono::steady_clock::now();
          for (index_t face_idx = 0; face_idx < new_voronoi_cell.cell_triangle_mesh.NumFaces();
               ++face_idx) {
            if (birth_face_indices(face_idx) < shell_mesh.NumFaces()) {  // from shell mesh
              new_polygons[0].triangle_face_indices_in_cell.push_back(face_idx);
              new_voronoi_cell.map_triangle_to_polygon_idx[face_idx] = 0;
            } else {
              index_t new_polygon_idx =
                  unrestricted_cell.map_triangle_to_polygon_idx[birth_face_indices(face_idx) -
                                                                shell_mesh.NumFaces()];
              new_polygon_idx += cell_totally_in_shell ? 0 : 1;
              new_polygons[new_polygon_idx].triangle_face_indices_in_cell.push_back(face_idx);
              new_voronoi_cell.map_triangle_to_polygon_idx[face_idx] = new_polygon_idx;
            }
          }
        }  // end if totally in shell

        for (auto &polygon : new_voronoi_cell.polyhedron.polygons) {
          if (polygon.triangle_face_indices_in_cell.empty()) continue;
          polygon.mat_triangle_faces =
              new_voronoi_mesh.mat_faces(polygon.triangle_face_indices_in_cell, Eigen::all);
        }

        index_t restricted_cell_idx = restricted_voronoi_diagram.cells.size();
        restricted_voronoi_diagram.cells.push_back(new_voronoi_cell);
        restricted_voronoi_diagram.map_cell_to_neighbor_indices.push_back(
            unrestricted_voronoi_diagram.map_cell_to_neighbor_indices[unrestricted_cell_idx]);
        map_unrestricted_cell_idx_to_restricted_cell_idx[unrestricted_cell_idx].insert(
            restricted_cell_idx);
        // auto t2 = std::chrono::steady_clock::now();

        // total_seconds["bool"] += std::chrono::duration<double>(t2 - t1).count();
      } else {  // Don't keep cell number
        Eigen::VectorXi face_component_indices;
        igl::facet_components(new_voronoi_mesh.mat_faces, face_component_indices);
        const size_t num_components = face_component_indices.maxCoeff() + 1;

        for (index_t component_idx = 0; component_idx < num_components; ++component_idx) {
          size_t num_faces_in_component = (face_component_indices.array() == component_idx).count();
          VoronoiCell component_voronoi_cell;
          MatMesh3 component_mesh;
          auto &component_polygons       = component_voronoi_cell.polyhedron.polygons;
          component_mesh.mat_coordinates = new_voronoi_mesh.mat_coordinates;
          component_mesh.mat_faces.resize(num_faces_in_component, 3);
          component_voronoi_cell.map_triangle_to_polygon_idx.resize(num_faces_in_component);
          component_polygons.resize(unrestricted_cell.polyhedron.polygons.size() +
                                    (cell_totally_in_shell ? 0 : 1));
          for (index_t face_idx = 0, component_face_idx = 0; face_idx < new_voronoi_mesh.NumFaces();
               ++face_idx) {
            if (face_component_indices(face_idx) != component_idx) continue;
            component_mesh.mat_faces.row(component_face_idx) =
                new_voronoi_mesh.mat_faces.row(face_idx);

            if (birth_face_indices(face_idx) < shell_mesh.NumFaces()) {  // from shell mesh
              component_polygons[0].triangle_face_indices_in_cell.push_back(component_face_idx);
              component_voronoi_cell.map_triangle_to_polygon_idx[component_face_idx] = 0;
            } else {
              index_t new_polygon_idx =
                  unrestricted_cell.map_triangle_to_polygon_idx[birth_face_indices(face_idx) -
                                                                shell_mesh.NumFaces()];
              new_polygon_idx += cell_totally_in_shell ? 0 : 1;
              component_polygons[new_polygon_idx].triangle_face_indices_in_cell.push_back(
                  component_face_idx);
              component_voronoi_cell.map_triangle_to_polygon_idx[component_face_idx] =
                  new_polygon_idx;
            }
            component_face_idx++;
          }
          component_voronoi_cell.seed = unrestricted_cell.seed;
          component_voronoi_cell.cell_triangle_mesh =
              sha::RemoveUnreferencedVertices(component_mesh);

          for (auto &polygon : component_polygons) {
            if (polygon.triangle_face_indices_in_cell.empty()) continue;
            polygon.mat_triangle_faces = component_voronoi_cell.cell_triangle_mesh.mat_faces(
                polygon.triangle_face_indices_in_cell, Eigen::all);
          }

          index_t restricted_cell_idx = restricted_voronoi_diagram.cells.size();
          restricted_voronoi_diagram.cells.push_back(component_voronoi_cell);
          map_unrestricted_cell_idx_to_restricted_cell_idx[unrestricted_cell_idx].insert(
              restricted_cell_idx);
          restricted_voronoi_diagram.map_cell_to_neighbor_indices.push_back(
              unrestricted_voronoi_diagram.map_cell_to_neighbor_indices[unrestricted_cell_idx]);

          map_restricted_cell_idx_to_is_totally_in_shell[restricted_cell_idx] =
              cell_totally_in_shell;
        }  // for component_idx
      }    // keep_number_flag
    }      // new_voronoi_mesh.NumVertices() == 0
  }
  // );

  log::info("Boolean done.");
  log::info("{}mm", total_mmsecs);

  auto &map_cell_to_neighbor_indices = restricted_voronoi_diagram.map_cell_to_neighbor_indices;
  const size_t num_restricted_cells  = restricted_voronoi_diagram.cells.size();
  for (index_t unrestricted_cell_idx = 0; unrestricted_cell_idx < num_restricted_cells;
       ++unrestricted_cell_idx) {
    const auto &old_neighbor_indices = map_cell_to_neighbor_indices.at(unrestricted_cell_idx);
    map_cell_to_neighbor_indices[unrestricted_cell_idx].clear();
    for (auto old_neighbor_cell_idx : old_neighbor_indices) {
      for (auto new_neighbor_cell_idx :
           map_unrestricted_cell_idx_to_restricted_cell_idx[old_neighbor_cell_idx]) {
        map_cell_to_neighbor_indices[unrestricted_cell_idx].insert(new_neighbor_cell_idx);
      }
    }
  }

  auto RebuildIndexingOfMapCellTriangleToPolygon = [&](VoronoiCell &voronoi_cell) {
    auto &polygons = voronoi_cell.polyhedron.polygons;
    for (size_t polygon_idx = 0; polygon_idx < polygons.size(); polygon_idx++) {
      auto &polygon = polygons[polygon_idx];
      for (index_t face_idx : polygon.triangle_face_indices_in_cell) {
        voronoi_cell.map_triangle_to_polygon_idx[face_idx] = polygon_idx;
      }
    }
  };

  log::info("Removing Empty Polygons From VoronoiCell");

  boost::for_each(restricted_voronoi_diagram.cells, RemoveEmptyPolygonsFromVoronoiCell);

  log::info("Removed");

  log::info("Separating Polygon If Not Connected");

  boost::for_each(restricted_voronoi_diagram.cells, [&](VoronoiCell &voronoi_cell) {
    auto old_polygons  = voronoi_cell.polyhedron.polygons;  // copy
    auto &new_polygons = voronoi_cell.polyhedron.polygons;  // ref
    new_polygons.clear();
    for (auto &old_polygon : old_polygons) {
      auto &&separate_polygons = SeparatePolygonIfNotConnected(old_polygon, voronoi_cell);
      boost::copy(separate_polygons, std::back_inserter(new_polygons));
    }
  });

  log::info("Separated");

  log::info("Separating Polygon If Edge Is Sharp");
  oneapi::tbb::parallel_for_each(
      restricted_voronoi_diagram.cells | boost::adaptors::indexed(0), [&](auto &cell_iterator) {
        VoronoiCell &voronoi_cell = cell_iterator.value();
        index_t cell_idx          = cell_iterator.index();
        if (map_restricted_cell_idx_to_is_totally_in_shell[cell_idx]) {
          return;
        }
        SurfaceMesh3 cell_mesh3 =
            sha::CreateSurfaceMesh3FromMatMesh3(voronoi_cell.cell_triangle_mesh);
        auto old_polygons  = voronoi_cell.polyhedron.polygons;  // copy
        auto &new_polygons = voronoi_cell.polyhedron.polygons;  // ref
        new_polygons.clear();
        for (auto &old_polygon : old_polygons) {
          auto separate_polygons =
              SeparatePolygonIfEdgeIsSharp(old_polygon, voronoi_cell, cell_mesh3, sharp_angle);
          boost::copy(separate_polygons, std::back_inserter(new_polygons));
        }
      });

  log::info("Separated");

  log::info("Rebuilding Indexing Of Map Cell Triangle To Polygon");

  boost::for_each(restricted_voronoi_diagram.cells, RebuildIndexingOfMapCellTriangleToPolygon);

  log::info("Rebuilt");

  log::info("Computing Loop Vertex By Its Triangles");

  oneapi::tbb::parallel_for_each(
      restricted_voronoi_diagram.cells | boost::adaptors::indexed(0), [&](auto &cell_iterator) {
        VoronoiCell &voronoi_cell = cell_iterator.value();
        index_t cell_idx          = cell_iterator.index();
        if (!map_restricted_cell_idx_to_is_totally_in_shell[cell_idx]) {
          boost::for_each(voronoi_cell.polyhedron.polygons, ComputeLoopVertexByItsTriangles);
        }
      });

  log::info("Computed");

  return restricted_voronoi_diagram;
}

auto SeparatePolygonIfNotConnected(const PolygonFace &polygon, const VoronoiCell &voronoi_cell)
    -> std::vector<PolygonFace> {
  const auto &mat_faces_in_polygon = polygon.mat_triangle_faces;
  Eigen::VectorXi polygon_face_component_indices;
  igl::facet_components(mat_faces_in_polygon, polygon_face_component_indices);
  const size_t num_components = polygon_face_component_indices.maxCoeff() + 1;
  if (num_components == 1) return {polygon};

  std::vector<PolygonFace> component_polygons(num_components);
  for (index_t idx = 0; idx < polygon_face_component_indices.size(); ++idx) {
    index_t face_idx      = polygon.triangle_face_indices_in_cell[idx];
    index_t component_idx = polygon_face_component_indices[idx];
    component_polygons[component_idx].triangle_face_indices_in_cell.push_back(face_idx);
  }

  for (auto &polygon : component_polygons) {
    if (polygon.triangle_face_indices_in_cell.empty()) continue;
    polygon.mat_triangle_faces = voronoi_cell.cell_triangle_mesh.mat_faces(
        polygon.triangle_face_indices_in_cell, Eigen::all);
  }

  return component_polygons;
}

auto SeparatePolygonIfEdgeIsSharp(const PolygonFace &polygon, const VoronoiCell &voronoi_cell,
                                  const SurfaceMesh3 &voronoi_mesh3, double angle)
    -> std::vector<PolygonFace> {
  std::vector<PolygonFace> component_polygons;
  const size_t num_polygon_faces = polygon.triangle_face_indices_in_cell.size();
  using Edge                     = std::pair<index_t, index_t>;
  std::set<Edge> edges;
  Eigen::MatrixXi mat_sharp_edges;
  igl::sharp_edges(voronoi_cell.cell_triangle_mesh.mat_coordinates, polygon.mat_triangle_faces,
                   angle / 180.0 * igl::PI, mat_sharp_edges);
  for (index_t idx = 0; idx < mat_sharp_edges.rows(); ++idx) {
    edges.emplace(std::min(mat_sharp_edges(idx, 0), mat_sharp_edges(idx, 1)),
                  std::max(mat_sharp_edges(idx, 0), mat_sharp_edges(idx, 1)));
  }

  std::vector<bool> face_visited_flags(num_polygon_faces, false);

  for (index_t idx = 0; idx < num_polygon_faces; ++idx) {
    if (face_visited_flags[idx]) continue;
    PolygonFace component_polygon;
    std::queue<index_t> breadth_first_search_queue;
    breadth_first_search_queue.push(idx);

    while (!breadth_first_search_queue.empty()) {
      index_t curr_idx = breadth_first_search_queue.front();
      index_t face_idx = polygon.triangle_face_indices_in_cell[curr_idx];
      breadth_first_search_queue.pop();
      if (face_visited_flags[curr_idx]) continue;

      component_polygon.triangle_face_indices_in_cell.push_back(face_idx);
      face_visited_flags[curr_idx] = true;

      auto face_halfedge = voronoi_mesh3.halfedge(SurfaceMesh3::Face_index(face_idx));
      for (auto halfedge : voronoi_mesh3.halfedges_around_face(face_halfedge)) {
        auto edge_handle          = voronoi_mesh3.edge(halfedge);
        index_t neighbor_face_idx = voronoi_mesh3.face(voronoi_mesh3.opposite(halfedge));
        if (neighbor_face_idx != -1 && neighbor_face_idx != face_idx) {
          auto find_iterator =
              boost::find(polygon.triangle_face_indices_in_cell, neighbor_face_idx);
          if (find_iterator != polygon.triangle_face_indices_in_cell.end()) {
            Edge edge = {voronoi_mesh3.vertex(edge_handle, 0).idx(),
                         voronoi_mesh3.vertex(edge_handle, 1).idx()};
            if (edge.first > edge.second) {
              std::swap(edge.first, edge.second);
            }
            if (edges.find(edge) == edges.end()) {
              breadth_first_search_queue.push(find_iterator -
                                              polygon.triangle_face_indices_in_cell.begin());
            }
          }  // neighbor_face exists in polygon
        }    // neighbor_face is valid
      }      // for neighbors
    }        // bfs
    component_polygon.mat_triangle_faces = voronoi_cell.cell_triangle_mesh.mat_faces(
        component_polygon.triangle_face_indices_in_cell, Eigen::all);
    component_polygons.push_back(component_polygon);
  }  // for every polygon
  return component_polygons;
}

void RemoveEmptyPolygonsFromVoronoiCell(VoronoiCell &voronoi_cell) {
  /**
  const size_t num_old_polygons = voronoi_cell.polyhedron.polygons.size();
  if (num_old_polygons == 0) return;
  auto old_map_triangle_to_polygon_idx = voronoi_cell.map_triangle_to_polygon_idx;
  auto old_polygons                    = voronoi_cell.polyhedron.polygons;

  voronoi_cell.polyhedron.polygons.clear();

  std::map<index_t, index_t> map_old_idx_to_new_idx_for_polygon;
  for (index_t old_polygon_idx = 0, new_polygon_idx = 0; old_polygon_idx < num_old_polygons;
       ++old_polygon_idx) {
    if (old_polygons[old_polygon_idx].triangle_face_indices_in_cell.empty()) continue;
    map_old_idx_to_new_idx_for_polygon[old_polygon_idx] = new_polygon_idx;
    voronoi_cell.polyhedron.polygons.push_back(old_polygons[old_polygon_idx]);
    new_polygon_idx++;
  }

  for (index_t idx = 0; idx < voronoi_cell.map_triangle_to_polygon_idx.size(); ++idx) {
    voronoi_cell.map_triangle_to_polygon_idx[idx] =
        map_old_idx_to_new_idx_for_polygon[old_map_triangle_to_polygon_idx[idx]];
  }
  **/

  const size_t num_old_polygons = voronoi_cell.polyhedron.polygons.size();
  if (num_old_polygons == 0) return;
  auto old_polygons = voronoi_cell.polyhedron.polygons;
  voronoi_cell.polyhedron.polygons.clear();
  for (index_t old_polygon_idx = 0; old_polygon_idx < num_old_polygons; ++old_polygon_idx) {
    if (old_polygons[old_polygon_idx].triangle_face_indices_in_cell.empty()) continue;
    voronoi_cell.polyhedron.polygons.push_back(old_polygons[old_polygon_idx]);
  }
}

void ComputeLoopVertexByItsTriangles(PolygonFace &polygon) {
  if (polygon.mat_triangle_faces.rows() == 0) return;
  polygon.boundary_vtx_loops.clear();
  igl::boundary_loop(polygon.mat_triangle_faces, polygon.boundary_vtx_loops);
}

double distance_between_two_edges(const MatMesh2 edge_mesh_1, const MatMesh2 edge_mesh_2) {
  double dis1, dis2;
  {
    Eigen::MatrixXd sqrD;
    Eigen::MatrixXi I;
    Eigen::MatrixXd C;
    igl::point_mesh_squared_distance(edge_mesh_2.mat_coordinates, edge_mesh_1.mat_coordinates,
                                     edge_mesh_1.mat_beams, sqrD, I, C);
    dis1 = sqrD.sum() / sqrD.size();
  }
  {
    Eigen::MatrixXd sqrD;
    Eigen::MatrixXi I;
    Eigen::MatrixXd C;
    igl::point_mesh_squared_distance(edge_mesh_1.mat_coordinates, edge_mesh_2.mat_coordinates,
                                     edge_mesh_2.mat_beams, sqrD, I, C);
    dis2 = sqrD.sum() / sqrD.size();
  }
  return std::max(dis1, dis2);
}

struct VoronoiCellChain {
  std::vector<Eigen::Vector3d> path;

  MatMesh2 edge_mesh;

  std::vector<index_t> cell_indices;

 public:
  bool equals(const std::vector<Eigen::Vector3d> &other, const MatMesh2 &other_mesh) {
    constexpr double kEpsilon = 1e-9;
    if (((path.front() - other.front()).norm() < kEpsilon) &&
            ((path.back() - other.back()).norm() < kEpsilon) ||
        ((path.front() - other.back()).norm() < kEpsilon) &&
            ((path.back() - other.front()).norm() < kEpsilon)) {
      // iff two points coincides
      return distance_between_two_edges(edge_mesh, other_mesh) < kEpsilon;
    } else {
      return false;
    }
  }

 public:
  Eigen::AlignedBox3d bbox;
};

using cell_edge_ptr = std::shared_ptr<VoronoiCellChain>;

struct SharedBoundaryChain {
  std::vector<index_t> boundary_vertices;
  index_t polygon_idx_0 = -1;
  index_t polygon_idx_1 = -1;
};

using SharedBoundaryPtr = std::shared_ptr<SharedBoundaryChain>;

auto ComputeBoundaryChainsOfCell(const VoronoiCell &cell) -> std::vector<SharedBoundaryPtr> {
  using Edge       = std::pair<index_t, index_t>;
  using PatchEdges = std::vector<Edge>;

  const size_t num_polygons = cell.polyhedron.polygons.size();

  std::vector<std::vector<SharedBoundaryPtr>> shared_boundaries_of_polygons(num_polygons);
  std::vector<std::vector<Edge>> boundary_edges_of_polygons(num_polygons);

  std::vector<SharedBoundaryPtr> cell_boundaries;
  std::set<index_t> corner_vertex_indices;
  std::vector<PatchEdges> patches_edges(num_polygons);
  // find corner  visit more than twice
  std::map<int, int> visited;

  for (int polygon_idx = 0; polygon_idx < num_polygons; ++polygon_idx) {
    auto &patch = cell.polyhedron.polygons[polygon_idx];
    PatchEdges polygon_edges;
    for (auto &boundary_loop : patch.boundary_vtx_loops) {
      for (int i = 0; i < boundary_loop.size(); ++i) {
        index_t a = boundary_loop[i];
        index_t b = boundary_loop[(i + 1) % boundary_loop.size()];
        visited[a] += 1;
        if (visited[a] > 2) {
          corner_vertex_indices.insert(a);
        }
        polygon_edges.emplace_back(a, b);
      }
    }
    patches_edges[polygon_idx] = polygon_edges;
  }
  // divide edges via corner
  auto find_or_create_an_edge = [&](std::vector<index_t> vs) -> SharedBoundaryPtr {
    auto fd = boost::find_if(cell_boundaries, [&](const SharedBoundaryPtr &boundary) {
      return boost::equal(boundary->boundary_vertices, vs) ||
             boost::equal(boundary->boundary_vertices, boost::reversed_range(vs));
    });
    if (fd == cell_boundaries.end()) {  // create
      SharedBoundaryPtr new_e  = std::make_shared<SharedBoundaryChain>();
      new_e->boundary_vertices = vs;
      new_e->polygon_idx_0     = -1;
      new_e->polygon_idx_1     = -1;
      cell_boundaries.push_back(new_e);
      return new_e;
    } else {
      return *fd;
    }
  };

  for (int polygon_idx = 0; polygon_idx < num_polygons; ++polygon_idx) {
    auto &polygon                      = cell.polyhedron.polygons[polygon_idx];
    const size_t num_loops             = polygon.boundary_vtx_loops.size();
    auto &shared_boundaries_of_polygon = shared_boundaries_of_polygons[polygon_idx];
    auto &boundary_edges_of_polygon    = boundary_edges_of_polygons[polygon_idx];

    std::vector<std::vector<int>> corners_of_loops(num_loops);
    for (auto &&[loop_idx, patch_loop] : polygon.boundary_vtx_loops | boost::adaptors::indexed(0)) {
      for (auto v : patch_loop) {
        if (corner_vertex_indices.find(v) != corner_vertex_indices.end()) {
          corners_of_loops[loop_idx].push_back(v);
        }
      }
    }

    for (auto &&[loop_idx, corners_of_loop] : corners_of_loops | boost::adaptors::indexed(0)) {
      auto &boundary_vtx_loop = polygon.boundary_vtx_loops[loop_idx];
      for (index_t idx = 0; idx < corners_of_loop.size(); ++idx) {
        index_t corner_vtx_0 = corners_of_loop[idx];
        index_t corner_vtx_1 = corners_of_loop[(idx + 1) % corners_of_loop.size()];

        std::vector<index_t> vs;  // truncate
        int ai = std::find(boundary_vtx_loop.begin(), boundary_vtx_loop.end(), corner_vtx_0) -
                 boundary_vtx_loop.begin();
        int bi = std::find(boundary_vtx_loop.begin(), boundary_vtx_loop.end(), corner_vtx_1) -
                 boundary_vtx_loop.begin();
        if (ai < bi) {
          for (int j = ai; j <= bi; ++j) {
            vs.push_back(boundary_vtx_loop[j]);
          }
        } else {
          for (int j = ai; j < boundary_vtx_loop.size(); ++j) {
            vs.push_back(boundary_vtx_loop[j]);
          }
          for (int j = 0; j <= bi; ++j) {
            vs.push_back(boundary_vtx_loop[j]);
          }
        }

        auto e = find_or_create_an_edge(vs);

        if (e->polygon_idx_0 == -1) {
          e->polygon_idx_0 = polygon_idx;
          shared_boundaries_of_polygon.push_back(e);
          boundary_edges_of_polygon.emplace_back(corner_vtx_0, corner_vtx_1);
        } else if (e->polygon_idx_1 == -1) {
          e->polygon_idx_1 = polygon_idx;
          shared_boundaries_of_polygon.push_back(e);
          boundary_edges_of_polygon.emplace_back(corner_vtx_0, corner_vtx_1);
        } else {
          log::info("Non manifold");
          continue;
          // Terminate("Non manifold");
        }
      }
    }
  }
  return cell_boundaries;
}

auto ComputeRelatedEdgesFromVoronoiDiagram(const VoronoiDiagram &voronoi,
                                           const Eigen::AlignedBox3d &domain) -> MatMesh2 {
  std::vector<std::set<index_t>> map_beam_idx_to_cell_indices;
  return ComputeRelatedEdgesFromVoronoiDiagram(voronoi, domain, map_beam_idx_to_cell_indices);
}

auto ComputeRelatedEdgesFromVoronoiDiagram(
    const VoronoiDiagram &voronoi, const Eigen::AlignedBox3d &domain,
    std::vector<std::set<index_t>> &map_beam_idx_to_cell_indices) -> MatMesh2 {
  map_beam_idx_to_cell_indices.clear();
  const size_t num_cells = voronoi.cells.size();
  std::vector<std::vector<SharedBoundaryPtr>> boundary_chains_of_cells;

  // log::info("Computing BoundaryChainsOfCell");
  boost::transform(voronoi.cells, std::back_inserter(boundary_chains_of_cells),
                   ComputeBoundaryChainsOfCell);

  // log::info("Computed");

  // log::info("Building tree");
  DynamicBoundingBoxOctree<cell_edge_ptr> octree(domain, 6);

  std::vector<cell_edge_ptr> &cells_edges = octree.data_vector;

  // log::info("Built");

  double total_seconds = 0;

  auto FindOrCreateCellEdge = [&](const std::vector<Eigen::Vector3d> &path) -> cell_edge_ptr {
    MatMesh2 edge_mesh;
    edge_mesh.mat_coordinates.resize(path.size(), 3);
    for (int i = 0; i < path.size(); ++i) {
      edge_mesh.mat_coordinates.row(i) << path[i].x(), path[i].y(), path[i].z();
    }

    edge_mesh.mat_beams.resize(edge_mesh.NumVertices() - 1, 2);
    for (int i = 0; i < edge_mesh.NumVertices() - 1; ++i) {
      edge_mesh.mat_beams.row(i) << i, i + 1;
    }

    Eigen::AlignedBox3d edge_boundingbox(edge_mesh.mat_coordinates.colwise().minCoeff(),
                                         edge_mesh.mat_coordinates.colwise().maxCoeff());

    // auto t1           = std::chrono::steady_clock::now();
    auto data_indices = octree.FindDataIndicesByBoundingBox(edge_boundingbox);

    cell_edge_ptr find_edge_ptr = nullptr;
    for (index_t data_idx : data_indices) {
      if (cells_edges.at(data_idx)->equals(path, edge_mesh)) {
        find_edge_ptr = cells_edges.at(data_idx);
        break;
      }
    }
    // auto t2 = std::chrono::steady_clock::now();
    // total_seconds += std::chrono::duration<double>(t2 - t1).count();

    if (find_edge_ptr == nullptr) {  // create
      auto new_e       = std::make_shared<VoronoiCellChain>();
      new_e->path      = path;
      new_e->edge_mesh = edge_mesh;
      octree.InsertData(edge_boundingbox, new_e);
      return new_e;
    } else {
      return find_edge_ptr;
    }

    // auto t1 = std::chrono::steady_clock::now();
    // auto fd = boost::find_if(cells_edges,
    //                          [&](const cell_edge_ptr &e) { return e->equals(path, edge_mesh); });
    // auto t2 = std::chrono::steady_clock::now();

    // total_seconds += std::chrono::duration<double>(t2 - t1).count();
    // if (fd == cells_edges.end()) {  // create
    //   auto new_e       = std::make_shared<VoronoiCellChain>();
    //   new_e->path      = path;
    //   new_e->edge_mesh = edge_mesh;
    //   cells_edges.push_back(new_e);
    //   return new_e;
    // } else {
    //   return *fd;
    // }
  };

  // log::info("FindOrCreateCellEdge");

  for (index_t cell_idx = 0; cell_idx < num_cells; ++cell_idx) {
    auto &cell          = voronoi.cells.at(cell_idx);
    auto &shared_chains = boundary_chains_of_cells.at(cell_idx);
    for (const auto &chain : shared_chains) {
      std::vector<Eigen::Vector3d> edge_path(chain->boundary_vertices.size());
      for (index_t idx = 0; idx < chain->boundary_vertices.size(); ++idx) {
        index_t vertex_idx = chain->boundary_vertices[idx];
        edge_path[idx]     = cell.cell_triangle_mesh.mat_coordinates.row(vertex_idx);
      }
      auto cell_edge = FindOrCreateCellEdge(edge_path);
      cell_edge->cell_indices.push_back(cell_idx);
    }
  }
  // log::info("FindOrCreateCellEdge Over");

  const size_t num_beams = boost::accumulate(
      cells_edges, 0, [](size_t accumulated_value, cell_edge_ptr chain) -> size_t {
        return accumulated_value + chain->path.size() - 1;
      });

  MatMesh2 beam_mesh;
  beam_mesh.mat_coordinates.resize(num_beams * 2, 3);
  beam_mesh.mat_beams.resize(num_beams, 2);
  map_beam_idx_to_cell_indices.resize(num_beams);

  index_t beam_idx = 0;
  for (index_t chain_idx = 0; chain_idx < cells_edges.size(); chain_idx++) {
    auto &chain = cells_edges[chain_idx];
    for (index_t idx = 0; idx < chain->path.size() - 1; ++idx, ++beam_idx) {
      beam_mesh.mat_coordinates.row(beam_idx * 2 + 0) = chain->path[idx];
      beam_mesh.mat_coordinates.row(beam_idx * 2 + 1) = chain->path[idx + 1];
      beam_mesh.mat_beams.row(beam_idx) << beam_idx * 2, beam_idx * 2 + 1;
      boost::copy(chain->cell_indices,
                  std::inserter(map_beam_idx_to_cell_indices.at(beam_idx),
                                map_beam_idx_to_cell_indices.at(beam_idx).begin()));
    }
  }
  // log::info("total_fd_seconds: {}", total_seconds);
  return beam_mesh;
}
}  // namespace sha
}  // namespace da
