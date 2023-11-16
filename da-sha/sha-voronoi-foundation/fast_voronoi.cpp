#include "fast_voronoi.h"

#include <vector>

#include <oneapi/tbb.h>

#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_tetrahedralize.h>
#include <geogram/voronoi/RVD.h>
#include <geogram/voronoi/RVD_callback.h>
#include <geogram/voronoi/RVD_mesh_builder.h>
#include <geogram/voronoi/convex_cell.h>

#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/combine.hpp>
#include <boost/range/numeric.hpp>

#include "utility.h"
#include "voronoi.h"

namespace GEO {
class RVDCellTraverser : public RVDPolyhedronCallback {
 public:
  /**
   * \brief RVDCellTraverser constructor.
   * \param[out] output_mesh a reference to the generated mesh
   */
  RVDCellTraverser(da::sha::VoronoiDiagram& voronoi, da::size_t num_seeds)
      : voronoi_(voronoi), num_seeds_(num_seeds) {
    my_vertex_map_ = nullptr;
    // If set, then only one polyhedron per (connected component of) restricted Voronoi
    // cell is generated.
    set_simplify_internal_tet_facets(true);

    // If set, then only one polygon per Voronoi facet is generated.
    set_simplify_voronoi_facets(true);

    // If set, then the intersection between a Voronoi cell and the boundary surface is
    // replaced with a single polygon whenever possible (i.e. when its topology is a
    // disk and when it has at least 3 corners).
    set_simplify_boundary_facets(false);

    // If set, then the intersections are available as Mesh objects through the function
    // process_polyhedron_mesh(). Note that this is implied by simplify_voronoi_facets
    // or simplify_boundary.
    // set_use_mesh(false);
    voronoi_.cells.resize(num_seeds);
    voronoi_.map_cell_to_neighbor_indices.resize(num_seeds);
  }

  ~RVDCellTraverser() override {
    delete my_vertex_map_;
    my_vertex_map_ = nullptr;
  }

  /**
   * \brief Called at the beginning of RVD traversal.
   */
  void begin() override { RVDPolyhedronCallback::begin(); }

  /**
   * \brief Called at the end of RVD traversal.
   */
  void end() override { RVDPolyhedronCallback::end(); }

  /**
   * \brief Called at the beginning of each RVD polyhedron.
   * \param[in] seed , tetrahedron the (seed,tetrahedron) pair that
   *  defines the RVD polyhedron, as the intersection between the Voronoi
   *  cell of the seed and the tetrahedron.
   */
  void begin_polyhedron(index_t seed, index_t tetrahedron) override {
    geo_argused(tetrahedron);
    geo_argused(seed);

    //   The RVDVertexMap is used to map the symbolic representation of vertices
    // to indices. Here we reset indexing for each new cell, so that vertices shared
    // by the faces of two different cells will be duplicated. We do that because
    // we construct the boundary of the cells in a surfacic mesh (for visualization
    // purposes). Client code that has a data structure for polyhedral volumetric mesh
    // will not want to reset indexing (and will comment-out the following three lines).
    // It will also construct the RVDVertexMap in the constructor.

    delete my_vertex_map_;
    my_vertex_map_ = new RVDVertexMap;
    my_vertex_map_->set_first_vertex_index(0);
    current_cell_mesh_.clear();
    current_cell_mesh_.vertices.set_dimension(3);
    cell_totally_in_shell_ = true;
    current_cell_face_opposites_.clear();
    current_cell_boundary_loop_.clear();
  }

  /**
   * \brief Called at the beginning of each RVD polyhedron.
   * \param[in] facet_seed if the facet is on a Voronoi bisector,
   *  the index of the Voronoi seed on the other side of the bisector,
   *  else index_t(-1)
   * \param[in] facet_tet if the facet is on a tethedral facet, then
   *  the index of the tetrahedron on the other side, else index_t(-1)
   */
  void begin_facet(index_t facet_seed, index_t facet_tet) override {
    geo_argused(facet_seed);
    geo_argused(facet_tet);
    face_opposite_seed_ = facet_seed;
    if (index_t(-1) == facet_seed) {
      face_opposite_seed_    = -1;
      cell_totally_in_shell_ = false;
    }
    current_facet_.resize(0);
  }

  void vertex(const double* geometry, const GEOGen::SymbolicVertex& symb) override {
    // Find the index of the vertex associated with its symbolic representation.
    index_t vid = my_vertex_map_->find_or_create_vertex(seed(), symb);

    // If the vertex does not exist in the mesh, create it.
    if (vid >= current_cell_mesh_.vertices.nb()) {
      current_cell_mesh_.vertices.create_vertex(geometry);
    }

    // Memorize the current facet.
    current_facet_.push_back(vid);
  }

  void end_facet() override {
    // Create the facet from the memorized indices.
    index_t f = current_cell_mesh_.facets.nb();
    for (int idx = 2; idx < current_facet_.size(); ++idx) {
      current_cell_mesh_.facets.create_triangle(current_facet_[0], current_facet_[idx - 1],
                                                current_facet_[idx]);
      current_cell_face_opposites_.push_back(face_opposite_seed_);
    }
    if (face_opposite_seed_ != -1) {
      boost::transform(current_facet_,
                       std::back_inserter(current_cell_boundary_loop_[face_opposite_seed_]),
                       [](GEO::index_t idx) -> da::index_t { return idx; });
    }
  }

  void end_polyhedron() override {
    current_cell_mesh_.facets.connect();
    map_restricted_cell_idx_to_is_totally_in_shell_[seed()] = cell_totally_in_shell_;
    da::sha::VoronoiCell& new_voronoi_cell                  = voronoi_.cells[seed()];
    auto& new_voronoi_mesh = voronoi_.cells[seed()].cell_triangle_mesh;
    da::sha::ConvertGeoMeshToMatmesh3(current_cell_mesh_, new_voronoi_mesh);
    auto& new_polygons = new_voronoi_cell.polyhedron.polygons;
    // -1 (means surface) must be first
    std::set<int> unique_face_opposites(current_cell_face_opposites_.begin(),
                                        current_cell_face_opposites_.end());
    std::vector<int> unique_face_opposite_array(unique_face_opposites.begin(),
                                                unique_face_opposites.end());
    std::map<int, int> map_opposite_to_polygon_idx;
    for (int idx = 0; idx < unique_face_opposite_array.size(); ++idx) {
      map_opposite_to_polygon_idx[unique_face_opposite_array[idx]] = idx;
      if (unique_face_opposite_array[idx] != -1) {
        voronoi_.map_cell_to_neighbor_indices[seed()].insert(unique_face_opposite_array[idx]);
      }
    }

    new_polygons.resize(unique_face_opposites.size());
    new_voronoi_cell.map_triangle_to_polygon_idx.resize(new_voronoi_mesh.NumFaces());
    for (int face_idx = 0; face_idx < new_voronoi_cell.cell_triangle_mesh.NumFaces(); ++face_idx) {
      int polygon_idx = map_opposite_to_polygon_idx.at(current_cell_face_opposites_[face_idx]);
      new_polygons.at(polygon_idx).triangle_face_indices_in_cell.push_back(face_idx);
      new_voronoi_cell.map_triangle_to_polygon_idx.at(face_idx) = polygon_idx;
      if (current_cell_face_opposites_[face_idx] != -1) {
        new_polygons.at(polygon_idx).boundary_vtx_loops = {
            current_cell_boundary_loop_[current_cell_face_opposites_[face_idx]]};
      }
    }

    for (auto& polygon : new_polygons) {
      if (polygon.triangle_face_indices_in_cell.empty()) continue;
      polygon.mat_triangle_faces =
          new_voronoi_mesh.mat_faces(polygon.triangle_face_indices_in_cell, Eigen::all);
    }
  }

 public:
  vector<index_t> current_facet_;
  RVDVertexMap* my_vertex_map_;

  std::vector<int> current_cell_face_opposites_;

  int face_opposite_seed_;

  Mesh current_cell_mesh_;
  std::map<int, da::sha::BoundaryVertexLoop> current_cell_boundary_loop_;
  bool cell_totally_in_shell_;

 public:
  da::sha::VoronoiDiagram& voronoi_;
  da::size_t num_seeds_;
  std::map<index_t, bool> map_restricted_cell_idx_to_is_totally_in_shell_;
};
}  // namespace GEO

namespace da {
namespace sha {

auto FastCreateRestrictedVoronoiDiagramFromMesh(GEO::Mesh& boundary_mesh,
                                                const Eigen::MatrixXd& mat_seeds,
                                                double sharp_angle) -> VoronoiDiagram {
  const size_t num_voronoi_seeds = mat_seeds.rows();

  GEO::initialize();

  GEO::CmdLine::import_arg_group("standard");
  GEO::CmdLine::import_arg_group("algo");
  // GEO::CmdLine::declare_arg("log:quiet", true, "Turns logging on/off");

  std::vector<double> points_array(num_voronoi_seeds * 3);
  for (int idx = 0; idx < num_voronoi_seeds; ++idx) {
    points_array[idx * 3]     = mat_seeds(idx, 0);
    points_array[idx * 3 + 1] = mat_seeds(idx, 1);
    points_array[idx * 3 + 2] = mat_seeds(idx, 2);
  }

  GEO::mesh_tetrahedralize(boundary_mesh);
  GEO::Delaunay_var delaunay = GEO::Delaunay::create(3);
  GEO::RestrictedVoronoiDiagram_var RVD =
      GEO::RestrictedVoronoiDiagram::create(delaunay, &boundary_mesh);
  delaunay->set_vertices(num_voronoi_seeds, points_array.data());

  RVD->set_volumetric(true);

  VoronoiDiagram voronoi;
  GEO::RVDCellTraverser traveraer(voronoi, num_voronoi_seeds);
  // bug while testing femur model
  RVD->for_each_polyhedron(traveraer);
  // bug
  for (index_t cell_idx = 0; cell_idx < num_voronoi_seeds; ++cell_idx) {
    voronoi.cells.at(cell_idx).seed = mat_seeds.row(cell_idx);
  }

  auto& map_restricted_cell_idx_to_is_totally_in_shell =
      traveraer.map_restricted_cell_idx_to_is_totally_in_shell_;
  log::info("Separating Polygon If Edge Is Sharp");
  oneapi::tbb::parallel_for_each(
      voronoi.cells | boost::adaptors::indexed(0), [&](auto& cell_iterator) {
        VoronoiCell& voronoi_cell = cell_iterator.value();
        index_t cell_idx          = cell_iterator.index();
        if (map_restricted_cell_idx_to_is_totally_in_shell[cell_idx]) {
          return;
        }
        SurfaceMesh3 cell_mesh3 =
            sha::CreateSurfaceMesh3FromMatMesh3(voronoi_cell.cell_triangle_mesh);
        auto old_polygons  = voronoi_cell.polyhedron.polygons;  // copy
        auto& new_polygons = voronoi_cell.polyhedron.polygons;  // ref
        new_polygons.clear();
        for (auto& old_polygon : old_polygons) {
          auto separate_polygons =
              SeparatePolygonIfEdgeIsSharp(old_polygon, voronoi_cell, cell_mesh3, sharp_angle);
          boost::copy(separate_polygons, std::back_inserter(new_polygons));
        }
      });

  log::info("Separated");

  log::info("Rebuilding Indexing Of Map Cell Triangle To Polygon");
  auto RebuildIndexingOfMapCellTriangleToPolygon = [&](VoronoiCell& voronoi_cell) {
    auto& polygons = voronoi_cell.polyhedron.polygons;
    for (size_t polygon_idx = 0; polygon_idx < polygons.size(); polygon_idx++) {
      auto& polygon = polygons[polygon_idx];
      for (index_t face_idx : polygon.triangle_face_indices_in_cell) {
        voronoi_cell.map_triangle_to_polygon_idx[face_idx] = polygon_idx;
      }
    }
  };
  boost::for_each(voronoi.cells, RebuildIndexingOfMapCellTriangleToPolygon);

  log::info("Rebuilt");

  log::info("Computing Loop Vertex By Its Triangles");

  oneapi::tbb::parallel_for_each(
      voronoi.cells | boost::adaptors::indexed(0), [&](auto& cell_iterator) {
        VoronoiCell& voronoi_cell = cell_iterator.value();
        index_t cell_idx          = cell_iterator.index();
        // if (!map_restricted_cell_idx_to_is_totally_in_shell[cell_idx]) {
        //   boost::for_each(voronoi_cell.polyhedron.polygons, ComputeLoopVertexByItsTriangles);
        // }
        boost::for_each(voronoi_cell.polyhedron.polygons, ComputeLoopVertexByItsTriangles);
      });

  log::info("Computed");

  return voronoi;
}

auto FastCreateRestrictedVoronoiDiagramFromMesh(const MatMesh3& mesh,
                                                const Eigen::MatrixXd& mat_seeds,
                                                double sharp_angle) -> VoronoiDiagram {
  GEO::Mesh boundary_mesh;
  ConvertMatmesh3ToGeoMesh(mesh, boundary_mesh);
  return FastCreateRestrictedVoronoiDiagramFromMesh(boundary_mesh, mat_seeds, sharp_angle);
}
}  // namespace sha
}  // namespace da