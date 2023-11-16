#include "utility.h"

#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/intersections.h>
#include <igl/copyleft/cgal/intersect_other.h>

#include <Eigen/Eigen>

#include <list>
#include <vector>

#include "sha-surface-mesh/matmesh.h"
#include "sha-surface-mesh/mesh3.h"

namespace da {
MatMesh2 ComputeCommonLinesFromTwoMeshes(const MatMesh3 &mesh1, const MatMesh3 &mesh2) {
  // typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
  typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
  typedef Kernel::Point_3 Point;
  typedef Kernel::Triangle_3 Triangle;
  typedef Kernel::Segment_3 Segment;
  typedef CGAL::Polyhedron_3<Kernel> Polyhedron;
  typedef Polyhedron::Edge_iterator Edge_iterator;
  typedef std::list<Triangle>::iterator Iterator;
  typedef CGAL::AABB_triangle_primitive<Kernel, Iterator> Primitive;
  typedef CGAL::AABB_traits<Kernel, Primitive> AABB_triangle_traits;
  typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
  typedef boost::optional<Tree::Intersection_and_primitive_id<Triangle>::Type>
      Triangle_intersection;
  // step.2 build aabb tree for all triangle faces
  std::list<Triangle> triangles_model;
  for (int f = 0; f < mesh1.mat_faces.rows(); ++f) {
    std::vector<Point> pnts_m(3);
    int i = 0;
    for (int v = 0; v < 3; ++v) {
      int pid     = mesh1.mat_faces(f, v);
      pnts_m[i++] = Point(mesh1.mat_coordinates(pid, 0), mesh1.mat_coordinates(pid, 1),
                          mesh1.mat_coordinates(pid, 2));
    }
    triangles_model.emplace_back(pnts_m[0], pnts_m[1], pnts_m[2]);
  }
  Tree tree(triangles_model.begin(), triangles_model.end());
  // tree.accelerate_distance_queries();
  // step3. read every voxel (microstructure)
  std::vector<Segment> inter_segs;  // resulting intersection segments

#pragma omp parallel for
  for (int f = 0; f < mesh2.mat_faces.rows(); ++f) {
    std::vector<Point> pnts(3);
    int i = 0;
    for (int v = 0; v < 3; ++v) {
      int pid   = mesh2.mat_faces(f, v);
      pnts[i++] = Point(mesh2.mat_coordinates(pid, 0), mesh2.mat_coordinates(pid, 1),
                        mesh2.mat_coordinates(pid, 2));
    }
    Triangle triangle_query(pnts[0], pnts[1], pnts[2]);
    std::vector<Triangle_intersection> intersections;
    try {
      tree.all_intersections(triangle_query,
                             std::back_inserter(intersections));  // get all intersections
    } catch (const std::exception &e) {
      log::error(e.what());
    }
    for (auto &intersection : intersections) {
      if (boost::get<Segment>(&(intersection)->first)) {
        // inter_segs.push_back(boost::get<Segment>((intersection)->first));
        Segment *seg = boost::get<Segment>(&(intersection)->first);
#pragma omp critical
        inter_segs.push_back(*seg);
      }
    }
  }

  // step4.
  MatMesh2 cut_lines_mesh;
  cut_lines_mesh.mat_coordinates.resize(inter_segs.size() * 2, 3);
  cut_lines_mesh.mat_beams.resize(inter_segs.size(), 2);
  for (int i = 0; i < inter_segs.size(); i++) {
    auto &seg = inter_segs[i];
    cut_lines_mesh.mat_coordinates.row(i * 2) << CGAL::to_double(seg.source().x()),
        CGAL::to_double(seg.source().y()), CGAL::to_double(seg.source().z());
    cut_lines_mesh.mat_coordinates.row(i * 2 + 1) << CGAL::to_double(seg.target().x()),
        CGAL::to_double(seg.target().y()), CGAL::to_double(seg.target().z());
    cut_lines_mesh.mat_beams.row(i) << i * 2, i * 2 + 1;
  }
  return cut_lines_mesh;
}
}  // namespace da
