#pragma once

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <boost/graph/undirected_graph.hpp>

#include <vector>

#include "sha-base-framework/frame.h"

namespace da {
namespace sha {
using Kernel           = CGAL::Simple_cartesian<double>;
using Point            = Kernel::Point_3;
using SurfaceMesh3     = CGAL::Surface_mesh<Point>;
using SurfaceTopoMesh3 = SurfaceMesh3;

namespace Beam {
template <typename VertexProperty = boost::no_property>
class BeamTopoMesh3 {
 public:
  using GraphType =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperty>;
  using VertexHandle = typename GraphType::vertex_descriptor;
  using BeamHandle   = typename GraphType::edge_descriptor;

 public:
  VertexHandle AddVertex() { return boost::add_vertex(internal_graph_); }

  BeamHandle AddBeam(VertexHandle vertex_u, VertexHandle vertex_v) {
    return boost::add_edge(vertex_u, vertex_v, internal_graph_).first;
  }

  auto Beams(VertexHandle vertex) { boost::incident_edges(vertex, internal_graph_); }

  size_t NumVertices() const { return internal_graph_.num_vertices(); }

  size_t NumBeams() const { return internal_graph_.num_edges(); }

  GraphType internal_graph_;
};

class BeamMesh3 : public BeamTopoMesh3<Eigen::Vector3d> {
  using PointType = Eigen::Vector3d;

 public:
  explicit BeamMesh3() : BeamTopoMesh3<Eigen::Vector3d>() {}

 public:
  VertexHandle AddVertex() { return boost::add_vertex(internal_graph_); }
  VertexHandle AddVertex(const PointType &point) {
    return boost::add_vertex(point, internal_graph_);
  }

  PointType &Point(VertexHandle vertex) { return internal_graph_[vertex]; }

  BeamHandle AddBeam(VertexHandle vertex_u, VertexHandle vertex_v) {
    return boost::add_edge(vertex_u, vertex_v, internal_graph_).first;
  }

  VertexHandle BeamVertex(BeamHandle beam, index_t idx = 0) {
    if (idx == 0)
      return beam.m_source;
    else
      return beam.m_target;
  }

  void RemoveBeam(BeamHandle vtx) { internal_graph_.remove_edge(vtx); }
};
}  // namespace Beam
}  // namespace sha
using sha::SurfaceMesh3;
using sha::SurfaceTopoMesh3;
}  // namespace da
