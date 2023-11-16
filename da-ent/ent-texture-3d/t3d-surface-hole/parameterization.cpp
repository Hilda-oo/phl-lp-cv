#include "parameterization.h"

#include <CGAL/polygon_mesh_processing.h>
#include <igl/cotmatrix.h>
#include <igl/per_face_normals.h>

#include <tuple>

namespace da {
namespace PMP = CGAL::Polygon_mesh_processing;
void ParameterizeByTuttesMethod(const SurfaceMesh3 &mesh3, Eigen::MatrixXd &uv) {
  const size_t num_vertices = mesh3.num_vertices();
  const size_t num_faces    = mesh3.num_faces();
  uv.setZero(num_vertices, 2);

  double area_of_mesh   = PMP::area(mesh3);
  auto center_of_mesh   = PMP::centroid(mesh3);
  double radius_of_mesh = sqrt(area_of_mesh / (M_PI));

  log::info("mesh area: {}", area_of_mesh);
  log::info("mesh radius: {}", radius_of_mesh);
  log::info("mesh center: ({}, {}, {})", center_of_mesh.x(), center_of_mesh.y(),
            center_of_mesh.z());

  // find boundary
  size_t num_boundary_edges = 0;
  auto halfedge_iterator    = mesh3.halfedges_begin();
  while (halfedge_iterator != mesh3.halfedges_end() && !mesh3.is_border(*halfedge_iterator)) {
    halfedge_iterator++;
  }
  if (halfedge_iterator == mesh3.halfedges_end()) {
    Terminate("Mesh has no border");
  }
  auto boundary_halfedges_begin = *halfedge_iterator;
  auto boundary_halfedge        = boundary_halfedges_begin;

  do {
    boundary_halfedge = mesh3.next(boundary_halfedge);
    num_boundary_edges++;
  } while (boundary_halfedge != boundary_halfedges_begin);
  log::info("boundary_cnt: {}", num_boundary_edges);

  double theta_step = 2 * M_PI / num_boundary_edges;
  for (index_t i = 0; i < num_boundary_edges; ++i) {
    double theta             = i * theta_step;
    index_t target_vertex    = mesh3.target(boundary_halfedges_begin).idx();
    uv(target_vertex, 0)     = center_of_mesh[0] + radius_of_mesh * cos(theta);
    uv(target_vertex, 1)     = center_of_mesh[1] + radius_of_mesh * sin(theta);
    boundary_halfedges_begin = mesh3.next(boundary_halfedges_begin);
  }

  typedef Eigen::SparseMatrix<double>
      SparseMatrix;  // declares a column-major sparse matrix type of double
  typedef Eigen::Triplet<double> Triplet;
  std::vector<Triplet> triplet_list;
  Eigen::VectorXd right_hand_u = Eigen::VectorXd::Zero(num_vertices);
  Eigen::VectorXd right_hand_v = Eigen::VectorXd::Zero(num_vertices);

  for (auto &vertex : mesh3.vertices()) {
    if (mesh3.is_border(vertex)) {
      triplet_list.push_back(Triplet(vertex.idx(), vertex.idx(), 1));
      right_hand_u(vertex.idx()) = uv(vertex.idx(), 0);
      right_hand_v(vertex.idx()) = uv(vertex.idx(), 1);
    } else {
      for (auto neighbor_vtx : mesh3.vertices_around_target(mesh3.halfedge(vertex))) {
        triplet_list.push_back(Triplet(vertex.idx(), neighbor_vtx.idx(), -1));
      }
      triplet_list.push_back(Triplet(vertex.idx(), vertex.idx(), mesh3.degree(vertex)));
    }
  }
  log::info("tt");
  SparseMatrix mat_left_hand(num_vertices, num_vertices);
  mat_left_hand.setFromTriplets(triplet_list.begin(), triplet_list.end());

  Eigen::SparseLU<SparseMatrix> solver;
  solver.compute(mat_left_hand);

  uv.col(0) = solver.solve(right_hand_u);
  uv.col(1) = solver.solve(right_hand_v);
}

void ParameterizeByArapMethod(const SurfaceMesh3 &mesh3, Eigen::MatrixXd &uv) {
  const size_t num_vertices = mesh3.num_vertices();
  const size_t num_faces    = mesh3.num_faces();

  auto matmesh = sha::CreateMatMesh3FromSurfaceMesh3(mesh3);
  Eigen::MatrixXd mat_surface_normals;
  igl::per_face_normals(matmesh.mat_coordinates, matmesh.mat_faces, mat_surface_normals);
  uv.setZero(num_vertices, 2);
  Eigen::MatrixXd tuttes_uv;
  ParameterizeByTuttesMethod(mesh3, tuttes_uv);

  uv = tuttes_uv;

  std::vector<Eigen::Matrix2d> Lt(num_faces);
  std::vector<std::tuple<Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d>> localCoords(num_faces);
  for (auto &f : mesh3.faces()) {
    auto point_0           = matmesh.mat_coordinates.row(matmesh.mat_faces(f.idx(), 0));
    auto point_1           = matmesh.mat_coordinates.row(matmesh.mat_faces(f.idx(), 1));
    auto point_2           = matmesh.mat_coordinates.row(matmesh.mat_faces(f.idx(), 2));
    Eigen::Vector3d normal = mat_surface_normals.row(f.idx()).normalized();
    Eigen::Vector3d e20    = point_2 - point_0;
    Eigen::Vector3d u      = (point_1 - point_0).normalized();
    Eigen::Vector3d v      = normal.cross(u).normalized();

    localCoords[f.idx()] =
        std::make_tuple(Eigen::Vector2d(0, 0), Eigen::Vector2d((point_1 - point_0).norm(), 0),
                        Eigen::Vector2d(e20.dot(u), e20.dot(v)));
  }

  typedef Eigen::SparseMatrix<double>
      SparseMatrix;  // declares a column-major sparse matrix type of double
  typedef Eigen::Triplet<double> Triplet;
  std::vector<Triplet> triplet_list;
  std::vector<double> cots(mesh3.num_halfedges(), 0);
  for (auto &he : mesh3.halfedges()) {
    if (mesh3.is_border(he)) continue;

    auto v1            = matmesh.mat_coordinates.row(mesh3.source(he).idx());
    auto v2            = matmesh.mat_coordinates.row(mesh3.target(he).idx());
    auto v0            = matmesh.mat_coordinates.row(mesh3.target(mesh3.next(he)).idx());
    Eigen::Vector3d e0 = v1 - v0;
    Eigen::Vector3d e1 = v2 - v0;
    double cot         = (e0.dot(e1)) / (e0.cross(e1)).norm();
    cots[he.idx()]     = cot;
    auto from          = mesh3.source(he).idx();
    auto to            = mesh3.target(he).idx();
    triplet_list.push_back(Triplet(from, from, cot));
    triplet_list.push_back(Triplet(to, to, cot));
    triplet_list.push_back(Triplet(from, to, -cot));
    triplet_list.push_back(Triplet(to, from, -cot));
  }

  SparseMatrix mat_lefthand(num_vertices, num_vertices);
  mat_lefthand.setFromTriplets(triplet_list.begin(), triplet_list.end());
  Eigen::SparseLU<SparseMatrix> solver;
  solver.compute(mat_lefthand);

  for (int i = 0; i < 100; ++i) {
    for (auto &f : mesh3.faces()) {
      index_t v0       = matmesh.mat_faces(f.idx(), 0);
      index_t v1       = matmesh.mat_faces(f.idx(), 1);
      index_t v2       = matmesh.mat_faces(f.idx(), 2);
      auto &localCoord = localCoords[f.idx()];

      Eigen::Matrix2d LL;
      Eigen::Matrix2d OO;
      LL.col(0) = (uv.row(v1) - uv.row(v0)).transpose();
      LL.col(1) = (uv.row(v2) - uv.row(v0)).transpose();

      OO.col(0) = std::get<1>(localCoord) - std::get<0>(localCoord);
      OO.col(1) = std::get<2>(localCoord) - std::get<0>(localCoord);

      Eigen::Matrix2d J = LL * OO.inverse();
      Eigen::Matrix2d U, V, R;
      Eigen::JacobiSVD<Eigen::Matrix2d> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
      U = svd.matrixU();
      V = svd.matrixV();
      R = U * V.transpose();
      if (R.determinant() < 0) {
        U.col(1) *= -1;
        R = U * V.transpose();
      }
      Lt[f.idx()] = R;
    }

    Eigen::VectorXd bu, bv;
    bu.setZero(num_vertices);
    bv.setZero(num_vertices);

    for (auto &f : mesh3.faces()) {
      auto vtx_around_face = mesh3.vertices_around_face(mesh3.halfedge(f));
      auto v0              = *(vtx_around_face.first++);
      auto v1              = *(vtx_around_face.first++);
      auto v2              = *(vtx_around_face.first++);

      auto e2          = mesh3.halfedge(f);
      auto e0          = mesh3.next(e2);
      auto e1          = mesh3.next(e0);
      auto &localCoord = localCoords[f.idx()];
      auto local0      = std::get<0>(localCoord);
      auto local1      = std::get<1>(localCoord);
      auto local2      = std::get<2>(localCoord);

      Eigen::Vector2d localE0, localE1, localE2;
      localE0 = local1 - local0;
      localE1 = local2 - local1;
      localE2 = local0 - local2;

      Eigen::Vector2d b0, b1, b2;
      b0 = cots[e0.idx()] * Lt[f.idx()] * localE0;
      b1 = cots[e1.idx()] * Lt[f.idx()] * localE1;
      b2 = cots[e2.idx()] * Lt[f.idx()] * localE2;
      bu[v0.idx()] -= b0.x();
      bv[v0.idx()] -= b0.y();
      bu[v1.idx()] += b0.x();
      bv[v1.idx()] += b0.y();

      bu[v1.idx()] -= b1.x();
      bv[v1.idx()] -= b1.y();
      bu[v2.idx()] += b1.x();
      bv[v2.idx()] += b1.y();

      bu[v2.idx()] -= b2.x();
      bv[v2.idx()] -= b2.y();
      bu[v0.idx()] += b2.x();
      bv[v0.idx()] += b2.y();
    }
    uv.col(0) = solver.solve(bu);
    uv.col(1) = solver.solve(bv);
  }
}
}  // namespace da
