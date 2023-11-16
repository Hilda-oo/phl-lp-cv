#include "matmesh.h"

#include <igl/MeshBooleanType.h>
#include <igl/combine.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/remove_unreferenced.h>

#include <algorithm>
#include <vector>

namespace da {
namespace sha {
auto CreateMatMesh3FromSurfaceMesh3(const SurfaceMesh3 &mesh3, size_t num_face_vertices)
    -> MatMesh3 {
  MatMesh3 matmesh;
  matmesh.mat_coordinates.resize(mesh3.num_vertices(), 3);
  matmesh.mat_faces.resize(mesh3.num_faces(), num_face_vertices);
  for (const auto &vertex : mesh3.vertices()) {
    const auto &point = mesh3.point(vertex);
    matmesh.mat_coordinates.row(vertex.idx()) << point.x(), point.y(), point.z();
  }
  for (const auto &face : mesh3.faces()) {
    auto face_vertex_range = mesh3.vertices_around_face(mesh3.halfedge(face));
    for (int idx = 0; idx < num_face_vertices; idx++) {
      matmesh.mat_faces(face.idx(), idx) = (face_vertex_range.first++)->idx();
    }
  }
  return matmesh;
}

auto CreateFaceMatrixFromSurfaceTopoMesh3(const SurfaceTopoMesh3 &mesh3, size_t num_face_vertices)
    -> Eigen::MatrixXi {
  Eigen::MatrixXi mat_faces(mesh3.num_faces(), num_face_vertices);
  for (const auto &face : mesh3.faces()) {
    auto face_vertex_range = mesh3.vertices_around_face(mesh3.halfedge(face));
    for (int idx = 0; idx < num_face_vertices; idx++) {
      mat_faces(face.idx(), idx) = (face_vertex_range.first++)->idx();
    }
  }
  return mat_faces;
}

auto CreateSurfaceMesh3FromMatMesh3(const MatMesh3 &matmesh) -> SurfaceMesh3 {
  SurfaceMesh3 mesh;
  for (int vertex_idx = 0; vertex_idx < matmesh.NumVertices(); ++vertex_idx) {
    sha::Point point(matmesh.mat_coordinates(vertex_idx, 0), matmesh.mat_coordinates(vertex_idx, 1),
                     matmesh.mat_coordinates(vertex_idx, 2));
    mesh.add_vertex(point);
  }
  for (int face_idx = 0; face_idx < matmesh.NumFaces(); face_idx++) {
    mesh.add_face(SurfaceMesh3::Vertex_index(matmesh.mat_faces(face_idx, 0)),
                  SurfaceMesh3::Vertex_index(matmesh.mat_faces(face_idx, 1)),
                  SurfaceMesh3::Vertex_index(matmesh.mat_faces(face_idx, 2)));
  }
  return mesh;
}

auto CombineTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2) -> MatMesh3 {
  if (matmesh_1.NumVertices() == 0) return matmesh_2;
  if (matmesh_2.NumVertices() == 0) return matmesh_1;
  MatMesh3 result_mesh;
  igl::combine<Eigen::MatrixXd, Eigen::MatrixXi>(
      {matmesh_1.mat_coordinates, matmesh_2.mat_coordinates},
      {matmesh_1.mat_faces, matmesh_2.mat_faces}, result_mesh.mat_coordinates,
      result_mesh.mat_faces);
  return result_mesh;
}
auto CombineTwoMatMesh2(const MatMesh2 &matmesh_1, const MatMesh2 &matmesh_2) -> MatMesh2 {
  if (matmesh_1.NumVertices() == 0) return matmesh_2;
  if (matmesh_2.NumVertices() == 0) return matmesh_1;
  MatMesh2 result_mesh;
  igl::combine<Eigen::MatrixXd, Eigen::MatrixXi>(
      {matmesh_1.mat_coordinates, matmesh_2.mat_coordinates},
      {matmesh_1.mat_beams, matmesh_2.mat_beams}, result_mesh.mat_coordinates,
      result_mesh.mat_beams);
  return result_mesh;
}

auto BooleanIntersectTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2,
                                 Eigen::VectorXi &brith_face_indices) -> MatMesh3 {
  MatMesh3 result_matmesh;
  igl::copyleft::cgal::mesh_boolean(
      matmesh_1.mat_coordinates, matmesh_1.mat_faces, matmesh_2.mat_coordinates,
      matmesh_2.mat_faces, igl::MESH_BOOLEAN_TYPE_INTERSECT, result_matmesh.mat_coordinates,
      result_matmesh.mat_faces, brith_face_indices);
  return result_matmesh;
}
auto BooleanUnionTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2,
                             Eigen::VectorXi &brith_face_indices) -> MatMesh3 {
  MatMesh3 result_matmesh;
  igl::copyleft::cgal::mesh_boolean(matmesh_1.mat_coordinates, matmesh_1.mat_faces,
                                    matmesh_2.mat_coordinates, matmesh_2.mat_faces,
                                    igl::MESH_BOOLEAN_TYPE_UNION, result_matmesh.mat_coordinates,
                                    result_matmesh.mat_faces, brith_face_indices);
  return result_matmesh;
}
auto BooleanMinusTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2,
                             Eigen::VectorXi &brith_face_indices) -> MatMesh3 {
  MatMesh3 result_matmesh;
  igl::copyleft::cgal::mesh_boolean(matmesh_1.mat_coordinates, matmesh_1.mat_faces,
                                    matmesh_2.mat_coordinates, matmesh_2.mat_faces,
                                    igl::MESH_BOOLEAN_TYPE_MINUS, result_matmesh.mat_coordinates,
                                    result_matmesh.mat_faces, brith_face_indices);
  return result_matmesh;
}

auto BooleanIntersectTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2) -> MatMesh3 {
  Eigen::VectorXi brith_face_indices;
  return BooleanIntersectTwoMatMesh3(matmesh_1, matmesh_2, brith_face_indices);
}

auto BooleanUnionTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2) -> MatMesh3 {
  Eigen::VectorXi brith_face_indices;
  return BooleanUnionTwoMatMesh3(matmesh_1, matmesh_2, brith_face_indices);
}

auto BooleanMinusTwoMatMesh3(const MatMesh3 &matmesh_1, const MatMesh3 &matmesh_2) -> MatMesh3 {
  Eigen::VectorXi brith_face_indices;
  return BooleanMinusTwoMatMesh3(matmesh_1, matmesh_2, brith_face_indices);
}

auto RemoveUnreferencedVertices(const MatMesh3 &matmesh) -> MatMesh3 {
  MatMesh3 new_matmesh;
  Eigen::VectorXi map_new_vertex_to_old;
  igl::remove_unreferenced(matmesh.mat_coordinates, matmesh.mat_faces, new_matmesh.mat_coordinates,
                           new_matmesh.mat_faces, map_new_vertex_to_old);
  return new_matmesh;
}

auto RemoveUnreferencedVertices(const MatMesh2 &matmesh) -> MatMesh2 {
  MatMesh2 new_matmesh;
  Eigen::VectorXi map_new_vertex_to_old;
  igl::remove_unreferenced(matmesh.mat_coordinates, matmesh.mat_beams, new_matmesh.mat_coordinates,
                           new_matmesh.mat_beams, map_new_vertex_to_old);
  return new_matmesh;
}

auto Materialize(const MatMesh2 &beam_mesh, const Eigen::VectorXd &beam_radiuses) -> MatMesh3 {
  Assert(beam_mesh.NumBeams() == beam_radiuses.size(),
         "beam_radiuses.size() != matmesh.NumBeams()");
  Eigen::VectorXd vertex_radiuses(beam_mesh.NumVertices(), 0);
  for (index_t beam_idx = 0; beam_idx < beam_mesh.NumBeams(); ++beam_idx) {
    double beam_radius         = beam_radiuses(beam_idx);
    index_t vtx_a_idx          = beam_mesh.mat_beams(beam_idx, 0);
    index_t vtx_b_idx          = beam_mesh.mat_beams(beam_idx, 1);
    vertex_radiuses(vtx_a_idx) = std::max(vertex_radiuses(vtx_a_idx), beam_radius);
    vertex_radiuses(vtx_b_idx) = std::max(vertex_radiuses(vtx_b_idx), beam_radius);
  }

  auto MakeUnitCylinder = [](double height, double radius) -> sha::MatMesh3 {
    sha::MatMesh3 matmesh;
    const int axis_devisions = 12;
    double top_z             = height;
    double bottom_z          = 0;

    Eigen::MatrixXd &V = matmesh.mat_coordinates;
    Eigen::MatrixXi &F = matmesh.mat_faces;
    V.resize(2 * axis_devisions + 2, 3);
    F.resize(2 * axis_devisions + 2 * axis_devisions, 3);

    V.row(2 * axis_devisions) << 0, 0, bottom_z;
    V.row(2 * axis_devisions + 1) << 0, 0, top_z;
    int face_idx = 0;
    for (int th = 0; th < axis_devisions; th++) {
      double x = radius * cos(2. * M_PI * th / axis_devisions);
      double y = radius * sin(2. * M_PI * th / axis_devisions);
      V(th, 0) = x;
      V(th, 1) = y;
      V(th, 2) = bottom_z;

      V(th + axis_devisions, 0) = x;
      V(th + axis_devisions, 1) = y;
      V(th + axis_devisions, 2) = top_z;
      F(face_idx, 0)            = ((th + 0) % axis_devisions);
      F(face_idx, 2)            = ((th + 1) % axis_devisions);
      F(face_idx, 1)            = ((th + 0) % axis_devisions) + axis_devisions;
      face_idx++;
      F(face_idx, 0) = ((th + 1) % axis_devisions);
      F(face_idx, 2) = ((th + 1) % axis_devisions) + axis_devisions;
      F(face_idx, 1) = ((th + 0) % axis_devisions) + axis_devisions;
      face_idx++;

      // bottom
      F(face_idx, 0) = ((th + 0) % axis_devisions);
      F(face_idx, 2) = 2 * axis_devisions;
      F(face_idx, 1) = ((th + 1) % axis_devisions);
      face_idx++;
      // top
      F(face_idx, 0) = ((th + 0) % axis_devisions) + axis_devisions;
      F(face_idx, 2) = ((th + 1) % axis_devisions) + axis_devisions;
      F(face_idx, 1) = 2 * axis_devisions + 1;
      face_idx++;
    }
    return matmesh;
  };

  auto MakeCylinder = [&](const Eigen::Vector3d &vtx_a, const Eigen::Vector3d &vtx_b,
                          double radius) -> sha::MatMesh3 {
    double height          = (vtx_b - vtx_a).norm();
    Eigen::Vector3d normal = (vtx_b - vtx_a).normalized();
    Eigen::Vector3d direction(0, 0, 1);
    bool change_dir = std::abs(std::abs(normal.z()) - 1) < 1e-2;
    if (change_dir) {
      direction = Eigen::Vector3d(0, 1, 0);
    }
    Eigen::Vector3d bidirection = normal.cross(direction).normalized();
    direction                   = normal.cross(bidirection).normalized();

    Eigen::Matrix<double, 4, 4> translation  = Eigen::Matrix<double, 4, 4>::Identity();
    Eigen::Matrix<double, 4, 4> translation0 = Eigen::Matrix<double, 4, 4>::Identity();

    Eigen::Matrix<double, 4, 4> rotation = Eigen::Matrix<double, 4, 4>::Identity();
    translation.block(0, 3, 3, 1)        = ((vtx_a)).transpose();
    translation0.block(0, 3, 3, 1)       = -((vtx_a)).transpose();
    if (change_dir) {
      rotation.block(0, 0, 3, 1) = direction;
      rotation.block(0, 1, 3, 1) = normal;
      rotation.block(0, 2, 3, 1) = bidirection;
    } else {
      rotation.block(0, 0, 3, 1) = direction;
      rotation.block(0, 1, 3, 1) = bidirection;
      rotation.block(0, 2, 3, 1) = normal;
    }

    Eigen::Matrix<double, 4, 4> transform = rotation;  // translation * rotation * translation0;
    MatMesh3 cylinder                     = MakeUnitCylinder(height, radius);
    if (change_dir) {
      cylinder.mat_coordinates.col(1).swap(cylinder.mat_coordinates.col(2));
    }
    for (index_t vtx = 0; vtx < cylinder.NumVertices(); ++vtx) {
      Eigen::Vector3d coord = cylinder.mat_coordinates.row(vtx);
      auto homo             = transform * coord.homogeneous();
      Eigen::Vector3d new_coord(homo.x() / homo.w(), homo.y() / homo.w(), homo.z() / homo.w());
      new_coord                         = new_coord + vtx_a;
      cylinder.mat_coordinates.row(vtx) = new_coord;
    }
    return cylinder;
  };

  auto MakeUnitIcoSphere = [](int depth = 3) -> sha::MatMesh3 {
    typedef Eigen::Vector3i Triangle;
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> triangles;

    double t = (1.0 + std::sqrt(5.0)) / 2.0;

    vertices.push_back(Eigen::Vector3d(-1, t, 0));
    vertices.push_back(Eigen::Vector3d(1, t, 0));
    vertices.push_back(Eigen::Vector3d(-1, -t, 0));
    vertices.push_back(Eigen::Vector3d(1, -t, 0));

    vertices.push_back(Eigen::Vector3d(0, -1, t));
    vertices.push_back(Eigen::Vector3d(0, 1, t));
    vertices.push_back(Eigen::Vector3d(0, -1, -t));
    vertices.push_back(Eigen::Vector3d(0, 1, -t));

    vertices.push_back(Eigen::Vector3d(t, 0, -1));
    vertices.push_back(Eigen::Vector3d(t, 0, 1));
    vertices.push_back(Eigen::Vector3d(-t, 0, -1));
    vertices.push_back(Eigen::Vector3d(-t, 0, 1));

    for (auto &vtx : vertices) {
      vtx.normalize();
    }

    triangles.push_back(Eigen::Vector3i(0, 11, 5));
    triangles.push_back(Eigen::Vector3i(0, 5, 1));
    triangles.push_back(Eigen::Vector3i(0, 1, 7));
    triangles.push_back(Eigen::Vector3i(0, 7, 10));
    triangles.push_back(Eigen::Vector3i(0, 10, 11));
    triangles.push_back(Eigen::Vector3i(1, 5, 9));
    triangles.push_back(Eigen::Vector3i(5, 11, 4));
    triangles.push_back(Eigen::Vector3i(11, 10, 2));
    triangles.push_back(Eigen::Vector3i(10, 7, 6));
    triangles.push_back(Eigen::Vector3i(7, 1, 8));
    triangles.push_back(Eigen::Vector3i(3, 9, 4));
    triangles.push_back(Eigen::Vector3i(3, 4, 2));
    triangles.push_back(Eigen::Vector3i(3, 2, 6));
    triangles.push_back(Eigen::Vector3i(3, 6, 8));
    triangles.push_back(Eigen::Vector3i(3, 8, 9));
    triangles.push_back(Eigen::Vector3i(4, 9, 5));
    triangles.push_back(Eigen::Vector3i(2, 4, 11));
    triangles.push_back(Eigen::Vector3i(6, 2, 10));
    triangles.push_back(Eigen::Vector3i(8, 6, 7));
    triangles.push_back(Eigen::Vector3i(9, 8, 1));

    // 递归细分三角形
    for (int i = 0; i < depth; i++) {
      std::vector<Triangle> new_triangles;

      for (const Triangle &triangle : triangles) {
        // 取出三角形的 3 个顶点
        Eigen::Vector3d v1 = vertices[triangle.x()];
        Eigen::Vector3d v2 = vertices[triangle.y()];
        Eigen::Vector3d v3 = vertices[triangle.z()];

        // 生成 3 个中点，并将其归一化
        Eigen::Vector3d v12 = (v1 + v2) / 2.0f;
        Eigen::Vector3d v23 = (v2 + v3) / 2.0f;
        Eigen::Vector3d v31 = (v3 + v1) / 2.0f;
        v12.normalize();
        v23.normalize();
        v31.normalize();

        // 计算中点所对应的点编号，并将其加入到新的顶点数组
        int i12 = vertices.size();
        vertices.push_back(v12);
        int i23 = vertices.size();
        vertices.push_back(v23);
        int i31 = vertices.size();
        vertices.push_back(v31);

        // 建立新的三角形，并将它们加入到新的三角形数组中
        new_triangles.push_back(Triangle(triangle.x(), i12, i31));
        new_triangles.push_back(Triangle(triangle.y(), i23, i12));
        new_triangles.push_back(Triangle(triangle.z(), i31, i23));
        new_triangles.push_back(Triangle(i12, i23, i31));
      }

      // 用新的三角形数组替代旧的三角形数组
      triangles = new_triangles;
    }

    MatMesh3 mesh;
    mesh.mat_coordinates.resize(vertices.size(), 3);
    for (index_t vtx_idx = 0; vtx_idx < vertices.size(); vtx_idx++) {
      mesh.mat_coordinates.row(vtx_idx) = vertices[vtx_idx].transpose();
    }
    mesh.mat_faces.resize(triangles.size(), 3);
    for (index_t face_idx = 0; face_idx < triangles.size(); face_idx++) {
      mesh.mat_faces.row(face_idx) = triangles[face_idx].transpose();
    }
    return mesh;
  };

  auto MakeSphere = [&](const Eigen::Vector3d &centre, double radius,
                        int depth = 1) -> sha::MatMesh3 {
    auto sphere = MakeUnitIcoSphere(depth);
    sphere.mat_coordinates *= radius;
    sphere.mat_coordinates.rowwise() += centre.transpose();
    return sphere;
  };

  MatMesh3 sphere_mesh;
  for (index_t vtx_idx = 0; vtx_idx < beam_mesh.NumVertices(); vtx_idx++) {
    auto sphere = MakeSphere(beam_mesh.mat_coordinates.row(vtx_idx), vertex_radiuses(vtx_idx), 2);
    sphere_mesh = sha::CombineTwoMatMesh3(sphere_mesh, sphere);
  }
  MatMesh3 beam_entity_mesh;

  for (index_t beam_idx = 0; beam_idx < beam_mesh.NumBeams(); beam_idx++) {
    const auto &beam      = beam_mesh.mat_beams.row(beam_idx);
    Eigen::Vector3d vtx_a = beam_mesh.mat_coordinates.row(beam(0));
    Eigen::Vector3d vtx_b = beam_mesh.mat_coordinates.row(beam(1));
    auto beam_entity      = MakeCylinder(vtx_a, vtx_b, beam_radiuses(beam_idx));
    beam_entity_mesh      = sha::CombineTwoMatMesh3(beam_entity_mesh, beam_entity);
  }
  return sha::CombineTwoMatMesh3(beam_entity_mesh, sphere_mesh);
}

}  // namespace sha
}  // namespace da
