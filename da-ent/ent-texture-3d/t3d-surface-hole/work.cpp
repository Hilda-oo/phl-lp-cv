#include <igl/writeOBJ.h>

#include <Eigen/Geometry>

#include "parameterization.h"
#include "sample.h"

#include "sha-base-framework/frame.h"
#include "sha-io-foundation/mesh_io.h"

// size_t NumberOfSamples();
// double HoleHeight();
// double HoleRadius();
// bool CurvatureField();

namespace da {
sha::MatMesh3 MakeUnitCylinder(double height, double radius) {
  sha::MatMesh3 matmesh;
  const int axis_devisions = 12;
  Eigen::MatrixXd &V       = matmesh.mat_coordinates;
  Eigen::MatrixXi &F       = matmesh.mat_faces;
  V.resize(2 * axis_devisions + 2, 3);
  F.resize(2 * axis_devisions + 2 * axis_devisions, 3);

  V.row(2 * axis_devisions) << 0, 0, -height / 2;
  V.row(2 * axis_devisions + 1) << 0, 0, height / 2;
  int f = 0;
  for (int th = 0; th < axis_devisions; th++) {
    double x = radius * cos(2. * M_PI * th / axis_devisions);
    double y = radius * sin(2. * M_PI * th / axis_devisions);
    V(th, 0) = x;
    V(th, 1) = y;
    V(th, 2) = -height / 2;

    V(th + axis_devisions, 0) = x;
    V(th + axis_devisions, 1) = y;
    V(th + axis_devisions, 2) = height / 2;
    F(f, 0)                   = ((th + 0) % axis_devisions);
    F(f, 2)                   = ((th + 1) % axis_devisions);
    F(f, 1)                   = ((th + 0) % axis_devisions) + axis_devisions;
    f++;
    F(f, 0) = ((th + 1) % axis_devisions);
    F(f, 2) = ((th + 1) % axis_devisions) + axis_devisions;
    F(f, 1) = ((th + 0) % axis_devisions) + axis_devisions;
    f++;

    // bottom
    F(f, 0) = ((th + 0) % axis_devisions);
    F(f, 2) = 2 * axis_devisions;
    F(f, 1) = ((th + 1) % axis_devisions);
    f++;
    // top
    F(f, 0) = ((th + 0) % axis_devisions) + axis_devisions;
    F(f, 2) = ((th + 1) % axis_devisions) + axis_devisions;
    F(f, 1) = 2 * axis_devisions + 1;
    f++;
  }
  return matmesh;
}
}  // namespace da

auto GenerateSurfaceHoles(size_t expected_num_samples, double hole_height, double hole_radius,
                          bool use_curvature_field_flag) -> da::MatMesh3 {
  using namespace da;  // NOLINT
  fs_path part_path             = WorkingAssetDirectoryPath() / "part.obj";
  fs_path mesh_path             = WorkingAssetDirectoryPath() / "model.obj";
  fs_path parameterization_path = WorkingResultDirectoryPath() / "parameterization.obj";
  fs_path sample_path           = WorkingResultDirectoryPath() / "samples.vtk";
  fs_path cylinders_path        = WorkingResultDirectoryPath() / "cylinders.vtk";
  fs_path model_with_hole_path  = WorkingResultDirectoryPath() / "model_with_hole.obj";

  log::info("Reading mesh from '{}'", mesh_path.string());
  auto part_matmesh = sha::ReadMatMeshFromOBJ(part_path);
  auto part_mesh3   = sha::CreateSurfaceMesh3FromMatMesh3(part_matmesh);

  auto matmesh = sha::ReadMatMeshFromOBJ(mesh_path);

  log::info("Mesh V: {}", part_mesh3.num_vertices());
  log::info("Mesh F: {}", part_mesh3.num_faces());
  log::info("Mesh E: {}", part_mesh3.num_edges());
  log::info("Mesh HE: {}", part_mesh3.num_halfedges());

  Eigen::MatrixXd part_mesh_uv, part_mesh_uv0;
  log::info("parameterizing");
  ParameterizeByArapMethod(part_mesh3, part_mesh_uv);
  log::info("parameterized");
  part_mesh_uv0 = part_mesh_uv;
  part_mesh_uv0.conservativeResize(part_mesh_uv.rows(), 3);
  part_mesh_uv0.col(2).setZero();

  igl::writeOBJ(parameterization_path.string(), part_mesh_uv0, part_matmesh.mat_faces);
  Eigen::MatrixXd mat_sample_points;
  Eigen::MatrixXd mat_sample_normals;
  Eigen::MatrixXd mat_sample_principal_curvatures;

  if (use_curvature_field_flag) {
    //这个函数是引用传值，所以不需要返回值
    SampleInBoundingSquareByCurvatureField(part_matmesh, part_mesh_uv, expected_num_samples,
                                           mat_sample_points, mat_sample_normals,
                                           mat_sample_principal_curvatures);
  } else {
    SampleInBoundingSquare(part_matmesh, part_mesh_uv, expected_num_samples, mat_sample_points,
                           mat_sample_normals);
  }

  sha::WriteToVtk(sample_path, mat_sample_points,
                  Eigen::VectorXi::LinSpaced(mat_sample_points.rows(), 0, mat_sample_points.rows()),
                  {}, {}, 1);
  size_t num_samples = mat_sample_points.rows();
  log::info("num_samples = {}", num_samples);

  MatMesh3 cylinders;
  for (index_t idx = 0; idx < num_samples; ++idx) {
    Eigen::Vector3d normal = mat_sample_normals.row(idx);
    Eigen::Vector3d direction;

    if (use_curvature_field_flag) {
      direction = mat_sample_principal_curvatures.row(idx);
    } else {
      direction << 0, 0, 1;
    }

    Eigen::Vector3d bidirection = normal.cross(direction).normalized();
    direction                   = normal.cross(bidirection).normalized();

    Eigen::Matrix<double, 4, 4> translation = Eigen::Matrix<double, 4, 4>::Identity();
    Eigen::Matrix<double, 4, 4> rotation    = Eigen::Matrix<double, 4, 4>::Identity();
    translation.block(0, 3, 3, 1)           = mat_sample_points.row(idx).transpose();
    rotation.block(0, 0, 3, 1)              = direction;
    rotation.block(0, 1, 3, 1)              = bidirection;
    rotation.block(0, 2, 3, 1)              = normal;
    Eigen::Matrix<double, 4, 4> transform   = translation * rotation;
    auto cylinder                           = MakeUnitCylinder(hole_height, hole_radius);
    for (index_t vtx = 0; vtx < cylinder.NumVertices(); ++vtx) {
      Eigen::Vector3d coord = cylinder.mat_coordinates.row(vtx);
      auto homo             = transform * coord.homogeneous();
      cylinder.mat_coordinates.row(vtx) << homo.x() / homo.w(), homo.y() / homo.w(),
          homo.z() / homo.w();
    }
    cylinders = sha::CombineTwoMatMesh3(cylinders, cylinder);
  }
  log::info("model V: {}, F: {}", matmesh.NumVertices(), matmesh.NumFaces());
  log::info("cylinders V: {}, F: {}", cylinders.NumVertices(), cylinders.NumFaces());
  sha::WriteMatMesh3ToVtk(cylinders_path.string(), cylinders);
  auto model_with_hole = sha::BooleanMinusTwoMatMesh3(matmesh, cylinders);
  log::info("model_with_hole V: {}, F: {}", model_with_hole.NumVertices(),
            model_with_hole.NumFaces());

  igl::writeOBJ(model_with_hole_path.string(), model_with_hole.mat_coordinates,
                model_with_hole.mat_faces);
  return model_with_hole;
}
