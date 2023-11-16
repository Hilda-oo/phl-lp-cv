#include <fstream>
#include <memory>

#include <fmt/format.h>
#include <igl/write_triangle_mesh.h>

#include "sha-base-framework/declarations.h"
#include "sha-base-framework/frame.h"

#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"

std::string MeshPath();
std::string MeshPropertyPath();

auto DensityFieldDisplay(const std::string &mesh_path, const std::string &mesh_property_path)
    -> std::tuple<da::MatMesh3, Eigen::VectorXd> {
  using namespace da;  // NOLINT
  MatMesh3 mesh                 = sha::ReadMatMeshFromOBJ(mesh_path);
  Eigen::VectorXd mesh_property = sha::ReadDoubleVectorFromFile(mesh_property_path);

  auto RegulateData = [](const Eigen::VectorXd &vector) -> Eigen::VectorXd {
    double min_value = vector.minCoeff();
    double max_value = vector.maxCoeff();
    return (vector.array() - min_value) / (max_value - min_value);
  };
  mesh_property = RegulateData(mesh_property);
  return std::make_tuple(mesh, mesh_property);
}