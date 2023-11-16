#include <fmt/format.h>
#include <boost/progress.hpp>
#include <iostream>
#include <json.hpp>
#include <ostream>

#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"
#include "sha-implicit-modeling/implicit.h"

#include "sha-base-framework/frame.h"
#include <igl/signed_distance.h>

#include "density.h"
#include <spdlog/spdlog.h>

namespace da {

void ComputeRhosForBackgroundMesh(da::fs_path backgroundmesh_dir) {
  using namespace da;
  // 1. read background mesh
  std::cout << backgroundmesh_dir;
  auto config_path               = backgroundmesh_dir / "config.json";
  auto background_cell_tets_path = backgroundmesh_dir / "tets";
  std::ifstream config_infile(config_path.string());
  nlohmann::json config  = nlohmann::json::parse(config_infile);
  const size_t num_cells = config["cells"];
  std::vector<TetrahedralMatMesh> tetrahedrons(num_cells);

  boost::progress_display progress(num_cells);
  for (index_t cell_idx = 0; cell_idx < num_cells; cell_idx++) {
    auto cell_tet_path     = background_cell_tets_path / fmt::format("tet{}.vtk", cell_idx);
    tetrahedrons[cell_idx] = sha::ReadTetrahedralMatMeshFromVtk(cell_tet_path);
    ++progress;
  }

  // 2. compute rhos on background mesh
  auto rhos = ComputeRhos(tetrahedrons);

  // 3. write rhos
  auto background_rhos_path = backgroundmesh_dir / "rhos";
  for (index_t cell_idx = 0; cell_idx < num_cells; cell_idx++) {
    auto cell_rho_path = background_rhos_path / fmt::format("rho{}.txt", cell_idx);
    sha::WriteVectorToFile(cell_rho_path, rhos[cell_idx]);
  }
}

auto ComputeRhos(const std::vector<da::TetrahedralMatMesh> &tetrahedrons)
    -> std::vector<Eigen::VectorXd> {
  using namespace da;
  const size_t num_macro_cells = tetrahedrons.size();
  std::vector<Eigen::VectorXd> rhos(num_macro_cells);

  auto Heaviside = [](double value, double epsilon) {
    const double a = 1e-9;
    double ret;
    if (value > epsilon) {
      ret = 1;
    } else if (value >= -epsilon) {
      ret = 3.0 * (1 - a) / 4.0 * (value / epsilon - std::pow(value / epsilon, 3) / 3.0) +
            (1 + a) / 2.0;
    } else {
      ret = a;
    }
    return ret;
  };

  const double eps = 1e-3;
  for (index_t cell_idx = 0; cell_idx < num_macro_cells; ++cell_idx) {
    auto &tet = tetrahedrons[cell_idx];
    auto &TV  = tet.mat_coordinates;
    auto &TT  = tet.mat_tetrahedrons;
    Eigen::VectorXd tet_rho;
    tet_rho.resize(TT.rows());

    Eigen::VectorXd S(TV.rows());
    for (index_t idx = 0; idx < S.rows(); ++idx) {
      log::info("remain:{} {}",num_macro_cells - cell_idx,S.rows() - idx);
      S(idx) = Heaviside(ComputeMinDistance(TV.row(idx)), eps);
    }

    for (index_t idx = 0; idx < TT.rows(); ++idx) {
      double value = (S(TT(idx, 0)) + S(TT(idx, 1)) + S(TT(idx, 2)) + S(TT(idx, 3))) / 4.0;
      tet_rho(idx) = value;
    }
    rhos[cell_idx] = tet_rho;
  }
  return rhos;
}

// TODO: provide ComputeMinDistance
double ComputeMinDistance(const Eigen::Vector3d &point) {
  log::info("point:{}  {}  {}",point.x(), point.y(), point.z());
  auto processing_mesh_path   = WorkingResultDirectoryPath() / "field-driven-voxel.obj";
  auto mesh_sdf_path          = WorkingResultDirectoryPath() / "mesh.sdf";
  sha::ScalarField sdf_matrix;
  sdf_matrix.LoadFrom(mesh_sdf_path.string());
  double value = sdf_matrix.Sample(point.x(), point.y(), point.z());
  if(value > 1e-8) { value = 1; }
  else { value = 0; }
  
  return value;
  // MatMesh3 matmesh = sha::ReadMatMeshFromOBJ(processing_mesh_path);
  // Eigen::VectorXd values_;
  // Eigen::VectorXi closest_triangle_indices;
  // Eigen::MatrixXd closest_points;
  // Eigen::MatrixXd closest_point_normals;

  // igl::signed_distance(point, matmesh.mat_coordinates, matmesh.mat_faces,
  //                      igl::SIGNED_DISTANCE_TYPE_DEFAULT, values_, closest_triangle_indices,
  //                      closest_points, closest_point_normals);
  // values_ = -values_;
  // return values_(0); 
}

}  // namespace da
