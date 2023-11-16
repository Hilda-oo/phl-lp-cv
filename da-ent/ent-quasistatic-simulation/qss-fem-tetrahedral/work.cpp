#include <fstream>
#include <memory>

#include <fmt/format.h>
#include <igl/write_triangle_mesh.h>
#include <boost/progress.hpp>

#include "sha-base-framework/declarations.h"
#include "sha-base-framework/frame.h"
#include "sha-fem-quasistatic/fem_tet.h"
#include "sha-simulation-utils/io_utils.h"

#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"

#include "config.h"

// std::string WorkingDirectory();

auto QuasistaticSimulationByFEMTetrahedral()
    -> std::tuple<da::MatMesh3, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> {
  using namespace da;  // NOLINT
  fs_path base_path = WorkingAssetDirectoryPath();
  auto config_path  = base_path / "config.json";

  log::info("Algo is working on path '{}'", base_path.string());

  // load config from json file
  Config config;
  if (!config.loadFromJSON(config_path.string())) {
    spdlog::error("error on reading json file!");
    exit(-1);
  }
  config.backUpConfig((WorkingResultDirectoryPath() / "config.json").string());

  // read tetrahedral mesh
  Eigen::MatrixXd TV;
  Eigen::MatrixXi TT;
  Eigen::MatrixXi SF;
  if (!sha::ReadTetMesh(config.mshFilePath, TV, TT, SF)) {
    spdlog::error("error on reading tet mesh");
    exit(-1);
  }

  // simulation
  sha::FEMTetQuasiSimulator simulator = sha::FEMTetQuasiSimulator(
      TV, TT, SF, config.YM, config.PR, config.DirichletBCs, config.NeumannBCs);
  simulator.simulation();

  // post process
  // writing result deformed model into 'WorkingResultDirectoryPath() / deformed-surf.obj'
  // writing displacement color model into 'WorkingResultDirectoryPath() / dis-color-surf.vtk'
  // writing stress color model into 'WorkingResultDirectoryPath() / stress-color-surf.vtk'
  // simulator.output_surf_result();

  Eigen::MatrixXd mat_deformed_coordinates;
  Eigen::VectorXd vtx_displacement;
  Eigen::VectorXd vtx_stress;
  sha::MatMesh3 surface_mesh =
      simulator.GetSimulatedSurfaceMesh(mat_deformed_coordinates, vtx_displacement, vtx_stress);

  auto RegulateData = [](const Eigen::VectorXd &vector) -> Eigen::VectorXd {
    double min_value = vector.minCoeff();
    double max_value = vector.maxCoeff();
    return (vector.array() - min_value) / (max_value - min_value);
  };

  vtx_stress = RegulateData(vtx_stress);
  vtx_displacement = RegulateData(vtx_displacement);

  Eigen::MatrixXd mat_vtx_displacement_color(vtx_displacement.rows(), 3);
  mat_vtx_displacement_color.setZero();
  mat_vtx_displacement_color.col(0) = vtx_displacement;
  sha::WriteMatMesh3ToObj(WorkingResultDirectoryPath() / "deformed-surf.obj", surface_mesh,
                          mat_vtx_displacement_color);

  return std::make_tuple(surface_mesh, mat_deformed_coordinates, vtx_displacement, vtx_stress);
#if 0
  /* postprocess */
  Eigen::MatrixXd Q;
  Eigen::VectorXd QU;
  Eigen::MatrixXd Qstress;
  SIM::Utils::readMatrix("/home/cw/Downloads/main/main-part/main-part__sf.obj", Q);
  mesh.postprocess(Q, QU, Qstress);
  SIM::Utils::writePntVTK(outputPath + "Q-deform.vtk", Q);
  SIM::Utils::writeMatrix(outputPath + "QU.txt", Eigen::MatrixXd(QU));
  SIM::Utils::writeMatrix(outputPath + "Qstress.txt", Qstress);
#endif
}