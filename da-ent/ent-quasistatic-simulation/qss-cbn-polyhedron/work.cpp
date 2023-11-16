#include <fstream>
#include <memory>

#include <fmt/format.h>
#include <igl/write_triangle_mesh.h>
#include <boost/progress.hpp>

#include "sha-base-framework/declarations.h"
#include "sha-base-framework/frame.h"
#include "sha-simulation-3d/CBN.h"
#include "sha-simulation-utils/io_utils.h"

#include <json.hpp>

#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"
#include "utilis.h"

std::string ConfigFile();
double YoungsModulus();
double PoissionRatio();
double Density();

namespace da {
class Worker {
 public:
  explicit Worker(double p_YM1, double p_YM0, double p_PR, double p_penaltyYM, double p_penaltyDBC)
      : YM1(p_YM1), YM0(p_YM0), PR(p_PR), penaltyYM(p_penaltyYM), penaltyDBC(p_penaltyDBC) {}

  void LoadDesignDomainMesh(const fs_path &mesh_path, const fs_path &query_mesh_path) {
    mesh_ = sha::ReadMatMeshFromOBJ(mesh_path);
    log::info("mesh num v: {}", mesh_.NumVertices());
    log::info("mesh num f: {}", mesh_.NumFaces());

    design_domain_.min() = mesh_.mat_coordinates.colwise().minCoeff();
    design_domain_.max() = mesh_.mat_coordinates.colwise().maxCoeff();

    query_mesh_ = sha::ReadMatMeshFromOBJ(query_mesh_path);
  }

  void LoadBackgroundMesh(const fs_path &background_cells_path,
                          const fs_path &background_polyhedrons_path,
                          const fs_path &background_cell_tets_path, size_t num_cells) {
    log::info("{} cells will be loaded", num_cells);
    std::vector<sha::NestedCell> background_cells(num_cells);
    boost::progress_display progress(num_cells);
    for (index_t cell_idx = 0; cell_idx < num_cells; cell_idx++) {
      sha::NestedCell cell;
      auto cell_path     = background_cells_path / fmt::format("cell{}.obj", cell_idx);
      auto cell_tet_path = background_cell_tets_path / fmt::format("tet{}.vtk", cell_idx);
      auto polyhedron_path =
          background_polyhedrons_path / fmt::format("polyhedron{}.txt", cell_idx);
      background_cells[cell_idx].tetrahedrons = sha::ReadTetrahedralMatMeshFromVtk(cell_tet_path);
      background_cells[cell_idx].macro_mesh   = sha::ReadMatMeshFromOBJ(cell_path);
      background_cells[cell_idx].polyhedron_edges = ReadPolyhedronEdgesFromPath(polyhedron_path);
      background_cells[cell_idx].cell_box = background_cells[cell_idx].macro_mesh.AlignedBox();
      ++progress;
    }
    nested_background_mesh_ = std::make_shared<sha::NestedBackgroundMesh>(background_cells);
  }

  void LoadBoundaryConditions(std::vector<Eigen::VectorXi> NBCIndex,
                              std::vector<Eigen::Vector3d> NBCVal,
                              std::vector<Eigen::VectorXi> DBCIndex) {
    // init physical domain
    std::vector<Eigen::Matrix<double, 2, 3>> NBCRelBBox_empty;
    std::vector<Eigen::Vector3d> NBCVal_empty;
    std::vector<Eigen::Matrix<double, 2, 3>> DBCRelBBox_empty;
    std::vector<Eigen::Vector3d> DBCVal_empty;
    physical_domain_ = std::make_shared<sha::PhysicalDomain>(mesh_, NBCRelBBox_empty, NBCVal_empty,
                                                             DBCRelBBox_empty, DBCVal_empty);

    int nNBC = 0;
    std::vector<std::pair<Eigen::RowVector3d, Eigen::Vector3d>> NBC;
    int nDBC = 0;
    std::vector<std::pair<int, Eigen::Vector3d>> DBC;

    for (int i = 0; i < NBCIndex.size(); ++i) {
      Eigen::Vector3d NBCVal_eachPoint = NBCVal[i] / NBCIndex[i].size();
      for (const auto idx : NBCIndex[i]) {
        NBC.emplace_back(std::make_pair(mesh_.mat_coordinates.row(idx), NBCVal_eachPoint));
        ++nNBC;
      }
    }

    Eigen::VectorXi mesh_idx_flag = Eigen::VectorXi::Zero(mesh_.NumVertices());
    for (int i = 0; i < DBCIndex.size(); ++i) {
      for (const auto idx : DBCIndex[i]) {
        mesh_idx_flag(idx) = 1;
      }
    }
    for (int fI = 0; fI < mesh_.NumFaces(); ++fI) {
      int flag = true;
      for (int j = 0; j < 3; ++j) {
        if (!mesh_idx_flag(mesh_.mat_faces(fI, j))) {
          flag = false;
          break;
        }
      }
      if (!flag) {
        continue;
      }
      DBC.emplace_back(std::make_pair(fI, Eigen::Vector3d::Zero()));
      ++nDBC;
    }
    physical_domain_->SetBoundaryConditions(nNBC, NBC, nDBC, DBC, penaltyDBC);

    physical_domain_->WriteNBCToVtk(WorkingResultDirectoryPath() / "NBC.vtk");
    physical_domain_->WriteDBCToObj(WorkingResultDirectoryPath() / "DBC-tri.obj");

    // init CBN background
    simulator_ = std::make_shared<sha::CBNSimulator>(YM1, YM0, PR, penaltyYM, physical_domain_,
                                                     nested_background_mesh_, false);
  }

  auto Run(const fs_path &base_path, const std::vector<Eigen::VectorXd> &rhos)
      -> std::tuple<da::MatMesh3, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> {
        // simulation
    Eigen::VectorXd displacements = simulator_->Simulate(rhos);
    // physical_domain_->WriteV1MeshToObj(WorkingResultDirectoryPath() / "mesh-deformed.obj");

#if 0
    // post-process
    spdlog::info("Postprocess query points");
    std::vector<Eigen::Vector3d> mesh_result_displacement;
    std::vector<Eigen::Matrix3d> mesh_result_stress;
    if (0) {
      simulator_->QueryResults(query_mesh_.mat_coordinates, mesh_result_displacement,
                               mesh_result_stress);
    } else {
      Eigen::VectorXi query_flag =
          sha::ReadVectorFromFile<int>(base_path / "query" / "query_flag.txt");
      Eigen::VectorXi query_mac_index =
          sha::ReadVectorFromFile<int>(base_path / "query" / "query_macI.txt");
      Eigen::VectorXi query_mic_index =
          sha::ReadVectorFromFile<int>(base_path / "query" / "query_micI.txt");
      simulator_->QueryResultsWithLocation(query_mesh_.mat_coordinates, query_flag, query_mac_index,
                                           query_mic_index, mesh_result_displacement,
                                           mesh_result_stress);
    }

    std::vector<double> vtx_displacement_vec(query_mesh_.mat_coordinates.rows());
    std::vector<double> vtx_stress_vec(query_mesh_.mat_coordinates.rows());
    for (int vI = 0; vI < query_mesh_.mat_coordinates.rows(); ++vI) {
      vtx_displacement_vec[vI] = mesh_result_displacement[vI].norm();
      vtx_stress_vec[vI]       = mesh_result_stress[vI].norm();
    }
    sha::WriteTriVTK((WorkingResultDirectoryPath() / "mesh-result-displacement.vtk").string(),
                     query_mesh_.mat_coordinates, query_mesh_.mat_faces, {}, vtx_displacement_vec);
    sha::WriteTriVTK((WorkingResultDirectoryPath() / "mesh-result-stress.vtk").string(),
                     query_mesh_.mat_coordinates, query_mesh_.mat_faces, {}, vtx_stress_vec);

    // Eigen::VectorXd vtx_displacement_eigen = Eigen::Map<Eigen::VectorXd>(vtx_displacement_vec.data(), vtx_displacement_vec.size());
    // Eigen::VectorXd vtx_stress_eigen = Eigen::Map<Eigen::VectorXd>(vtx_stress_vec.data(), vtx_stress_vec.size());
    // sha::WriteVectorToFile(WorkingResultDirectoryPath() / "vtx_displacement_eigen.txt", vtx_displacement_eigen);
    // sha::WriteVectorToFile(WorkingResultDirectoryPath() / "vtx_stress_eigen.txt", vtx_stress_eigen);
#endif

    // data for return
    // Eigen::MatrixXd mat_deformed_coordinates(query_mesh_.mat_coordinates.rows(), 3);
    // for (int vI = 0; vI < query_mesh_.mat_coordinates.rows(); ++vI) {
    //   mat_deformed_coordinates.row(vI) =
    //       query_mesh_.mat_coordinates.row(vI) + mesh_result_displacement[vI].transpose();
    // }
    // Eigen::VectorXd vtx_displacement =
    //     Eigen::Map<Eigen::VectorXd>(vtx_displacement_vec.data(), vtx_displacement_vec.size());
    // Eigen::VectorXd vtx_stress =
    //     Eigen::Map<Eigen::VectorXd>(vtx_stress_vec.data(), vtx_stress_vec.size());

    Eigen::VectorXd vtx_displacement = sha::ReadVectorFromFile<double>(base_path / "query" / "vtx_displacement_eigen.txt");
    Eigen::VectorXd vtx_stress = sha::ReadVectorFromFile<double>(base_path / "query" / "vtx_stress_eigen.txt");
    auto RegulateData = [](const Eigen::VectorXd &vector) -> Eigen::VectorXd {
      double min_value = vector.minCoeff();
      double max_value = vector.maxCoeff();
      return (vector.array() - min_value) / (max_value - min_value);
    };
    vtx_displacement = RegulateData(vtx_displacement);
    vtx_stress       = RegulateData(vtx_stress);

    // return std::make_tuple(query_mesh_, mat_deformed_coordinates, vtx_displacement, vtx_stress);
    return std::make_tuple(query_mesh_, query_mesh_.mat_coordinates, vtx_displacement, vtx_stress);
  }

 public:
  MatMesh3 mesh_;
  MatMesh3 query_mesh_;
  Eigen::AlignedBox3d design_domain_;
  std::shared_ptr<sha::NestedBackgroundMesh> nested_background_mesh_;
  std::shared_ptr<sha::CBNSimulator> simulator_;
  std::shared_ptr<sha::PhysicalDomain> physical_domain_;

 private:
  double YM1        = 1.0e5;
  double YM0        = 1.0e-5;
  double PR         = 0.3;
  double penaltyYM  = 1;
  double penaltyDBC = 1e10;
};
}  // namespace da

auto QuasistaticSimulationByCBN(std::string config_file, double YM, double PR, double density)
    -> std::tuple<da::MatMesh3, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> {
  using namespace da;  // NOLINT
  // ------------- Read Config ---------------
  std::ifstream config_infile(config_file);
  nlohmann::json config = nlohmann::json::parse(config_infile);

  fs_path base_path = std::string(config["working_dir"]);
  log::info("Algo is working on path '{}'", base_path.string());

  auto background_cells_path           = base_path / "macro";
  auto background_cell_tets_path       = base_path / "tets";
  auto background_cell_polyhedron_path = base_path / "polyhedrons";

  auto mesh_path         = base_path / "input" / std::string(config["mesh_path"]);
  auto query_mesh_path   = base_path / "query" / std::string(config["query_mesh_path"]);
  const size_t num_cells = config["cells"];
  // const double YM         = config["YM"];
  const double YM0 = config["YM0"];
  // const double PR         = config["PR"];
  const double penaltyYM  = config["penaltyYM"];
  const double penaltyDBC = config["penaltyDBC"];

  std::vector<Eigen::VectorXi> NBCIndex;
  std::vector<Eigen::Vector3d> NBCVal;
  std::vector<Eigen::VectorXi> DBCIndex;

  // NBC
  for (nlohmann::json nbc : config["NBC"]) {
    auto index_file = (base_path / "BC" / std::string(nbc["index_file"])).string();
    auto value_file = (base_path / "BC" / std::string(nbc["value_file"])).string();
    Eigen::VectorXi index;
    Eigen::Vector3d value;
    index = sha::ReadIntVectorFromFile(index_file);
    value = sha::ReadDoubleVectorFromFile(value_file);
    NBCIndex.emplace_back(index);
    NBCVal.emplace_back(value);
  }
  log::info("NBC: load {}", NBCIndex.size());

  // DBC
  for (nlohmann::json dbc : config["DBC"]) {
    auto index_file = (base_path / "BC" / std::string(dbc["index_file"])).string();
    Eigen::VectorXi index;
    index = sha::ReadIntVectorFromFile(index_file);
    DBCIndex.emplace_back(index);
  }
  log::info("DBC: load {}", DBCIndex.size());

  // rhos
  std::vector<Eigen::VectorXd> rhos(num_cells);
  for (int cell_idx = 0; cell_idx < num_cells; ++cell_idx) {
    auto cell_rho_path = base_path / "rhos" / fmt::format("rho{}.txt", cell_idx);
    rhos[cell_idx]     = sha::ReadVectorFromFile<double>(cell_rho_path);
  }

  // ------------- Read Config ---------------

  Worker woker(YM, YM0, PR, penaltyYM, penaltyDBC);
  log::info("Loading design domain");
  woker.LoadDesignDomainMesh(mesh_path, query_mesh_path);
  log::info("Loading background mesh");
  woker.LoadBackgroundMesh(background_cells_path, background_cell_polyhedron_path,
                           background_cell_tets_path, num_cells);
  log::info("Loading boundary conditions");
  woker.LoadBoundaryConditions(NBCIndex, NBCVal, DBCIndex);

  // // rhos
  // std::vector<Eigen::VectorXd> rhos(num_cells);
  // for (int cell_idx = 0; cell_idx < num_cells; ++cell_idx) {
  //   rhos[cell_idx].setOnes(
  //       woker.nested_background_mesh_->nested_cells_[cell_idx].tetrahedrons.NumTetrahedrons());
  //   // sha::WriteVectorToFile(
  //   //     WorkingResultDirectoryPath() / "rhos" / fmt::format("rho{}.txt", cell_idx), rhos[cell_idx]);
  // }

  log::info("Simulating");
  auto ret_tuple = woker.Run(base_path, rhos);
  log::info("Finish simulation!");

  return ret_tuple;
}
