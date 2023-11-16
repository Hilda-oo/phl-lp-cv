#include <chrono>
#include <memory>

#include <fmt/format.h>
#include <igl/write_triangle_mesh.h>
#include <spdlog/spdlog.h>

#include "sha-base-framework/declarations.h"
#include "sha-base-framework/frame.h"
#include "sha-fem-quasistatic/fem_tet.h"
#include "sha-fem-quasistatic/fem_tet/simulation.h"
#include "sha-implicit-modeling/implicit.h"
#include "sha-simulation-3d/CBN.h"
#include "sha-simulation-utils/io_utils.h"

#include "fem_config.h"
#include "modeling.h"
#include "optimization.h"
#include "utilis.h"

#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"

#include <boost/progress.hpp>

#include <json.hpp>

std::string WorkingDirectory();

namespace da {
class Worker {
 public:
  explicit Worker(size_t num_seeds, size_t num_varibales, int p)
      : num_seeds_(num_seeds), num_variables_(num_varibales), p_(p) {}

  void LoadDesignDomainMesh(const fs_path &mesh_path) {
    mesh_ = sha::ReadMatMeshFromOBJ(mesh_path);
    log::info("mesh num v: {}", mesh_.NumVertices());
    log::info("mesh num f: {}", mesh_.NumFaces());

    design_domain_.min() = mesh_.mat_coordinates.colwise().minCoeff();
    design_domain_.max() = mesh_.mat_coordinates.colwise().maxCoeff();
  }

  void LoadBackgroundMesh(const fs_path &background_cells_path,
                          const fs_path &background_polyhedrons_path,
                          const fs_path &background_cell_tets_path, size_t num_cells) {
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

  void LoadBoundaryConditions(const std::vector<Eigen::Matrix<double, 2, 3>> &NBCRelBBox,
                              const std::vector<Eigen::Vector3d> &NBCVal,
                              const std::vector<Eigen::Matrix<double, 2, 3>> &DBCRelBBox,
                              const std::vector<Eigen::Vector3d> &DBCVal, double shell_thickness) {
    physical_domain_ =
        std::make_shared<sha::PhysicalDomain>(mesh_, NBCRelBBox, NBCVal, DBCRelBBox, DBCVal);
    physical_domain_->WriteNBCToVtk(WorkingResultDirectoryPath() / "NBC.vtk");

    // init model algo
    model_algorithm_ =
        std::make_shared<ModelAlgorithm>(mesh_, *nested_background_mesh_, true, shell_thickness);

    // init CBN background
    simulator_ = std::make_shared<sha::CBNSimulator>(YM1, YM0, PR, penaltyYM, physical_domain_,
                                                     nested_background_mesh_);

    sha::WritePointsToVtk(WorkingResultDirectoryPath() / "direct_DBC.vtk", simulator_->DBCV);
  }

  void LoadFEMSimulator(const FEMConfig &fem_config) {
    Eigen::MatrixXd TV;
    Eigen::MatrixXi TT;
    Eigen::MatrixXi SF;
    if (!sha::ReadTetMesh(fem_config.mshFilePath, TV, TT, SF)) {
      log::error("error on reading tetrahedron mesh!");
      exit(-1);
    }
    fem_simulator_ = std::make_shared<sha::FEMTetQuasiSimulator>(
        TV, TT, SF, fem_config.YM, fem_config.PR, fem_config.DirichletBCs, fem_config.NeumannBCs);
  }

  void Run(const Eigen::MatrixXd &mat_init_seeds, double init_radius,
           const std::pair<double, double> &radius_range, double scalar_E, double volfrac,
           size_t num_iterations, const fs_path &beams_out_path, const fs_path &seeds_out_path, int mode) {
    Eigen::AlignedBox3d opt_domain;
    opt_domain.min() = simulator_->node.colwise().minCoeff();
    opt_domain.max() = simulator_->node.colwise().maxCoeff();
    anisotropic_mat_wrapper_ = std::make_shared<da::AnisotropicMatWrapper>();

    switch (mode) {
      case 1:
        optimizer_ = std::make_shared<LpCVTOptimizer>(mesh_, p_, opt_domain, model_algorithm_,
                                                    simulator_, fem_simulator_, anisotropic_mat_wrapper_, init_radius,
                                                    radius_range, scalar_E, volfrac, mat_init_seeds);
        break;                                                  
      default: 
        optimizer_ = std::make_shared<LpCVTOptimizer>(mesh_, p_, opt_domain, model_algorithm_,
                                                    simulator_, anisotropic_mat_wrapper_, init_radius,
                                                    radius_range, scalar_E, volfrac, mat_init_seeds);
        break;
    }
    Eigen::MatrixXd mat_seeds;
    Eigen::VectorXd radiuses;

    std::vector<double> sequential_E, sequential_V, sequential_C;

    optimizer_->Optimize(
        mat_seeds, radiuses, num_iterations, mode,
        [&](index_t iteration_idx, const Eigen::MatrixXd &mat_variables, double C,
            const Eigen::VectorXd &dC, double E, const Eigen::VectorXd &dE, double V,
            const Eigen::VectorXd &dV) {
          std::vector<double> related_num(
              optimizer_->GetModeling()->map_voronoi_beam_idx_to_cell_indices.size());
          for (index_t idx = 0; idx < optimizer_->GetModeling()->voronoi_beams_mesh_.NumBeams();
               ++idx) {
            related_num[idx] =
                optimizer_->GetModeling()->map_voronoi_beam_idx_to_cell_indices[idx].size();
          }
          sha::WriteToVtk(beams_out_path / fmt::format("beam{}.vtk", iteration_idx),
                          optimizer_->GetModeling()->voronoi_beams_mesh_.mat_coordinates,
                          optimizer_->GetModeling()->voronoi_beams_mesh_.mat_beams, {}, related_num,
                          3);
          sha::WritePointsToVtk(seeds_out_path / fmt::format("point{}.vtk", iteration_idx),
                                mat_variables.leftCols(3));

          if (iteration_idx == 0) return;
          sequential_E.push_back(E);
          sequential_V.push_back(V);
          sequential_C.push_back(C);
          optimizer_->GetSimulation()->physicalDomain->WriteV1MeshToObj(
              WorkingResultDirectoryPath() / "V1" / fmt::format("V1_{}.obj", iteration_idx));
        });

    sha::WriteVectorToFile(WorkingResultDirectoryPath() / "C.txt",
                           sha::ConvertStlVectorToEigenVector(sequential_C));
    sha::WriteVectorToFile(WorkingResultDirectoryPath() / "V.txt",
                           sha::ConvertStlVectorToEigenVector(sequential_V));
    sha::WriteVectorToFile(WorkingResultDirectoryPath() / "E.txt",
                           sha::ConvertStlVectorToEigenVector(sequential_E));

    // output optimized model in rods
    // Eigen::Vector3i num_xyz_samples(300, 300, 300);
    // Eigen::Vector3i num_xyz_samples(200, 200, 200);  //for 2-box-refine
    // Eigen::Vector3i num_xyz_samples(1000, 1000, 1000);  //max num for 2-box-refine
    Eigen::Vector3i num_xyz_samples(120, 120, 120);  //for cube
    // log::info("samples x: {}, y: {}, z: {}", num_xyz_samples.x(), num_xyz_samples.y(),
    //           num_xyz_samples.z());
    Eigen::AlignedBox3d sdf_domain_bbox = opt_domain;
    sdf_domain_bbox.min().array() -= 10 * init_radius;
    sdf_domain_bbox.max().array() += 10 * init_radius;
    sha::SDFSampleDomain sdf_domain{.domain = sdf_domain_bbox, .num_samples = num_xyz_samples};
    sha::NonUniformFrameSDF sdf(optimizer_->GetModeling()->voronoi_beams_mesh_,
                                optimizer_->GetModeling()->voronoi_beams_radiuses_, sdf_domain);
    sha::WriteMatrixToFile(WorkingResultDirectoryPath() / "processVoronoiEdge/voronoi_beams_mesh_vertex.txt", optimizer_->GetModeling()->voronoi_beams_mesh_.mat_coordinates);
    sha::WriteMatrixToFile(WorkingResultDirectoryPath() / "processVoronoiEdge/voronoi_beams_mesh_beams.txt", optimizer_->GetModeling()->voronoi_beams_mesh_.mat_beams);
    sha::WriteVectorToFile(WorkingResultDirectoryPath() / "processVoronoiEdge/voronoi_beams_radiuses.txt", optimizer_->GetModeling()->voronoi_beams_radiuses_);
    MatMesh3 mc_mesh = sdf.GenerateMeshByMarchingCubes(0.0);
    igl::write_triangle_mesh((WorkingResultDirectoryPath() / "resulted_rods.obj").string(),
                             mc_mesh.mat_coordinates, mc_mesh.mat_faces);
    log::info("end generation of rod");                             
  }

 public:
  MatMesh3 mesh_;
  Eigen::AlignedBox3d design_domain_;
  std::shared_ptr<LpCVTOptimizer> optimizer_;
  std::shared_ptr<sha::NestedBackgroundMesh> nested_background_mesh_;
  std::shared_ptr<sha::CBNSimulator> simulator_;
  std::shared_ptr<sha::PhysicalDomain> physical_domain_;
  std::shared_ptr<ModelAlgorithm> model_algorithm_;
  std::shared_ptr<sha::FEMTetQuasiSimulator> fem_simulator_;
  std::shared_ptr<da::AnisotropicMatWrapper> anisotropic_mat_wrapper_;

 private:
  double YM1        = 1.0e5;
  double YM0        = 1.0e-5;
  double PR         = 0.3;
  double penaltyYM  = 1;
  double penaltyDBC = 1e10;

  int p_;

  size_t num_seeds_;
  size_t num_variables_;
};
}  // namespace da

void GeneratePhysicalLpNormVoronoiLatticeStructure(std::string model_file_name, std::string model_name, int seed_num, int mode) {
  using namespace da;  // NOLINT
  fs_path base_path                    = WorkingDirectory();
  auto mesh_path                       = base_path / model_file_name / "input" / model_name;
  auto seeds_path                      = base_path / model_file_name / fmt::format("seed-{}.txt", seed_num);
  auto background_cells_path           = base_path / model_file_name / "macro";
  auto background_cell_tets_path       = base_path / model_file_name / "tet";
  auto background_cell_polyhedron_path = base_path / model_file_name / "polyhedrons";
  auto beams_out_path                  = WorkingResultDirectoryPath() / "beams";
  auto seeds_out_path                  = WorkingResultDirectoryPath() / "points";
  auto config_path                     = base_path / model_file_name / "config.json";

  // int mode = 1; // 0 for iso, 1 for fem, 2 for top density, 3 for top stress

  const int p = 6;

  log::info("Algo is working on path '{}'", base_path.string());

  // ------------- Read Config ---------------
  std::ifstream config_infile(config_path.string());
  nlohmann::json config = nlohmann::json::parse(config_infile);

  size_t num_iterations = config["iterations"];
  double init_radius    = config["radius"]["init"];
  auto radius_range =
      std::make_pair<double, double>(config["radius"]["min"], config["radius"]["max"]);
  const double scalar_E        = config["E"];
  const double volfrac         = config["volfrac"];
  const size_t num_cells       = config["cells"];
  const double shell_thickness = config["shell"];

  std::vector<Eigen::Matrix<double, 2, 3>> NBCRelBBox;
  std::vector<Eigen::Vector3d> NBCVal;
  std::vector<Eigen::Matrix<double, 2, 3>> DBCRelBBox;
  std::vector<Eigen::Vector3d> DBCVal;

  // NBC
  for (nlohmann::json nbc : config["NBC"]) {
    Eigen::Matrix<double, 2, 3> nbc_box;
    Eigen::Vector3d nbc_value;
    nbc_box.row(0) << nbc["min"][0], nbc["min"][1], nbc["min"][2];
    nbc_box.row(1) << nbc["max"][0], nbc["max"][1], nbc["max"][2];
    nbc_value << nbc["val"][0], nbc["val"][1], nbc["val"][2];
    NBCRelBBox.push_back(nbc_box);
    NBCVal.push_back(nbc_value);
  }

  log::info("NBC: load {}", NBCRelBBox.size());

  // DBC
  for (nlohmann::json dbc : config["DBC"]) {
    Eigen::Matrix<double, 2, 3> dbc_box;
    Eigen::Vector3d dbc_value;
    dbc_box.row(0) << dbc["min"][0], dbc["min"][1], dbc["min"][2];
    dbc_box.row(1) << dbc["max"][0], dbc["max"][1], dbc["max"][2];
    dbc_value << 0, 0, 0;
    DBCRelBBox.push_back(dbc_box);
    DBCVal.push_back(dbc_value);
  }
  log::info("DBC: load {}", DBCRelBBox.size());

  if (!boost::filesystem::exists(beams_out_path)) {
    boost::filesystem::create_directory(beams_out_path);
  }

  if (!boost::filesystem::exists(seeds_out_path)) {
    boost::filesystem::create_directory(seeds_out_path);
  }

  Eigen::MatrixXd mat_seeds = sha::ReadDoubleMatrixFromFile(seeds_path);

  size_t num_seeds     = mat_seeds.rows();
  size_t num_variables = num_seeds * 4;

  Worker woker(num_seeds, num_variables, p);
  log::info("Loading design domain");
  woker.LoadDesignDomainMesh(mesh_path);

  log::info("Loading background mesh");
  woker.LoadBackgroundMesh(background_cells_path, background_cell_polyhedron_path,
                           background_cell_tets_path, num_cells);
  log::info("Loading boundary conditions");
  woker.LoadBoundaryConditions(NBCRelBBox, NBCVal, DBCRelBBox, DBCVal, shell_thickness);

   if(mode == 1){
      // fem
      auto fem_config_path = base_path / model_file_name / "fem" / "config.json";
      // fem config
      FEMConfig fem_config;
      if (!fem_config.loadFromJSON(fem_config_path.string())) {
        log::error("error on reading fem config!");
        exit(-1);
      }
        log::info("Loading fem simulator");
      woker.LoadFEMSimulator(fem_config);
   }

  log::info("Optimizing");
  woker.Run(mat_seeds, init_radius, radius_range, scalar_E, volfrac, num_iterations, beams_out_path,
            seeds_out_path, mode);
}

void GenerateSampleSeed(int num_seed, std::string model_file, std::string model_name){
  // da::fs_path path = "femur/input/femur.obj";
  da::fs_path outpath = model_file;
  da::fs_path path = outpath / "input" / model_name;
  da::AnisotropicMatWrapper anisotropicMatWrapper;
  anisotropicMatWrapper.generateSampleSeedEntry(path, outpath, num_seed);
}