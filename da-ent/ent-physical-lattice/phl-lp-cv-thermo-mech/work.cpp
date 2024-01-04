#include <igl/write_triangle_mesh.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <boost/progress.hpp>
#include <iostream>
#include "config.h"
#include "optimization.h"
#include "sha-implicit-modeling/implicit.h"
#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"
#include "utilis.h"

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

  void LoadBoundaryConditions(double p_YM, double p_PR, double p_TC, double p_TEC,
                              const std::vector<sha::DirichletBC> &p_mechDBC,
                              const std::vector<sha::NeumannBC> &p_mechNBC,
                              const std::vector<sha::DirichletBC> &p_thermalDBC,
                              const std::vector<sha::NeumannBC> &p_thermalNBC,
                              const std::shared_ptr<sha::CtrlPara> &p_para,
                              double shell_thickness) {
    simulator_ = std::make_shared<sha::ThermoelasticWrapper>(nested_background_mesh_, p_YM, p_PR,
                                                             p_TC, p_TEC, p_mechDBC, p_mechNBC,
                                                             p_thermalDBC, p_thermalNBC, p_para);
    // init model algo
    spdlog::debug("init model algo");
    model_algorithm_ =
        std::make_shared<ModelAlgorithm>(mesh_, *nested_background_mesh_, true, shell_thickness);
  }

  void Run(const Eigen::MatrixXd &mat_init_seeds, double init_radius,
           const std::pair<double, double> &radius_range, double scalar_E, double volfrac,
           size_t num_iterations, const fs_path &beams_out_path, const fs_path &seeds_out_path,
           const fs_path out_path, int mode) {
    Eigen::AlignedBox3d opt_domain;
    opt_domain.min()         = simulator_->getNode().colwise().minCoeff();
    opt_domain.max()         = simulator_->getNode().colwise().maxCoeff();
    anisotropic_mat_wrapper_ = std::make_shared<da::AnisotropicMatWrapper>();

    optimizer_ = std::make_shared<LpCVTOptimizer>(mesh_, p_, opt_domain, model_algorithm_,
                                                  simulator_, anisotropic_mat_wrapper_, init_radius,
                                                  radius_range, scalar_E, volfrac, mat_init_seeds);

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
          std::vector<double> beam_radius(optimizer_->GetModeling()->voronoi_beams_radiuses_.size());
          std::copy_n(optimizer_->GetModeling()->voronoi_beams_radiuses_.data(), optimizer_->GetModeling()->voronoi_beams_radiuses_.size(), beam_radius.data());
          sha::WriteToVtk(beams_out_path / fmt::format("beam{}.vtk", iteration_idx),
                          optimizer_->GetModeling()->voronoi_beams_mesh_.mat_coordinates,
                          optimizer_->GetModeling()->voronoi_beams_mesh_.mat_beams, {}, related_num,
                          3);
          sha::WriteToVtk(beams_out_path / fmt::format("beam_r{}.vtk", iteration_idx),
                          optimizer_->GetModeling()->voronoi_beams_mesh_.mat_coordinates,
                          optimizer_->GetModeling()->voronoi_beams_mesh_.mat_beams, {}, beam_radius,
                          3);                          
          sha::WritePointsToVtk(seeds_out_path / fmt::format("point{}.vtk", iteration_idx),
                                mat_variables.leftCols(3));

          if (iteration_idx == 0) return;
          sequential_E.push_back(E);
          sequential_V.push_back(V);
          sequential_C.push_back(C);
          // optimizer_->GetSimulation()->physicalDomain->WriteV1MeshToObj(
          //     WorkingResultDirectoryPath() / "V1" / fmt::format("V1_{}.obj", iteration_idx));
        });

    optimizer_->GetSimulation()->extractResult(out_path);

    sha::WriteVectorToFile(out_path / "C.txt",
                           sha::ConvertStlVectorToEigenVector(sequential_C));
    sha::WriteVectorToFile(out_path / "V.txt",
                           sha::ConvertStlVectorToEigenVector(sequential_V));
    sha::WriteVectorToFile(out_path / "E.txt",
                           sha::ConvertStlVectorToEigenVector(sequential_E));

    // output optimized model in rods
    Eigen::Vector3i num_xyz_samples(140, 200, 50);
    // Eigen::Vector3i num_xyz_samples(200, 200, 200);  //for 2-box-refine
    // Eigen::Vector3i num_xyz_samples(1000, 1000, 1000);  //max num for 2-box-refine
    // Eigen::Vector3i num_xyz_samples(120, 120, 120);  // for cube
    // log::info("samples x: {}, y: {}, z: {}", num_xyz_samples.x(), num_xyz_samples.y(),
    //           num_xyz_samples.z());
    Eigen::AlignedBox3d sdf_domain_bbox = opt_domain;
    sdf_domain_bbox.min().array() -= 10 * init_radius;
    sdf_domain_bbox.max().array() += 10 * init_radius;
    sha::SDFSampleDomain sdf_domain{.domain = sdf_domain_bbox, .num_samples = num_xyz_samples};
    sha::NonUniformFrameSDF sdf(optimizer_->GetModeling()->voronoi_beams_mesh_,
                                optimizer_->GetModeling()->voronoi_beams_radiuses_, sdf_domain);
    MatMesh3 mc_mesh = sdf.GenerateMeshByMarchingCubes(0.0);
    igl::write_triangle_mesh((out_path / "resulted_rods.obj").string(),
                             mc_mesh.mat_coordinates, mc_mesh.mat_faces);
    log::info("end generation of rod");
  }

 public:
  MatMesh3 mesh_;
  Eigen::AlignedBox3d design_domain_;
  std::shared_ptr<LpCVTOptimizer> optimizer_;
  std::shared_ptr<sha::NestedBackgroundMesh> nested_background_mesh_;
  std::shared_ptr<sha::ThermoelasticWrapper> simulator_;
  std::shared_ptr<ModelAlgorithm> model_algorithm_;
  std::shared_ptr<da::AnisotropicMatWrapper> anisotropic_mat_wrapper_;

 private:
  int p_;

  size_t num_seeds_;
  size_t num_variables_;
};
}  // namespace da

void GeneratePhysicalLpNormVoronoiLatticeStructure(int mode) {
  using namespace da;  // NOLINT
  fs_path base_path = WorkingDirectory();
  auto config_path  = base_path / "da-ent/ent-physical-lattice/phl-lp-cv-thermo-mech/config.json";

  // int mode = 1; // 0 for iso, 1 for fem, 2 for top density, 3 for top stress

  const int p = 6;

  // ------------- Read Config ---------------
  // load config from json file
  Config config;
  if (!config.loadFromJSON(config_path.string())) {
    spdlog::error("error on reading json file!");
    exit(-1);
  }

  std::string condition = config.condition;
  log::info("Algo is working on path '{}' and example '{}'", base_path.string(), condition);

  std::string level = config.level;
  spdlog::set_level(spdlog::level::from_str(level));

  auto output_path                     = config.outputPath;
  auto mesh_path                       = config.meshFilePath;
  auto seeds_path                      = config.seedsPath;
  auto background_cells_path           = config.backgroundCellsPath;
  auto background_cell_tets_path       = config.cellTetsPath;
  auto background_cell_polyhedron_path = config.cellPolyhedronPath;

  auto beams_out_path = output_path + "/beams";
  auto seeds_out_path = output_path + "/points";
  config.backUpConfig(output_path);

  std::shared_ptr<sha::CtrlPara> para = config.para;

  size_t num_iterations = para->max_loop;
  const double scalar_E = para->E;
  const double volfrac  = para->volfrac;

  const size_t num_cells       = config.cellNum;
  const double shell_thickness = config.shell;

  double init_radius                     = config.radius[0];
  std::pair<double, double> radius_range = {config.radius[1], config.radius[0]};

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

  woker.LoadBoundaryConditions(config.YM, config.PR, config.TC, config.TEC,
                               config.mechanicalDirichletBCs, config.mechanicalNeumannBCs,
                               config.thermalDirichletBCs, config.thermalNeumannBCs, para,
                               shell_thickness);

  log::info("Optimizing");
  woker.Run(mat_seeds, init_radius, radius_range, scalar_E, volfrac, num_iterations, beams_out_path,
            seeds_out_path, output_path, mode);
}