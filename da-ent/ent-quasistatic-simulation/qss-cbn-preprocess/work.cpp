
#include "sha-base-framework/frame.h"
#include "sha-implicit-modeling/implicit.h"
#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-surface-mesh/mesh3.h"
#include "sha-volume-mesh/matmesh.h"

#include "nested_cages.h"

#include <vector>

#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/readOBJ.h>
#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>

#include <boost/progress.hpp>

size_t GenerateCBNBackgroundMesh(const std::string &fine_mesh_path,
                                 const std::string &output_dir_path,
                                 const std::vector<int> &level_num_faces,
                                 const std::string tetgen_macro_switch,
                                 const std::string tetgen_micro_switch) {
  using namespace da;  // NOLINT
  fs_path output_path            = output_dir_path;
  fs_path fine_mesh_input_path   = fine_mesh_path;
  fs_path coarse_tet_output_path = output_path / "coarse_tet.vtk";
  fs_path macro_mesh_output_path = output_path / "macro";
  fs_path polyhedron_output_path = output_path / "polyhedrons";
  fs_path tet_mesh_output_path   = output_path / "tet";
  fs_path level_mesh_output_path = output_path / "levels";

  std::vector<cage::LevelInfo> levels_info;
  for (auto level_num : level_num_faces) {
    levels_info.push_back(cage::LevelInfo{level_num, true});
  }

  auto CreateDirectoryIfNotExisted = [](const fs_path &dir_path) {
    if (boost::filesystem::exists(dir_path)) {
      boost::filesystem::remove_all(dir_path);
    }
    boost::filesystem::create_directory(dir_path);
  };
  CreateDirectoryIfNotExisted(macro_mesh_output_path);
  CreateDirectoryIfNotExisted(polyhedron_output_path);
  CreateDirectoryIfNotExisted(tet_mesh_output_path);
  CreateDirectoryIfNotExisted(level_mesh_output_path);

  cage::Mesh fine_mesh;
  igl::read_triangle_mesh(fine_mesh_input_path.string(), fine_mesh.V, fine_mesh.F);
  log::info("Fine mesh: #V:{}, #F:{}", fine_mesh.V.rows(), fine_mesh.F.rows());

  auto mesh_levels = cage::nested_cages(fine_mesh, 2, levels_info);

  for (index_t idx = 1; idx <= mesh_levels.size(); ++idx) {
    fs_path level_out_path = level_mesh_output_path / fmt::format("level_{}.obj", idx);
    igl::writeOBJ(level_out_path.string(), mesh_levels[idx - 1].V, mesh_levels[idx - 1].F);
  }

  sha::MatMesh3 coarse_shell{.mat_coordinates = mesh_levels.back().V,
                             .mat_faces       = mesh_levels.back().F};
  sha::TetrahedralMatMesh tetrahedral_coarse_shell;
  Eigen::MatrixXi mat_shell_surface_faces;
  log::info("tetrahedralizing...");
  igl::copyleft::tetgen::tetrahedralize(
      coarse_shell.mat_coordinates, coarse_shell.mat_faces, tetgen_macro_switch,
      tetrahedral_coarse_shell.mat_coordinates, tetrahedral_coarse_shell.mat_tetrahedrons,
      mat_shell_surface_faces);
  log::info("Tet #V:{}, #T:{}", tetrahedral_coarse_shell.NumVertices(),
            tetrahedral_coarse_shell.NumTetrahedrons());
  sha::WriteTetrahedralMatmeshToVtk(coarse_tet_output_path, tetrahedral_coarse_shell);

  Eigen::MatrixXi mat_micro_surface_triangle(4, 3);
  mat_micro_surface_triangle << 1, 0, 2,  //
      2, 0, 3,                            //
      3, 0, 1,                            //
      3, 1, 2;

  boost::progress_display cell_progress(tetrahedral_coarse_shell.NumTetrahedrons());

  size_t num_total_micro_tetrahedrons = 0;

  for (index_t cell_idx = 0; cell_idx < tetrahedral_coarse_shell.NumTetrahedrons(); ++cell_idx) {
    Eigen::MatrixXd mat_micro_surface_coordinates(4, 3);
    for (index_t vtx_idx = 0; vtx_idx < 4; ++vtx_idx) {
      mat_micro_surface_coordinates.row(vtx_idx) = tetrahedral_coarse_shell.mat_coordinates.row(
          tetrahedral_coarse_shell.mat_tetrahedrons(cell_idx, vtx_idx));
    }
    // Saving Macro OBJ
    igl::writeOBJ((macro_mesh_output_path / fmt::format("cell{}.obj", cell_idx)).string(),
                  mat_micro_surface_coordinates, mat_micro_surface_triangle);
    // Saving Macro Polyhedron
    {
      std::ofstream polyhedron_outstream(
          (polyhedron_output_path / fmt::format("polyhedron{}.txt", cell_idx)).string());
      polyhedron_outstream.precision(16);
      polyhedron_outstream << mat_micro_surface_coordinates.rows() << std::endl;
      polyhedron_outstream << mat_micro_surface_coordinates << std::endl;
      polyhedron_outstream << mat_micro_surface_triangle.rows() << std::endl;
      for (index_t micro_idx = 0; micro_idx < mat_micro_surface_triangle.rows(); ++micro_idx) {
        polyhedron_outstream << mat_micro_surface_triangle.cols() << std::endl;
        for (index_t edge_idx = 0; edge_idx < mat_micro_surface_triangle.cols(); ++edge_idx) {
          polyhedron_outstream << mat_micro_surface_triangle(micro_idx, edge_idx) << " "
                               << mat_micro_surface_triangle(
                                      micro_idx, (edge_idx + 1) % mat_micro_surface_triangle.cols())
                               << std::endl;
        }
      }
    }

    sha::TetrahedralMatMesh tetrahedral_micro_mesh;
    Eigen::MatrixXi mat_micro_surface_faces;
    igl::copyleft::tetgen::tetrahedralize(
        mat_micro_surface_coordinates, mat_micro_surface_triangle, tetgen_micro_switch,
        tetrahedral_micro_mesh.mat_coordinates, tetrahedral_micro_mesh.mat_tetrahedrons,
        mat_micro_surface_faces);
    sha::WriteTetrahedralMatmeshToVtk(tet_mesh_output_path / fmt::format("tet{}.vtk", cell_idx),
                                      tetrahedral_micro_mesh);

    ++cell_progress;
    num_total_micro_tetrahedrons += tetrahedral_micro_mesh.NumTetrahedrons();
  }
  log::info("Total macro: {}", tetrahedral_coarse_shell.NumTetrahedrons());
  log::info("Total micro: {}", num_total_micro_tetrahedrons);
  return tetrahedral_coarse_shell.NumTetrahedrons();
}