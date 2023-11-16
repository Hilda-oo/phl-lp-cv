#include <igl/remove_unreferenced.h>
#include <igl/sharp_edges.h>
#include <igl/writeOBJ.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <iostream>
#include <utility>

#include "sha-base-framework/frame.h"
#include "sha-implicit-modeling/implicit.h"
#include "sha-implicit-modeling/field_function.h"

#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-surface-mesh/mesh3.h"

#include "utility.h"
#include "density.h"


namespace da {
class FeaturePreservedLatticeStructureProcessor {
 public:
  explicit FeaturePreservedLatticeStructureProcessor(const MatMesh3 &matmesh,
                                                     const sha::SDFSampleDomain &sdf_sample_domain)
      : matmesh_(matmesh),
        mesh_bounding_box_(matmesh_.mat_coordinates.colwise().minCoeff(),
                           matmesh_.mat_coordinates.colwise().maxCoeff()),
        sdf_sample_domain_(sdf_sample_domain) {
    auto sizes = mesh_bounding_box_.sizes();
    mesh_bounding_box_.min().array() -= sizes.array() * 0.1;
    mesh_bounding_box_.max().array() += sizes.array() * 0.1;
  }

  void ExtractSharpEdges(double angle) {
    Eigen::MatrixXi mat_beams, mat_beam_indices;
    igl::sharp_edges(matmesh_.mat_coordinates, matmesh_.mat_faces, angle / 180.0 * igl::PI,
                     mat_beams);
    igl::remove_unreferenced(matmesh_.mat_coordinates, mat_beams, sharp_edges_mesh_.mat_coordinates,
                             sharp_edges_mesh_.mat_beams, mat_beam_indices);
  }

  void ExtractCutLines(size_t structure_type, const Eigen::Vector3i &structure_num) {
    microstructure_patch_mesh_ = sha::MicroStructureSDF::MakeMicroStructurePatchMesh(
        structure_num, mesh_bounding_box_, structure_type);
    cut_lines_mesh_ = ComputeCommonLinesFromTwoMeshes(microstructure_patch_mesh_, this->matmesh_);
  }

  void GenerateLatticeSigendDistanceFieldForBoundingBox(size_t structure_type,
                                                        const Eigen::Vector3i &structure_num,
                                                        double radius, const std::string &field_type,
                                                        double field_coeff) {
    std::map<std::string, std::function<sha::FieldFunctions::FieldFunction()>>
      map_name_to_field = {{"NoField", sha::FieldFunctions::NoField},
                            {"F1", sha::FieldFunctions::F1},
                            {"F2", sha::FieldFunctions::F2},
                            {"F3", sha::FieldFunctions::F3}};

    if (field_type == "Matrix") {
      sha::ScalarField field_matrix;
      auto field_matrix_path   = WorkingResultDirectoryPath() / "field_matrix.txt";
      log::info("{}",field_matrix_path.string());
      field_matrix.LoadFrom(field_matrix_path.string());
      lattice_structure_sdf_ = sha::NonUniformMicroStructureSDF(structure_type, structure_num,
                                                    mesh_bounding_box_, radius,
                                                    field_matrix,
                                                    field_coeff, sdf_sample_domain_);
    }
    else if (map_name_to_field.find(field_type) != map_name_to_field.end()) {
      lattice_structure_sdf_ = sha::NonUniformMicroStructureSDF(structure_type, structure_num,
                                                    mesh_bounding_box_, radius,
                                                    map_name_to_field[field_type](),
                                                    field_coeff, sdf_sample_domain_);
    }

    lattice_mesh_          = lattice_structure_sdf_.beam_mesh_;
  }

  void GenerateLatticeSigendDistanceFieldForShell(double radius) {
    shell_frame_mesh_  = sha::CombineTwoMatMesh2(cut_lines_mesh_, sharp_edges_mesh_);
    shell_lattice_sdf_ = sha::FrameSDF(shell_frame_mesh_, radius, sdf_sample_domain_);
  }

  void GenerateMeshSigendDistanceField() { mesh_sdf_ = sha::MeshSDF(matmesh_, sdf_sample_domain_); }

  MatMesh3 GenerateLatticeMesh() {
    // sha::DenseSDF shell_sdf = lattice_structure_sdf_;
    sha::DenseSDF shell_sdf = lattice_structure_sdf_.Union(shell_lattice_sdf_);
    sha::DenseSDF final_sdf = shell_sdf.Intersect(mesh_sdf_);

    auto marching_cubes_mesh = final_sdf.GenerateMeshByMarchingCubes(0);
    return marching_cubes_mesh;
  }

 public:
  const MatMesh3 matmesh_;
  Eigen::AlignedBox3d mesh_bounding_box_;
  sha::SDFSampleDomain sdf_sample_domain_;
  MatMesh2 sharp_edges_mesh_;
  MatMesh2 cut_lines_mesh_;

 public:
  MatMesh2 shell_frame_mesh_;
  MatMesh2 lattice_mesh_;
  MatMesh3 microstructure_patch_mesh_;

 public:
  sha::NonUniformMicroStructureSDF lattice_structure_sdf_;
  sha::FrameSDF shell_lattice_sdf_;
  sha::MeshSDF mesh_sdf_;
};
}  // namespace da

auto GenerateVoxelLatticeStructureByField(size_t num_samples_on_longest_side,
                                           size_t num_cells_along_x_axis,
                                           size_t num_cells_along_y_axis,
                                           size_t num_cells_along_z_axis,
                                           size_t lattice_structure_type,
                                           const std::string &field_type,
                                           double field_coeff, double radius_of_lattice_frame,
                                           double radius_of_shell_frame, double sharp_angle,
                                           bool reuse_mesh_sdf_flag)
    -> da::MatMesh3 {
  using namespace da;  // NOLINT

  auto processing_mesh_path   = WorkingAssetDirectoryPath() / "model.obj";
  auto lattice_out_path       = WorkingResultDirectoryPath() / "field-driven-voxel.obj";
  auto uncut_lattice_out_path = WorkingResultDirectoryPath() / "uncut_field-driven-voxel.obj";
  auto shell_frame_coordinates_out_path =
      WorkingResultDirectoryPath() / "shell_frame_coordinates.txt";
  auto shell_frame_beams_out_path = WorkingResultDirectoryPath() / "shell_frame_beams.txt";
  auto shell_frame_out_path       = WorkingResultDirectoryPath() / "shell_frame.vtk";
  auto lattice_frame_out_path     = WorkingResultDirectoryPath() / "lattice_frame.vtk";
  auto microstructure_patch_mesh_out_path =
      WorkingResultDirectoryPath() / "microstructure_patch.obj";
  auto lattice_coordinates_out_path = WorkingResultDirectoryPath() / "lattice_coordinates.txt";
  auto lattice_beams_out_path       = WorkingResultDirectoryPath() / "lattice_beams.txt";
  auto mesh_sdf_out_path            = WorkingResultDirectoryPath() / "mesh.sdf";

  // sha::MicroStructureSDF::StructBasePath =
  //     (ProjectAssetDirectoryPath() / "MicrostructureBase").string();

  log::info("Processing Mesh: '{}'", processing_mesh_path.string());
  log::info("Out Lattice Mesh: '{}'", lattice_out_path.string());
  // log::info("Structure Base Path: '{}'", sha::MicroStructureSDF::StructBasePath);

  MatMesh3 matmesh = sha::ReadMatMeshFromOBJ(processing_mesh_path);

  auto mesh_aligned_box    = matmesh.AlignedBox();
  double longest_side_step = mesh_aligned_box.sizes().maxCoeff() / num_samples_on_longest_side;

  Eigen::Vector3i num_xyz_samples(round(mesh_aligned_box.sizes().x() / longest_side_step),
                                  round(mesh_aligned_box.sizes().y() / longest_side_step),
                                  round(mesh_aligned_box.sizes().z() / longest_side_step));
  log::info("samples x: {}, y: {}, z: {}", num_xyz_samples.x(), num_xyz_samples.y(),
            num_xyz_samples.z());
  Eigen::Vector3i num_xyz_cells(num_cells_along_x_axis, num_cells_along_y_axis,
                                num_cells_along_z_axis);

  sha::SDFSampleDomain sdf_sample_domain;
  sdf_sample_domain.num_samples = num_xyz_samples;
  sdf_sample_domain.domain      = matmesh.AlignedBox(0.1);
  log::info("sdf_sample_domain:{},{},{}",sdf_sample_domain.domain.min().x(),sdf_sample_domain.domain.min().y(),
            sdf_sample_domain.domain.min().z());

  FeaturePreservedLatticeStructureProcessor processor(matmesh, sdf_sample_domain);

  if (reuse_mesh_sdf_flag) {
    log::info("Reusing Mesh Sigend Distance Field");
    processor.mesh_sdf_.LoadFrom(mesh_sdf_out_path.string());
    processor.sdf_sample_domain_ = processor.mesh_sdf_.sdf_sample_domain_;
  } else {
    log::info("Generating Mesh Sigend Distance Field");
    processor.GenerateMeshSigendDistanceField();
    processor.mesh_sdf_.DumpTo(mesh_sdf_out_path.string());
  }
  
  if (radius_of_lattice_frame > 0){
    log::info("Generating Lattice Sigend Distance Field For Bounding Box");
    processor.GenerateLatticeSigendDistanceFieldForBoundingBox(
        lattice_structure_type, num_xyz_cells, radius_of_lattice_frame, field_type, field_coeff);
    log::info("radius:{},FieldType:{},FieldCoeff:{}",radius_of_lattice_frame, field_type, field_coeff);
  }

  if (radius_of_shell_frame > 0) {
    log::info("Extracting Cut lines");
    processor.ExtractCutLines(lattice_structure_type, num_xyz_cells);

    log::info("Extracting Sharp Edges");
    processor.ExtractSharpEdges(sharp_angle);

    log::info("Generating Lattice Sigend Distance Field For Shell");
    processor.GenerateLatticeSigendDistanceFieldForShell(radius_of_shell_frame);
  }

  log::info("Generating Lattice Mesh");
  MatMesh3 lattice_mesh = processor.GenerateLatticeMesh();
  igl::writeOBJ(lattice_out_path.string(), lattice_mesh.mat_coordinates, lattice_mesh.mat_faces);

  log::info("Generated vertices: #{}, faces: #{}", lattice_mesh.NumVertices(),
            lattice_mesh.NumFaces());

  MatMesh3 uncut_lattice_mesh = processor.lattice_structure_sdf_.GenerateMeshByMarchingCubes(0);
  igl::writeOBJ(uncut_lattice_out_path.string(), uncut_lattice_mesh.mat_coordinates,
                uncut_lattice_mesh.mat_faces);
  sha::WriteMatMesh2ToVtk(shell_frame_out_path, processor.shell_frame_mesh_);
  sha::WriteMatMesh2ToVtk(lattice_frame_out_path, processor.lattice_mesh_);
  // igl::writeOBJ(microstructure_patch_mesh_out_path.string(),
  //               processor.microstructure_patch_mesh_.mat_coordinates,
  //               processor.microstructure_patch_mesh_.mat_faces);
  
  // auto backgroundmesh_dir = WorkingAssetDirectoryPath() / "2-box";
  // ComputeRhosForBackgroundMesh(backgroundmesh_dir);
  
  return lattice_mesh;
}
