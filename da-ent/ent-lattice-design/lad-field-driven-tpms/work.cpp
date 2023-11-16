#include <igl/remove_unreferenced.h>
#include <igl/sharp_edges.h>
#include <igl/writeOBJ.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <iostream>
#include <utility>

#include "sha-base-framework/frame.h"
#include "sha-implicit-modeling/implicit.h"
#include "sha-implicit-modeling/field_tpms.h"
#include "sha-implicit-modeling/field_function.h"

#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-surface-mesh/mesh3.h"

#include "density.h"

namespace da {
class TPMSLatticeStructureProcessor {
 public:
  explicit TPMSLatticeStructureProcessor(const MatMesh3 &matmesh,
                                         const sha::SDFSampleDomain &sdf_sample_domain)
      : matmesh_(matmesh), sdf_sample_domain_(sdf_sample_domain) {}

  void GenerateTPMSSigendDistanceFieldForBoundingBox(const std::string &tpms_type, double tpms_coeff,
                                                     double offset, const std::string &field_type,
                                                     double field_coeff) {
    std::map<std::string, std::function<sha::FieldTPMSFunctions::FieldTPMSFunction(double)>>
        map_name_to_tpms = {{"G", sha::FieldTPMSFunctions::G},
                            {"G_rec", sha::FieldTPMSFunctions::G_rec},
                            {"Schwarzp", sha::FieldTPMSFunctions::Schwarzp},
                            {"DoubleP", sha::FieldTPMSFunctions::DoubleP},
                            {"Schwarzd", sha::FieldTPMSFunctions::Schwarzd},
                            {"DoubleD", sha::FieldTPMSFunctions::DoubleD}};
    
    std::map<std::string, std::function<sha::FieldFunctions::FieldFunction()>>
        map_name_to_field = {{"NoField", sha::FieldFunctions::NoField},
                             {"F1", sha::FieldFunctions::F1},
                             {"F2", sha::FieldFunctions::F2},
                             {"F3", sha::FieldFunctions::F3}};

    if (map_name_to_tpms.find(tpms_type) != map_name_to_tpms.end()) {
          if (field_type == "Matrix") {
            sha::ScalarField field_matrix;
            auto field_matrix_path   = WorkingResultDirectoryPath() / "field_matrix.txt";
            log::info("{}",field_matrix_path.string());
            field_matrix.LoadFrom(field_matrix_path.string());
            tpms_structure_sdf_ = sha::FieldDrivenDenseExpressionSDF(map_name_to_tpms[tpms_type](offset), tpms_coeff,
                                                  field_matrix, field_coeff, sdf_sample_domain_);
          }
          else if (map_name_to_field.find(field_type) != map_name_to_field.end()) {
            tpms_structure_sdf_ = sha::FieldDrivenDenseExpressionSDF(map_name_to_tpms[tpms_type](offset), tpms_coeff,
                                                  map_name_to_field[field_type](), field_coeff, sdf_sample_domain_);
          }
    }
  }

  void GenerateMeshSigendDistanceField() { mesh_sdf_ = sha::MeshSDF(matmesh_, sdf_sample_domain_); }

  MatMesh3 GenerateLatticeMesh() {
    sha::DenseSDF final_sdf  = tpms_structure_sdf_.Intersect(mesh_sdf_);
    auto marching_cubes_mesh = final_sdf.GenerateMeshByMarchingCubes(0);
    return marching_cubes_mesh;
  }

 public:
  const MatMesh3 matmesh_;
  sha::SDFSampleDomain sdf_sample_domain_;

 public:
  MatMesh2 shell_frame_mesh_;
  MatMesh2 lattice_mesh_;
  MatMesh3 microstructure_patch_mesh_;

 public:
  sha::FieldDrivenDenseExpressionSDF tpms_structure_sdf_;
  sha::MeshSDF mesh_sdf_;
};
}  // namespace da

auto GenerateTPMSLatticeStructureByField(size_t num_samples_on_longest_side, 
                                         const std::string &tpms_type,double tpms_coeff, 
                                         double offset, const std::string &field_type,
                                         double field_coeff, 
                                         bool reuse_mesh_sdf_flag) -> da::MatMesh3 {
  using namespace da;  // NOLINT
  auto processing_mesh_path   = WorkingAssetDirectoryPath() / "model.obj";
  auto lattice_out_path       = WorkingResultDirectoryPath() / "field-driven-tpms.obj";
  auto uncut_lattice_out_path = WorkingResultDirectoryPath() / "uncut_field-driven-tpms.obj";
  auto mesh_sdf_out_path      = WorkingResultDirectoryPath() / "mesh.sdf";

  MatMesh3 matmesh = sha::ReadMatMeshFromOBJ(processing_mesh_path);

  auto mesh_aligned_box    = matmesh.AlignedBox();
  double longest_side_step = mesh_aligned_box.sizes().maxCoeff() / num_samples_on_longest_side;

  Eigen::Vector3i num_xyz_samples(round(mesh_aligned_box.sizes().x() / longest_side_step),
                                  round(mesh_aligned_box.sizes().y() / longest_side_step),
                                  round(mesh_aligned_box.sizes().z() / longest_side_step));
  log::info("samples x: {}, y: {}, z: {}", num_xyz_samples.x(), num_xyz_samples.y(),
            num_xyz_samples.z());
  double diagonal = matmesh.AlignedBox().sizes().maxCoeff();
  sha::SDFSampleDomain sdf_sample_domain;
  sdf_sample_domain.num_samples = num_xyz_samples;
  sdf_sample_domain.domain      = matmesh.AlignedBox(0.1);
  log::info("sdf_sample_domain:{},{},{}",sdf_sample_domain.domain.min().x(),sdf_sample_domain.domain.min().y(),
            sdf_sample_domain.domain.min().z());

  TPMSLatticeStructureProcessor processor(matmesh, sdf_sample_domain);

  if (reuse_mesh_sdf_flag) {
    log::info("Reusing Mesh Sigend Distance Field");
    processor.mesh_sdf_.LoadFrom(mesh_sdf_out_path.string());
    processor.sdf_sample_domain_ = processor.mesh_sdf_.sdf_sample_domain_;
  } else {
    log::info("Generating Mesh Sigend Distance Field");
    processor.GenerateMeshSigendDistanceField();
    processor.mesh_sdf_.DumpTo(mesh_sdf_out_path.string());
  }

  log::info("Generating TPMS Sigend Distance Field For Bounding Box");
  processor.GenerateTPMSSigendDistanceFieldForBoundingBox(tpms_type, tpms_coeff, offset, field_type, field_coeff);
  log::info("TPMSType: {}, TPMSCoeff:{}, Offset: #{}, FieldType: {}, field_coeff: #{}", 
             tpms_type, tpms_coeff, offset, field_type, field_coeff);

  log::info("Generating Lattice Mesh");
  MatMesh3 lattice_mesh = processor.GenerateLatticeMesh();
  igl::writeOBJ(lattice_out_path.string(), lattice_mesh.mat_coordinates, lattice_mesh.mat_faces);

  log::info("Generated vertices: #{}, faces: #{}", lattice_mesh.NumVertices(),
            lattice_mesh.NumFaces());

  MatMesh3 uncut_lattice_mesh = processor.tpms_structure_sdf_.GenerateMeshByMarchingCubes(0);
  igl::writeOBJ(uncut_lattice_out_path.string(), uncut_lattice_mesh.mat_coordinates,
                uncut_lattice_mesh.mat_faces);

  // auto backgroundmesh_dir = WorkingAssetDirectoryPath() / "1";
  // ComputeRhosForBackgroundMesh(backgroundmesh_dir);
  return lattice_mesh;
}