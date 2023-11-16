#include <igl/remove_unreferenced.h>
#include <igl/sharp_edges.h>
#include <igl/writeOBJ.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <iostream>
#include <utility>

#include "sha-base-framework/frame.h"
#include "sha-implicit-modeling/implicit.h"
#include "sha-implicit-modeling/tpms_family.h"

#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-surface-mesh/mesh3.h"

namespace da {
class TPMSLatticeStructureProcessor {
 public:
  explicit TPMSLatticeStructureProcessor(const MatMesh3 &matmesh,
                                         const sha::SDFSampleDomain &sdf_sample_domain)
      : matmesh_(matmesh), sdf_sample_domain_(sdf_sample_domain) {}

  void GenerateTPMSSigendDistanceFieldForBoundingBox(const std::string &tpms_type, double coeffient,
                                                     double offset) {
    std::map<std::string, std::function<sha::TPMSFunctions::TPMSFunction(double, double)>>
        map_name_to_tpms = {{"Schwarzp", sha::TPMSFunctions::Schwarzp},
                            {"DoubleP", sha::TPMSFunctions::DoubleP},
                            {"Schwarzd", sha::TPMSFunctions::Schwarzd},
                            {"DoubleD", sha::TPMSFunctions::DoubleD}};
    if (map_name_to_tpms.find(tpms_type) != map_name_to_tpms.end()) {
      tpms_structure_sdf_ = sha::DenseExpressionSDF(map_name_to_tpms[tpms_type](coeffient, offset),
                                                    sdf_sample_domain_);
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
  sha::DenseExpressionSDF tpms_structure_sdf_;
  sha::MeshSDF mesh_sdf_;
};
}  // namespace da

auto GenerateTPMSLatticeStructure(size_t num_samples_on_longest_side, double coeffient,
                                  double offset, const std::string &tpms_type,
                                  bool reuse_mesh_sdf_flag) -> da::MatMesh3 {
  using namespace da;  // NOLINT

  auto processing_mesh_path   = WorkingAssetDirectoryPath() / "model.obj";
  auto lattice_out_path       = WorkingResultDirectoryPath() / "lattice-tpms.obj";
  auto uncut_lattice_out_path = WorkingResultDirectoryPath() / "uncut_lattice-tpms.obj";
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
  sdf_sample_domain.domain      = matmesh.AlignedBox(0.02 * diagonal);

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
  processor.GenerateTPMSSigendDistanceFieldForBoundingBox(tpms_type, coeffient, offset);

  log::info("Generating Lattice Mesh");
  MatMesh3 lattice_mesh = processor.GenerateLatticeMesh();
  igl::writeOBJ(lattice_out_path.string(), lattice_mesh.mat_coordinates, lattice_mesh.mat_faces);

  log::info("Generated vertices: #{}, faces: #{}", lattice_mesh.NumVertices(),
            lattice_mesh.NumFaces());

  MatMesh3 uncut_lattice_mesh = processor.tpms_structure_sdf_.GenerateMeshByMarchingCubes(0);
  igl::writeOBJ(uncut_lattice_out_path.string(), uncut_lattice_mesh.mat_coordinates,
                uncut_lattice_mesh.mat_faces);
  return lattice_mesh;
}
