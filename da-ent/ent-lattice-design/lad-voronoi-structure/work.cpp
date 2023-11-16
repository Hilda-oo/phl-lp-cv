#include <igl/writeOBJ.h>
#include <spdlog/spdlog.h>
#include <Eigen/Eigen>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include <boost/assign.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include "sha-base-framework/declarations.h"
#include "sha-implicit-modeling/implicit.h"
#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/sample.h"
#include "sha-voronoi-foundation/fast_voronoi.h"
#include "sha-voronoi-foundation/voronoi.h"

namespace da {
class VoronoiLatticeProcessor {
 public:
  explicit VoronoiLatticeProcessor(const MatMesh3 &matmesh,
                                   const sha::SDFSampleDomain sdf_sample_domain)
      : matmesh_(matmesh),
        mesh_bounding_box_(matmesh_.mat_coordinates.colwise().minCoeff(),
                           matmesh_.mat_coordinates.colwise().maxCoeff()),
        sdf_sample_domain_(sdf_sample_domain) {}

  void GenerateMeshSignedDistanceField() { mesh_sdf_ = sha::MeshSDF(matmesh_, sdf_sample_domain_); }

  void GenerateVoronoiBeamsMesh(const size_t num_seeds, double sharp_angle) {
    mat_voronoi_seeds_ = sha::SamplePointsInMeshVolumeUniformly(matmesh_, num_seeds);
    log::info("#seed: {}", mat_voronoi_seeds_.rows());
    auto voronoi =
        FastCreateRestrictedVoronoiDiagramFromMesh(matmesh_, mat_voronoi_seeds_, sharp_angle);
    log::info("#voronoi: {}", voronoi.cells.size());
    voronoi_beams_mesh_ = ComputeRelatedEdgesFromVoronoiDiagram(voronoi, mesh_bounding_box_);
    log::info("#beam: {}", voronoi_beams_mesh_.NumBeams());
  }

  void GenerateLatticeSignedDistanceField(double radius) {
    lattice_sdf_ = sha::FrameSDF(voronoi_beams_mesh_, radius, sdf_sample_domain_);
  }

  MatMesh3 GenerateLatticeMesh() {
    sha::DenseSDF final_sdf  = lattice_sdf_.Intersect(mesh_sdf_);
    auto marching_cubes_mesh = final_sdf.GenerateMeshByMarchingCubes(0);
    return marching_cubes_mesh;
  }

 public:
  const MatMesh3 matmesh_;
  const Eigen::AlignedBox3d mesh_bounding_box_;
  const sha::SDFSampleDomain sdf_sample_domain_;
  Eigen::MatrixXd mat_voronoi_seeds_;

 public:
  MatMesh2 voronoi_beams_mesh_;
  sha::FrameSDF lattice_sdf_;
  sha::MeshSDF mesh_sdf_;
};
}  // namespace da

auto GenerateVoronoiLatticeStructure(size_t num_samples_on_longest_side, size_t num_voronoi_seeds,
                                     double radius_of_lattice_frame, double sharp_angle)
    -> da::MatMesh3 {
  using namespace da;  // NOLINT

  auto processing_mesh_path         = WorkingAssetDirectoryPath() / "model.obj";
  auto lattice_out_path             = WorkingResultDirectoryPath() / "lattice-voronoi.obj";
  auto lattice_frame_out_path       = WorkingResultDirectoryPath() / "lattice_frame.vtk";
  auto lattice_coordinates_out_path = WorkingResultDirectoryPath() / "lattice_coordinates.txt";
  auto lattice_beams_out_path       = WorkingResultDirectoryPath() / "lattice_beams.txt";
  auto lattice_seeds_out_path       = WorkingResultDirectoryPath() / "seeds.txt";
  auto mesh_sdf_out_path            = WorkingResultDirectoryPath() / "mesh.sdf";

  log::info("Processing Mesh: '{}'", processing_mesh_path.string());
  log::info("Out Lattice Mesh: '{}'", lattice_out_path.string());

  MatMesh3 matmesh = sha::ReadMatMeshFromOBJ(processing_mesh_path);

  auto mesh_aligned_box    = matmesh.AlignedBox();
  double longest_side_step = mesh_aligned_box.sizes().maxCoeff() / num_samples_on_longest_side;

  sha::SDFSampleDomain sdf_sample_domain;

  sdf_sample_domain.num_samples =
      Eigen::Vector3i(round(mesh_aligned_box.sizes().x() / longest_side_step),
                      round(mesh_aligned_box.sizes().y() / longest_side_step),
                      round(mesh_aligned_box.sizes().z() / longest_side_step));
  log::info("samples x: {}, y: {}, z: {}", sdf_sample_domain.num_samples.x(),
            sdf_sample_domain.num_samples.y(), sdf_sample_domain.num_samples.z());

  sdf_sample_domain.domain = Eigen::AlignedBox3d(
      matmesh.mat_coordinates.colwise().minCoeff().array() - 2 * radius_of_lattice_frame,
      matmesh.mat_coordinates.colwise().maxCoeff().array() + 2 * radius_of_lattice_frame);

  VoronoiLatticeProcessor processor(matmesh, sdf_sample_domain);

  log::info("Generating Voronoi Beams Mesh");
  processor.GenerateVoronoiBeamsMesh(num_voronoi_seeds, sharp_angle);

  sha::WriteMatrixToFile(lattice_seeds_out_path, processor.mat_voronoi_seeds_);

  sha::WriteMatMesh2ToVtk(lattice_frame_out_path, processor.voronoi_beams_mesh_);

  log::info("Generating Mesh Sigend Distance Field");
  processor.GenerateMeshSignedDistanceField();

  log::info("Generating Lattice Sigend Distance Field");
  processor.GenerateLatticeSignedDistanceField(radius_of_lattice_frame);

  log::info("Generating Lattice Mesh");
  auto voronoi_lattice_mesh = processor.GenerateLatticeMesh();
  log::info("Generated vertices: #{}, faces: #{}", voronoi_lattice_mesh.NumVertices(),
            voronoi_lattice_mesh.NumFaces());

  igl::writeOBJ(lattice_out_path.string(), voronoi_lattice_mesh.mat_coordinates,
                voronoi_lattice_mesh.mat_faces);
  return voronoi_lattice_mesh;
  //  sha::WriteMatrixToFile(lattice_coordinates_out_path,
  //                         processor.voronoi_beams_mesh_.mat_coordinates);
  //  sha::WriteMatrixToFile(lattice_beams_out_path, processor.voronoi_beams_mesh_.mat_beams);
}
