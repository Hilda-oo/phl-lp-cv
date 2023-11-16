#include <igl/adjacency_list.h>
#include <igl/boundary_facets.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/edges.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/volume.h>
#include <igl/writeOBJ.h>

#include <algorithm>

#include <boost/range/adaptor/indexed.hpp>

#include "utility.h"

#include "sha-base-framework/declarations.h"
#include "sha-base-framework/frame.h"
#include "sha-hexahedron-generation/polycube_based.h"
#include "sha-implicit-modeling/implicit.h"
#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-volume-mesh/matmesh.h"

namespace da {
class HexahedronStructureLatticeDesignProcessor {
 public:
  explicit HexahedronStructureLatticeDesignProcessor(const MatMesh3 &matmesh,
                                                     const sha::SDFSampleDomain sdf_sample_domain)
      : matmesh_(matmesh), sdf_sample_domain_(sdf_sample_domain) {}

  void GenerateTetrahedronByTetGen(const std::string &tetgen_switches) {
    log::info("Generate tetrahedral mesh by TetGen");
    Eigen::MatrixXi mat_surface_faces;
    igl::copyleft::tetgen::tetrahedralize(matmesh_.mat_coordinates, matmesh_.mat_faces,
                                          tetgen_switches, tetrahedral_matmesh_.mat_coordinates,
                                          tetrahedral_matmesh_.mat_tetrahedrons, mat_surface_faces);
  }

  void GenerateTetrahedralPolycube(double alpha, double beta, size_t num_iterations) {
    log::info("Generate tetrahedral polycube");
    mat_tetrahedral_polycube_coordinates_ =
        sha::GeneratePolycubeForTetrahedralMesh(tetrahedral_matmesh_, alpha, beta, num_iterations);
  }

  void GenerateHexahedralMesh(const Eigen::Vector3i &num_cells) {
    log::info("Generate hexahedral mesh");
    hexahedral_matmesh_ = sha::GenerateHexahedralMeshByPolycube(
        tetrahedral_matmesh_, mat_tetrahedral_polycube_coordinates_, num_cells,
        mat_hexahedral_polycube_coordinates_);
  }

  void GenerateSigendDistanceFieldForBeams(index_t structure_type, double radius) {
    log::info("Generate sigend distance field for beams");
    sha::HexahedralMeshSDF hexahedral_meshsdf(structure_type, hexahedral_matmesh_, radius,
                                              sdf_sample_domain_);
    lattice_skeleton_matmesh_ = hexahedral_meshsdf.beam_matmesh_;
    lattice_matmesh_          = hexahedral_meshsdf.GenerateMeshByMarchingCubes(0);
  }

 private:
  const MatMesh3 matmesh_;
  const sha::SDFSampleDomain sdf_sample_domain_;

 public:
  TetrahedralMatMesh tetrahedral_matmesh_;
  Eigen::MatrixXd mat_tetrahedral_polycube_coordinates_;
  Eigen::MatrixXd mat_hexahedral_polycube_coordinates_;
  HexahedralMatMesh hexahedral_matmesh_;
  MatMesh2 lattice_skeleton_matmesh_;
  MatMesh3 lattice_matmesh_;
};
}  // namespace da

auto GenerateHexahedronLatticeStructure(size_t num_samples_on_longest_side, double alpha,
                                        double beta, size_t num_iterations,
                                        Eigen::Vector3i num_cells, double radius,
                                        size_t structure_type, const std::string &tetgen_switches)
    -> da::MatMesh3 {
  using namespace da;  // NOLINT
  auto processing_mesh_path        = WorkingAssetDirectoryPath() / "model.obj";
  auto lattice_out_path            = WorkingResultDirectoryPath() / "lattice-hexahedron.obj";
  auto quadrilateral_mesh_out_path = WorkingResultDirectoryPath() / "quadrangle.obj";
  auto tetrahedral_polycube_mesh_out_path =
      WorkingResultDirectoryPath() / "tetrahedral_polycube.vtk";
  auto hexahedral_polycube_mesh_out_path = WorkingResultDirectoryPath() / "hexahedral_polycube.vtk";
  auto hexahedral_mesh_out_path          = WorkingResultDirectoryPath() / "hexahedron.vtk";
  auto lattice_coordinates_out_path      = WorkingResultDirectoryPath() / "lattice_coordinates.txt";
  auto lattice_beams_out_path            = WorkingResultDirectoryPath() / "lattice_beams.txt";
  auto mesh_sdf_out_path                 = WorkingResultDirectoryPath() / "mesh.sdf";

  MatMesh3 matmesh = sha::ReadMatMeshFromOBJ(processing_mesh_path);
  log::info("load mesh: V: {}, F: {}", matmesh.NumVertices(), matmesh.NumFaces());

  auto mesh_aligned_box    = matmesh.AlignedBox();
  double longest_side_step = mesh_aligned_box.sizes().maxCoeff() / num_samples_on_longest_side;

  Eigen::Vector3i num_xyz_samples(round(mesh_aligned_box.sizes().x() / longest_side_step),
                                  round(mesh_aligned_box.sizes().y() / longest_side_step),
                                  round(mesh_aligned_box.sizes().z() / longest_side_step));
  log::info("samples x: {}, y: {}, z: {}", num_xyz_samples.x(), num_xyz_samples.y(),
            num_xyz_samples.z());
  Eigen::AlignedBox3d sdf_domain(matmesh.mat_coordinates.colwise().minCoeff().array() - 2 * radius,
                                 matmesh.mat_coordinates.colwise().maxCoeff().array() + 2 * radius);

  sha::SDFSampleDomain sdf_sample_domain{.num_samples = num_xyz_samples, .domain = sdf_domain};

  HexahedronStructureLatticeDesignProcessor processor(matmesh, sdf_sample_domain);
  processor.GenerateTetrahedronByTetGen(tetgen_switches);
  processor.GenerateTetrahedralPolycube(alpha, beta, num_iterations);

  sha::WriteTetrahedralMatmeshToVtk(
      tetrahedral_polycube_mesh_out_path.string(),
      TetrahedralMatMesh{.mat_coordinates  = processor.mat_tetrahedral_polycube_coordinates_,
                         .mat_tetrahedrons = processor.tetrahedral_matmesh_.mat_tetrahedrons});
  processor.GenerateHexahedralMesh(num_cells);
  processor.GenerateSigendDistanceFieldForBeams(structure_type, radius);

  HexahedralMesh hexahedral_mesh =
      sha::CreateHexahedralMeshFromMatMesh(processor.hexahedral_matmesh_);
  auto quadrilateral_mesh    = sha::CreateSurfaceMesh3FromHexahedralMesh(hexahedral_mesh);
  auto quadrilateral_matmesh = sha::CreateMatMesh3FromSurfaceMesh3(quadrilateral_mesh, 4);
  igl::writeOBJ(quadrilateral_mesh_out_path.string(), quadrilateral_matmesh.mat_coordinates,
                quadrilateral_matmesh.mat_faces);

  igl::writeOBJ(lattice_out_path.string(), processor.lattice_matmesh_.mat_coordinates,
                processor.lattice_matmesh_.mat_faces);
  sha::WriteMatrixToFile(lattice_coordinates_out_path,
                         processor.lattice_skeleton_matmesh_.mat_coordinates);
  sha::WriteMatrixToFile(lattice_beams_out_path, processor.lattice_skeleton_matmesh_.mat_beams);

  sha::WriteHexahedralMatmeshToVtk(
      hexahedral_polycube_mesh_out_path.string(),
      HexahedralMatMesh{.mat_coordinates = processor.mat_hexahedral_polycube_coordinates_,
                        .mat_hexahedrons = processor.hexahedral_matmesh_.mat_hexahedrons});

  sha::WriteHexahedralMatmeshToVtk(hexahedral_mesh_out_path.string(),
                                   processor.hexahedral_matmesh_);
  return processor.lattice_matmesh_;
}
