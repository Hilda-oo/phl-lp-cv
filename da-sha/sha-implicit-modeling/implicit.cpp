#include "implicit.h"

#include <igl/marching_cubes.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/signed_distance.h>
#include <igl/writeOBJ.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/date_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>

#include "sha-base-framework/declarations.h"
#include "sha-io-foundation/data_io.h"
#include "utility.h"

namespace da {
namespace sha {
inline fs_path GetStructureBasePath() {
  return (ProjectAssetDirectoryPath() / "MicrostructureBase");
}

// ---------------- FrameSDF --------------------
FrameSDF::FrameSDF(const MatMesh2 &frame_mesh, double radius, const SDFSampleDomain &sdf_domain)
    : DenseSDF(sdf_domain) {
  Eigen::MatrixXd query_points = GetSamplePoints();
  Eigen::VectorXi closest_beam_indices;
  Eigen::MatrixXd closest_points;
  igl::point_mesh_squared_distance(query_points, frame_mesh.mat_coordinates, frame_mesh.mat_beams,
                                   values_, closest_beam_indices, closest_points);
  values_ = -values_.array().sqrt();
  OffsetInPlace(radius);
}
// ---------------- FrameSDF --------------------

// ---------------- MeshSDF --------------------
MeshSDF::MeshSDF(const MatMesh3 &mesh, const SDFSampleDomain &sdf_domain) : DenseSDF(sdf_domain) {
  Eigen::MatrixXd query_points = GetSamplePoints();

  Eigen::VectorXi closest_triangle_indices;
  Eigen::MatrixXd closest_points;
  Eigen::MatrixXd closest_point_normals;

  igl::signed_distance(query_points, mesh.mat_coordinates, mesh.mat_faces,
                       igl::SIGNED_DISTANCE_TYPE_DEFAULT, values_, closest_triangle_indices,
                       closest_points, closest_point_normals);
  values_ = -values_;
}
// ---------------- MeshSDF --------------------

//  -------------- MicroStructureSDF --------------
MicroStructureSDF::MicroStructureSDF(const index_t structure_type,
                                     const Eigen::Vector3i &num_structures,
                                     const Eigen::AlignedBox3d &structure_domain, double radius,
                                     const SDFSampleDomain &sdf_domain)
    : DenseSDF(sdf_domain) {
  MatMesh2 microstructure_beam_mesh = LoadStructureVF(structure_type, GetStructureBasePath());
  beam_mesh_ =
      MakeMicroStructureBeamMesh(microstructure_beam_mesh, num_structures, structure_domain);
  Eigen::MatrixXd sample_points = GetSamplePoints();
  Eigen::MatrixXi closest_beam_indices;
  Eigen::MatrixXd closest_points;
  igl::point_mesh_squared_distance(sample_points, beam_mesh_.mat_coordinates, beam_mesh_.mat_beams,
                                   values_, closest_beam_indices, closest_points);
  values_ = -values_.array().sqrt();
  OffsetInPlace(radius);
}

MatMesh2 MicroStructureSDF::MakeMicroStructureBeamMesh(
    const MatMesh2 &microstructure_beam_mesh, const Eigen::Vector3i &num_structures,
    const Eigen::AlignedBox3d &structure_domain) {
  MatMesh2 beam_mesh;

  double x_min  = structure_domain.min()[0];
  double x_max  = structure_domain.max()[0];
  double y_min  = structure_domain.min()[1];
  double y_max  = structure_domain.max()[1];
  double z_min  = structure_domain.min()[2];
  double z_max  = structure_domain.max()[2];
  double x_step = (x_max - x_min) / num_structures.x();
  double y_step = (y_max - y_min) / num_structures.y();
  double z_step = (z_max - z_min) / num_structures.z();

  Eigen::Vector3d resize_factor = Eigen::Vector3d(x_step, y_step, z_step);

  Eigen::MatrixXd anchor_points = GetSamplePoints(num_structures, structure_domain);
  size_t num_samples            = anchor_points.rows();

  beam_mesh.mat_coordinates.resize(num_samples * microstructure_beam_mesh.NumVertices(), 3);
  beam_mesh.mat_beams.resize(num_samples * microstructure_beam_mesh.NumBeams(), 2);

  // reindex V
  for (index_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    for (index_t micro_vtx_idx = 0; micro_vtx_idx < microstructure_beam_mesh.NumVertices();
         ++micro_vtx_idx) {
      beam_mesh.mat_coordinates(sample_idx * microstructure_beam_mesh.NumVertices() + micro_vtx_idx,
                                0) =
          microstructure_beam_mesh.mat_coordinates(micro_vtx_idx, 0) * resize_factor(0) +
          anchor_points(sample_idx, 0);
      beam_mesh.mat_coordinates(sample_idx * microstructure_beam_mesh.NumVertices() + micro_vtx_idx,
                                1) =
          microstructure_beam_mesh.mat_coordinates(micro_vtx_idx, 1) * resize_factor(1) +
          anchor_points(sample_idx, 1);
      beam_mesh.mat_coordinates(sample_idx * microstructure_beam_mesh.NumVertices() + micro_vtx_idx,
                                2) =
          microstructure_beam_mesh.mat_coordinates(micro_vtx_idx, 2) * resize_factor(2) +
          anchor_points(sample_idx, 2);
    }
  }
  // reindex F
  for (index_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    for (index_t micro_beam_idx = 0; micro_beam_idx < microstructure_beam_mesh.NumBeams();
         ++micro_beam_idx) {
      beam_mesh.mat_beams(sample_idx * microstructure_beam_mesh.NumBeams() + micro_beam_idx, 0) =
          microstructure_beam_mesh.mat_beams(micro_beam_idx, 0) +
          sample_idx * microstructure_beam_mesh.NumVertices();
      beam_mesh.mat_beams(sample_idx * microstructure_beam_mesh.NumBeams() + micro_beam_idx, 1) =
          microstructure_beam_mesh.mat_beams(micro_beam_idx, 1) +
          sample_idx * microstructure_beam_mesh.NumVertices();
    }
  }
  return beam_mesh;
}

MatMesh3 MicroStructureSDF::MakeMicroStructurePatchMesh(const Eigen::Vector3i &num_structures,
                                                        const Eigen::AlignedBox3d &struct_domain,
                                                        index_t microstructure_type) {
  std::vector<index_t> microstructure_types(
      num_structures.x() * num_structures.y() * num_structures.z(), microstructure_type);
  return MakeMicroStructurePatchMesh(num_structures, struct_domain, microstructure_types);
}

static Eigen::Vector<double, 8> ComputeShapeBase(double x, double y, double z) {
  Eigen::Vector<double, 8> shape_base;
  shape_base(1 - 1) = -(x - 1) * (y - 1) * (z - 1);
  shape_base(2 - 1) = x * (y - 1) * (z - 1);
  shape_base(3 - 1) = -x * y * (z - 1);
  shape_base(4 - 1) = y * (x - 1) * (z - 1);
  shape_base(5 - 1) = z * (x - 1) * (y - 1);
  shape_base(6 - 1) = -x * z * (y - 1);
  shape_base(7 - 1) = x * y * z;
  shape_base(8 - 1) = -y * z * (x - 1);
  return shape_base;
}

MatMesh3 MicroStructureSDF::MakeMicroStructurePatchMesh(const Eigen::Vector3i &num_structures,
                                                        const Eigen::AlignedBox3d &structure_domain,
                                                        const std::vector<index_t> &types) {
  MatMesh3 patch_mesh;

  Assert(types.size() == num_structures.x() * num_structures.y() * num_structures.z() ||
         "types size must be equal to structNum");
  auto map_type_idx_to_patch_mesh = ReadTrianglePatchFromMicrostructureBase(
      types, (ProjectAssetDirectoryPath() / "MicrostructureBase").string());

  int num_voxels   = num_structures.x() * num_structures.y() * num_structures.z();
  int num_x_voxels = num_structures.x();
  int num_y_voxels = num_structures.y();
  int num_z_voxels = num_structures.z();

  const Eigen::Vector3d voxel_step =
      structure_domain.sizes().array() / num_structures.cast<double>().array();

  for (index_t voxel_k_idx = 0; voxel_k_idx < num_z_voxels; ++voxel_k_idx) {
    for (index_t voxel_j_idx = 0; voxel_j_idx < num_y_voxels; ++voxel_j_idx) {
      for (index_t voxel_i_idx = 0; voxel_i_idx < num_x_voxels; ++voxel_i_idx) {
        index_t flatten_idx         = (voxel_i_idx + voxel_j_idx * num_structures(0) +
                               voxel_k_idx * num_structures(0) * num_structures(1));
        index_t type_name           = types[flatten_idx];
        auto microstructure_matmesh = map_type_idx_to_patch_mesh.at(type_name);
        Eigen::Vector3d voxel_position =
            structure_domain.min().array() +
            Eigen::Vector3d(voxel_i_idx, voxel_j_idx, voxel_k_idx).array() * voxel_step.array();
        for (int i = 0; i < microstructure_matmesh.NumVertices(); ++i) {
          microstructure_matmesh.mat_coordinates.row(i).array() *= voxel_step.array();
          microstructure_matmesh.mat_coordinates.row(i).array() += voxel_position.array();
        }
        microstructure_matmesh.mat_faces.array() += patch_mesh.NumVertices();

        if (patch_mesh.NumVertices() == 0) {
          patch_mesh = microstructure_matmesh;
        } else {
          MatMesh3 new_matmesh;
          size_t num_vertices_of_new_mesh =
              patch_mesh.mat_coordinates.rows() + microstructure_matmesh.mat_coordinates.rows();
          size_t num_faces_of_new_mesh =
              patch_mesh.mat_faces.rows() + microstructure_matmesh.mat_faces.rows();
          new_matmesh.mat_coordinates.resize(num_vertices_of_new_mesh, 3);
          new_matmesh.mat_coordinates.block(0, 0, patch_mesh.mat_coordinates.rows(), 3) =
              patch_mesh.mat_coordinates;
          new_matmesh.mat_coordinates.block(patch_mesh.mat_coordinates.rows(), 0,
                                            microstructure_matmesh.mat_coordinates.rows(), 3) =
              microstructure_matmesh.mat_coordinates;
          new_matmesh.mat_faces.resize(num_faces_of_new_mesh, 3);
          new_matmesh.mat_faces.block(0, 0, patch_mesh.mat_faces.rows(), 3) = patch_mesh.mat_faces;
          new_matmesh.mat_faces.block(patch_mesh.mat_faces.rows(), 0,
                                      microstructure_matmesh.mat_faces.rows(), 3) =
              microstructure_matmesh.mat_faces;
          patch_mesh = new_matmesh;
        }
      }
    }
  }
  return patch_mesh;
}
//  -------------- MicroStructureSDF --------------

// ------------ NonUniformMicroStructureSDF ----------------
NonUniformMicroStructureSDF::NonUniformMicroStructureSDF
    (const index_t structure_type, const Eigen::Vector3i &num_structures,
     const Eigen::AlignedBox3d &structure_domain, double radius,
     const Expression_1 &field_expression, double field_coeff,
     const SDFSampleDomain &sdf_domain)
    : DenseSDF(sdf_domain), exp_field_(field_expression) {
  MatMesh2 microstructure_beam_mesh = LoadStructureVF(structure_type, GetStructureBasePath());
  beam_mesh_ =
      MakeMicroStructureBeamMesh(microstructure_beam_mesh, num_structures, structure_domain);
  Eigen::MatrixXd sample_points = GetSamplePoints();
  Eigen::MatrixXi closest_beam_indices;
  Eigen::MatrixXd closest_points;
  igl::point_mesh_squared_distance(sample_points, beam_mesh_.mat_coordinates, beam_mesh_.mat_beams,
                                   values_, closest_beam_indices, closest_points);
  values_ = -values_.array().sqrt();

  // calculate radius of each sample point
  size_t num_samples            = sample_points.rows();
  Eigen::MatrixXd radiuses(num_samples, 1);
  for (index_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    const auto &point          = sample_points.row(sample_idx);
    double field_value         = exp_field_(point.x(), point.y(), point.z());
    radiuses(sample_idx, 0) = radius + field_value * field_coeff;
    radiuses(sample_idx, 0) = abs(radiuses(sample_idx, 0));
  }
  values_ += radiuses;
}

NonUniformMicroStructureSDF::NonUniformMicroStructureSDF
    (const index_t structure_type, const Eigen::Vector3i &num_structures,
     const Eigen::AlignedBox3d &structure_domain, double radius,
     const ScalarField &field_matrix, double field_coeff,
     const SDFSampleDomain &sdf_domain)
    : DenseSDF(sdf_domain), matrix_field_(field_matrix) {
  MatMesh2 microstructure_beam_mesh = LoadStructureVF(structure_type, GetStructureBasePath());
  beam_mesh_ =
      MakeMicroStructureBeamMesh(microstructure_beam_mesh, num_structures, structure_domain);
  Eigen::MatrixXd sample_points = GetSamplePoints();
  Eigen::MatrixXi closest_beam_indices;
  Eigen::MatrixXd closest_points;
  igl::point_mesh_squared_distance(sample_points, beam_mesh_.mat_coordinates, beam_mesh_.mat_beams,
                                   values_, closest_beam_indices, closest_points);
  values_ = -values_.array().sqrt();

  // calculate radius of each sample point
  size_t num_samples            = sample_points.rows();
  Eigen::MatrixXd radiuses(num_samples, 1);
  for (index_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    const auto &point          = sample_points.row(sample_idx);
    double field_value         = matrix_field_.Sample(point.x(), point.y(), point.z());
    radiuses(sample_idx, 0) = radius + field_value * field_coeff;
    radiuses(sample_idx, 0) = abs(radiuses(sample_idx, 0));
  }
  values_ += radiuses;
}

MatMesh2 NonUniformMicroStructureSDF::MakeMicroStructureBeamMesh(
    const MatMesh2 &microstructure_beam_mesh, const Eigen::Vector3i &num_structures,
    const Eigen::AlignedBox3d &structure_domain) {
  MatMesh2 beam_mesh;

  double x_min  = structure_domain.min()[0];
  double x_max  = structure_domain.max()[0];
  double y_min  = structure_domain.min()[1];
  double y_max  = structure_domain.max()[1];
  double z_min  = structure_domain.min()[2];
  double z_max  = structure_domain.max()[2];
  double x_step = (x_max - x_min) / num_structures.x();
  double y_step = (y_max - y_min) / num_structures.y();
  double z_step = (z_max - z_min) / num_structures.z();

  Eigen::Vector3d resize_factor = Eigen::Vector3d(x_step, y_step, z_step);

  Eigen::MatrixXd anchor_points = GetSamplePoints(num_structures, structure_domain);
  size_t num_samples            = anchor_points.rows();

  beam_mesh.mat_coordinates.resize(num_samples * microstructure_beam_mesh.NumVertices(), 3);
  beam_mesh.mat_beams.resize(num_samples * microstructure_beam_mesh.NumBeams(), 2);

  // reindex V
  for (index_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    for (index_t micro_vtx_idx = 0; micro_vtx_idx < microstructure_beam_mesh.NumVertices();
         ++micro_vtx_idx) {
      beam_mesh.mat_coordinates(sample_idx * microstructure_beam_mesh.NumVertices() + micro_vtx_idx,
                                0) =
          microstructure_beam_mesh.mat_coordinates(micro_vtx_idx, 0) * resize_factor(0) +
          anchor_points(sample_idx, 0);
      beam_mesh.mat_coordinates(sample_idx * microstructure_beam_mesh.NumVertices() + micro_vtx_idx,
                                1) =
          microstructure_beam_mesh.mat_coordinates(micro_vtx_idx, 1) * resize_factor(1) +
          anchor_points(sample_idx, 1);
      beam_mesh.mat_coordinates(sample_idx * microstructure_beam_mesh.NumVertices() + micro_vtx_idx,
                                2) =
          microstructure_beam_mesh.mat_coordinates(micro_vtx_idx, 2) * resize_factor(2) +
          anchor_points(sample_idx, 2);
    }
  }
  // reindex F
  for (index_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    for (index_t micro_beam_idx = 0; micro_beam_idx < microstructure_beam_mesh.NumBeams();
         ++micro_beam_idx) {
      beam_mesh.mat_beams(sample_idx * microstructure_beam_mesh.NumBeams() + micro_beam_idx, 0) =
          microstructure_beam_mesh.mat_beams(micro_beam_idx, 0) +
          sample_idx * microstructure_beam_mesh.NumVertices();
      beam_mesh.mat_beams(sample_idx * microstructure_beam_mesh.NumBeams() + micro_beam_idx, 1) =
          microstructure_beam_mesh.mat_beams(micro_beam_idx, 1) +
          sample_idx * microstructure_beam_mesh.NumVertices();
    }
  }
  return beam_mesh;
}

MatMesh3 NonUniformMicroStructureSDF::MakeMicroStructurePatchMesh(
  const Eigen::Vector3i &num_structures,
  const Eigen::AlignedBox3d &struct_domain, index_t microstructure_type) {
  std::vector<index_t> microstructure_types(
      num_structures.x() * num_structures.y() * num_structures.z(), microstructure_type);
  return MakeMicroStructurePatchMesh(num_structures, struct_domain, microstructure_types);
}

MatMesh3 NonUniformMicroStructureSDF::MakeMicroStructurePatchMesh(
  const Eigen::Vector3i &num_structures,
  const Eigen::AlignedBox3d &structure_domain,
  const std::vector<index_t> &types) {
  MatMesh3 patch_mesh;
  Assert(types.size() == num_structures.x() * num_structures.y() * num_structures.z() ||
         "types size must be equal to structNum");
  auto map_type_idx_to_patch_mesh = ReadTrianglePatchFromMicrostructureBase(
      types, (ProjectAssetDirectoryPath() / "MicrostructureBase").string());

  int num_voxels   = num_structures.x() * num_structures.y() * num_structures.z();
  int num_x_voxels = num_structures.x();
  int num_y_voxels = num_structures.y();
  int num_z_voxels = num_structures.z();

  const Eigen::Vector3d voxel_step =
      structure_domain.sizes().array() / num_structures.cast<double>().array();

  for (index_t voxel_k_idx = 0; voxel_k_idx < num_z_voxels; ++voxel_k_idx) {
    for (index_t voxel_j_idx = 0; voxel_j_idx < num_y_voxels; ++voxel_j_idx) {
      for (index_t voxel_i_idx = 0; voxel_i_idx < num_x_voxels; ++voxel_i_idx) {
        index_t flatten_idx         = (voxel_i_idx + voxel_j_idx * num_structures(0) +
                               voxel_k_idx * num_structures(0) * num_structures(1));
        index_t type_name           = types[flatten_idx];
        auto microstructure_matmesh = map_type_idx_to_patch_mesh.at(type_name);
        Eigen::Vector3d voxel_position =
            structure_domain.min().array() +
            Eigen::Vector3d(voxel_i_idx, voxel_j_idx, voxel_k_idx).array() * voxel_step.array();
        for (int i = 0; i < microstructure_matmesh.NumVertices(); ++i) {
          microstructure_matmesh.mat_coordinates.row(i).array() *= voxel_step.array();
          microstructure_matmesh.mat_coordinates.row(i).array() += voxel_position.array();
        }
        microstructure_matmesh.mat_faces.array() += patch_mesh.NumVertices();

        if (patch_mesh.NumVertices() == 0) {
          patch_mesh = microstructure_matmesh;
        } else {
          MatMesh3 new_matmesh;
          size_t num_vertices_of_new_mesh =
              patch_mesh.mat_coordinates.rows() + microstructure_matmesh.mat_coordinates.rows();
          size_t num_faces_of_new_mesh =
              patch_mesh.mat_faces.rows() + microstructure_matmesh.mat_faces.rows();
          new_matmesh.mat_coordinates.resize(num_vertices_of_new_mesh, 3);
          new_matmesh.mat_coordinates.block(0, 0, patch_mesh.mat_coordinates.rows(), 3) =
              patch_mesh.mat_coordinates;
          new_matmesh.mat_coordinates.block(patch_mesh.mat_coordinates.rows(), 0,
                                            microstructure_matmesh.mat_coordinates.rows(), 3) =
              microstructure_matmesh.mat_coordinates;
          new_matmesh.mat_faces.resize(num_faces_of_new_mesh, 3);
          new_matmesh.mat_faces.block(0, 0, patch_mesh.mat_faces.rows(), 3) = patch_mesh.mat_faces;
          new_matmesh.mat_faces.block(patch_mesh.mat_faces.rows(), 0,
                                      microstructure_matmesh.mat_faces.rows(), 3) =
              microstructure_matmesh.mat_faces;
          patch_mesh = new_matmesh;
        }
      }
    }
  }
  return patch_mesh;
}
// ------------ NonUniformMicroStructureSDF ----------------

// ------------ NonUniformFrameSDF ----------------
NonUniformFrameSDF::NonUniformFrameSDF(const MatMesh2 &frame_mesh, const Eigen::MatrixXd &radiuses,
                                       const SDFSampleDomain &sdf_domain)
    : DenseSDF(sdf_domain) {
  Assert(frame_mesh.NumBeams() == radiuses.rows(),
         "Beams and radius must have the same number of rows");
  Eigen::MatrixXd sample_points = GetSamplePoints();
  size_t num_sample_points =
      sdf_domain.num_samples.x() * sdf_domain.num_samples.y() * sdf_domain.num_samples.z();

  values_ = Eigen::MatrixXd::Ones(num_sample_points, 1) * 1e6;

  auto PointInAlignedBox = [](const double *aabb, const double *point) -> bool {
    return point[0] >= aabb[0] and point[0] <= aabb[1] and point[1] >= aabb[2] and
           point[1] <= aabb[3] and point[2] >= aabb[4] and point[2] <= aabb[5];
  };

  auto ComputeDistanceBetweenPointAndBeam = [](const Eigen::Vector3d &point,
                                               const Eigen::Vector3d &beam_point_a,
                                               Eigen::Vector3d &beam_point_b) -> double {
    Eigen::Vector3d vector_ab = beam_point_b - beam_point_a;
    Eigen::Vector3d vector_av = point - beam_point_a;
    if (vector_av.dot(vector_ab) <= 0.0)  // Point is lagging behind start of the segment, so
                                          // perpendicular distance is not viable.
      return vector_av.norm();            // Use distance to start of segment instead.
    Eigen::Vector3d vector_bv = point - beam_point_b;
    if (vector_bv.dot(vector_ab) >= 0.0)  // Point is advanced past the end of the segment, so
                                          // perpendicular distance is not viable.
      return vector_bv.norm();            // Use distance to end of the segment instead.
    return (vector_ab.cross(vector_av)).norm() / vector_ab.norm();
  };

  boost::progress_display progress(frame_mesh.NumBeams());

  for (index_t beam_idx = 0; beam_idx < frame_mesh.NumBeams(); ++beam_idx) {
    double radius                 = radiuses(beam_idx, 0);
    double expand                 = 2 * radius;
    Eigen::Vector2i beam_vertices = frame_mesh.mat_beams.row(beam_idx);
    Eigen::Vector3d point_0       = frame_mesh.mat_coordinates.row(beam_vertices.x());
    Eigen::Vector3d point_1       = frame_mesh.mat_coordinates.row(beam_vertices.y());

    double aabb[] = {
        std::min(point_0.x(), point_1.x()) - expand, std::max(point_0.x(), point_1.x()) + expand,
        std::min(point_0.y(), point_1.y()) - expand, std::max(point_0.y(), point_1.y()) + expand,
        std::min(point_0.z(), point_1.z()) - expand, std::max(point_0.z(), point_1.z()) + expand};
#pragma omp parallel for
    for (index_t sample_idx = 0; sample_idx < num_sample_points; ++sample_idx) {
      Eigen::Vector3d point = sample_points.row(sample_idx);
      if (PointInAlignedBox(aabb, point.data())) {
        double distance = ComputeDistanceBetweenPointAndBeam(point, point_0, point_1);
#pragma omp critical(sample_idx)
        values_.data()[sample_idx] = std::min(values_.data()[sample_idx], distance - radius);
      }
    }
    ++progress;
  }
}

HexahedralMeshSDF::HexahedralMeshSDF(const index_t structure_type,
                                     const HexahedralMatMesh &hexahedral_matmesh, double radius,
                                     const SDFSampleDomain &sdf_domain)
    : DenseSDF(sdf_domain) {
  MatMesh2 microstructure_beam_mesh = LoadStructureVF(structure_type, GetStructureBasePath());
  beam_matmesh_ = GetMicroStructureBeamMesh(hexahedral_matmesh, microstructure_beam_mesh);
  Eigen::MatrixXd sample_points = GetSamplePoints();
  Eigen::MatrixXi closest_beam_indices;
  Eigen::MatrixXd closest_points;
  igl::point_mesh_squared_distance(sample_points, beam_matmesh_.mat_coordinates,
                                   beam_matmesh_.mat_beams, values_, closest_beam_indices,
                                   closest_points);
  values_ = -values_.array().sqrt();
  OffsetInPlace(radius);
}
MatMesh2 HexahedralMeshSDF::GetMicroStructureBeamMesh(const HexahedralMatMesh &hexahedral_matmesh,
                                                      const MatMesh2 &microstructure_beam_mesh) {
  MatMesh2 beam_matmesh;
  beam_matmesh.mat_coordinates.resize(
      hexahedral_matmesh.NumHexahedrons() * microstructure_beam_mesh.NumVertices(), 3);
  beam_matmesh.mat_beams.resize(
      hexahedral_matmesh.NumHexahedrons() * microstructure_beam_mesh.NumBeams(), 2);

  Eigen::MatrixXd mat_shape_base;
  mat_shape_base.resize(microstructure_beam_mesh.NumVertices(), 8);
  for (index_t vertex_idx = 0; vertex_idx < microstructure_beam_mesh.NumVertices(); ++vertex_idx) {
    const auto &point              = microstructure_beam_mesh.mat_coordinates.row(vertex_idx);
    mat_shape_base.row(vertex_idx) = ComputeShapeBase(point.x(), point.y(), point.z()).transpose();
  }

  for (index_t cell_idx = 0; cell_idx < hexahedral_matmesh.NumHexahedrons(); ++cell_idx) {
    index_t vertices_base = cell_idx * microstructure_beam_mesh.NumVertices();
    index_t beams_base    = cell_idx * microstructure_beam_mesh.NumBeams();
    Eigen::MatrixXd mat_hexahedron_cell_coords(8, 3);
    for (index_t idx = 0; idx < 8; ++idx) {
      index_t vertex_idx                  = hexahedral_matmesh.mat_hexahedrons(cell_idx, idx);
      mat_hexahedron_cell_coords.row(idx) = hexahedral_matmesh.mat_coordinates.row(vertex_idx);
    }
    Eigen::MatrixXd mat_cell_coords       = mat_shape_base * mat_hexahedron_cell_coords;
    beam_matmesh.mat_coordinates.block(vertices_base, 0, microstructure_beam_mesh.NumVertices(),
                                       3) = mat_cell_coords;
    beam_matmesh.mat_beams.block(beams_base, 0, microstructure_beam_mesh.NumBeams(), 2) =
        microstructure_beam_mesh.mat_beams.array() + vertices_base;
  }
  return beam_matmesh;
}

DenseExpressionSDF::DenseExpressionSDF(const ExpressionFunction &tpms_expression,
                                       const SDFSampleDomain &sdf_domain)
    : DenseSDF(sdf_domain), exp_tpms_(tpms_expression) {
  Eigen::MatrixXd sample_points  = GetSamplePoints();
  const size_t num_sample_points = sample_points.rows();
  values_.resize(num_sample_points, 1);
#pragma omp parallel for
  for (index_t sample_idx = 0; sample_idx < num_sample_points; ++sample_idx) {
    const auto &point          = sample_points.row(sample_idx);
    double value               = exp_tpms_(point.x(), point.y(), point.z());
    values_.data()[sample_idx] = value;
  }
  values_ = -values_;
}

FieldDrivenDenseExpressionSDF::FieldDrivenDenseExpressionSDF(
                                       const Expression_2 &tpms_expression,
                                       double tpms_coeff,
                                       const Expression_1 &field_expression,
                                       double field_coeff,
                                       const SDFSampleDomain &sdf_domain)
    : DenseSDF(sdf_domain), exp_tpms_(tpms_expression), exp_field_(field_expression) {
  Eigen::MatrixXd sample_points  = GetSamplePoints();
  const size_t num_sample_points = sample_points.rows();
  values_.resize(num_sample_points, 1);
#pragma omp parallel for
  for (index_t sample_idx = 0; sample_idx < num_sample_points; ++sample_idx) {
    const auto &point          = sample_points.row(sample_idx);
    double field_value         = exp_field_(point.x() / 50, point.y() / 50, point.z() / 50);
    double field2tpms_coeff    = tpms_coeff;
    double value               = exp_tpms_(point.x(), point.y(), point.z(), field2tpms_coeff)
                                          + field_coeff * field_value;
    values_.data()[sample_idx] = value;
  }
  values_ = -values_;
}

FieldDrivenDenseExpressionSDF::FieldDrivenDenseExpressionSDF(
                                       const Expression_2 &tpms_expression,
                                       double tpms_coeff,
                                       const ScalarField &field_matrix,
                                       double field_coeff,
                                       const SDFSampleDomain &sdf_domain)
    : DenseSDF(sdf_domain), exp_tpms_(tpms_expression), matrix_field_(field_matrix) {
  Eigen::MatrixXd sample_points  = GetSamplePoints();
  const size_t num_sample_points = sample_points.rows();
  values_.resize(num_sample_points, 1);
  double range = matrix_field_.values_.maxCoeff() - matrix_field_.values_.minCoeff();
  double field_matrix_min = matrix_field_.values_.minCoeff();
#pragma omp parallel for
  for (index_t sample_idx = 0; sample_idx < num_sample_points; ++sample_idx) {
    const auto &point          = sample_points.row(sample_idx);
    double field_value         = matrix_field_.Sample(point.x(), point.y(), point.z());
    double homo_field_value    = (field_value - field_matrix_min) / range;
    double field2tpms_coeff    = tpms_coeff;
    double value               = exp_tpms_(point.x(), point.y(), point.z(), field2tpms_coeff)
                                          + field_coeff * field_value;
    values_.data()[sample_idx] = value;
  }
  values_ = -values_;
}

}  // namespace sha
}  // namespace da