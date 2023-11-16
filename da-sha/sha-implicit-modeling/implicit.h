#pragma once

#include <Eigen/Eigen>
#include "sha-base-framework/frame.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-volume-mesh/matmesh.h"

namespace da {
namespace sha {
using DataType = Eigen::VectorXd;

struct SDFSampleDomain {
  Eigen::AlignedBox3d domain;
  Eigen::Vector3i num_samples = Eigen::Vector3i::Ones();

 public:
  bool operator==(const SDFSampleDomain &other) const {
    constexpr double kEps = 1e-6;
    return ((domain.min() - other.domain.min()).cwiseAbs().maxCoeff() < kEps &&
            (domain.max() - other.domain.max()).cwiseAbs().maxCoeff() < kEps) &&
           (num_samples == other.num_samples);
  }
};

class ScalarField {
 public:
  DataType values_;
  SDFSampleDomain sdf_sample_domain_;

 public:
  explicit ScalarField(const DataType &values, const SDFSampleDomain &sdf_domain);

  explicit ScalarField(const SDFSampleDomain &sdf_domain);

  explicit ScalarField() = default;

 public:
  index_t ComputeArrayIndex(index_t idx_i, index_t idx_j, index_t idx_k) const;

  double StandardSample(double x, double y, double z) const;

  double Sample(double x, double y, double z) const;

 public:
  void DumpTo(const std::string &path);

  void LoadFrom(const std::string &path);
};

class DenseSDF;

class DenseSDF : public ScalarField {
 public:
  explicit DenseSDF(const DataType &values, const SDFSampleDomain &sdf_domain);

  explicit DenseSDF(const SDFSampleDomain &sdf_domain);

  explicit DenseSDF() = default;

  double &operator()(index_t idx_i, index_t idx_j, index_t idx_k);

  const double &operator()(index_t idx_i, index_t idx_j, index_t idx_k) const;

  double &At(index_t idx_i, index_t idx_j, index_t idx_k);

  const double &At(index_t idx_i, index_t idx_j, index_t idx_k) const;

 public:
  DenseSDF Union(const DenseSDF &other) const;

  DenseSDF Intersect(const DenseSDF &other) const;

  DenseSDF Complement() const;

  DenseSDF Difference(const DenseSDF &other) const;

 public:
  DenseSDF Offset(double offset);

  void OffsetInPlace(double offset);

 public:
  enum PaddingType { PADDING_INF, PADDING_NEGINF, PADDING_CONSTANT, PADDING_ZERO, PADDING_SAME };

  DenseSDF Move(const Eigen::Vector3i &offset, PaddingType padding_type = PADDING_INF,
                double padding = 0) const;

  void MoveInPlace(const Eigen::Vector3i &offset, PaddingType padding_type = PADDING_INF,
                   double padding = 0);

  DenseSDF MoveInPart(const Eigen::AlignedBox<int, 3> &range, const Eigen::Vector3i &offset,
                      PaddingType padding_type = PADDING_INF, double padding = 0) const;

  void MoveInPartInPlace(const Eigen::AlignedBox<int, 3> &range, const Eigen::Vector3i &offset,
                         PaddingType padding_type = PADDING_INF, double padding = 0);

 public:
  MatMesh3 GenerateMeshByMarchingCubes(double isovalue) const;

  void ExportObj(const std::string &filename) const;

 public:
  Eigen::MatrixXd GetSamplePoints() const;

  static Eigen::MatrixXd GetSamplePoints(const Eigen::Vector3i &num_cells,
                                         const Eigen::AlignedBox3d &domain);
};

class FrameSDF : public DenseSDF {
 public:
  explicit FrameSDF(const MatMesh2 &frame_mesh, double radius, const SDFSampleDomain &sdf_domain);

  explicit FrameSDF() = default;
};

class NonUniformFrameSDF : public DenseSDF {
 public:
  explicit NonUniformFrameSDF(const MatMesh2 &frame_mesh, const Eigen::MatrixXd &radius,
                              const SDFSampleDomain &sdf_domain);
};

class MeshSDF : public DenseSDF {
 public:
  explicit MeshSDF(const MatMesh3 &mesh, const SDFSampleDomain &sdf_domain);

  explicit MeshSDF() = default;
};

class MicroStructureSDF : public DenseSDF {
 public:
  explicit MicroStructureSDF(const index_t structure_type, const Eigen::Vector3i &num_structures,
                             const Eigen::AlignedBox3d &structure_domain, double radius,
                             const SDFSampleDomain &sdf_domain);

  explicit MicroStructureSDF() = default;

 private:
  static MatMesh2 MakeMicroStructureBeamMesh(const MatMesh2 &microstructure_beam_mesh,
                                             const Eigen::Vector3i &num_structures,
                                             const Eigen::AlignedBox3d &structure_domain);

 public:
  static MatMesh3 MakeMicroStructurePatchMesh(const Eigen::Vector3i &num_structures,
                                              const Eigen::AlignedBox3d &structure_domain,
                                              index_t structure_type);
  static MatMesh3 MakeMicroStructurePatchMesh(const Eigen::Vector3i &num_structures,
                                              const Eigen::AlignedBox3d &structure_domain,
                                              const std::vector<index_t> &structure_types);

 public:
  MatMesh2 beam_mesh_;
};

class NonUniformMicroStructureSDF : public DenseSDF {
  // zcp
 public:
  using Expression_1 = std::function<double(double x, double y, double z)>;
 public:
  explicit NonUniformMicroStructureSDF(const index_t structure_type, const Eigen::Vector3i &num_structures,
                                       const Eigen::AlignedBox3d &structure_domain, double radius,
                                       const Expression_1 &field_expression, double field_coeff,
                                       const SDFSampleDomain &sdf_domain);

  explicit NonUniformMicroStructureSDF(const index_t structure_type, const Eigen::Vector3i &num_structures,
                                       const Eigen::AlignedBox3d &structure_domain, double radius,
                                       const ScalarField &field_matrix, double field_coeff,
                                       const SDFSampleDomain &sdf_domain);
  // zcp
  explicit NonUniformMicroStructureSDF() = default;

 private:
  static MatMesh2 MakeMicroStructureBeamMesh(const MatMesh2 &microstructure_beam_mesh,
                                             const Eigen::Vector3i &num_structures,
                                             const Eigen::AlignedBox3d &structure_domain);

 public:
  static MatMesh3 MakeMicroStructurePatchMesh(const Eigen::Vector3i &num_structures,
                                              const Eigen::AlignedBox3d &structure_domain,
                                              index_t structure_type);
  static MatMesh3 MakeMicroStructurePatchMesh(const Eigen::Vector3i &num_structures,
                                              const Eigen::AlignedBox3d &structure_domain,
                                              const std::vector<index_t> &structure_types);

 public:
  MatMesh2 beam_mesh_;
  // zcp
  Expression_1 exp_field_;
  ScalarField matrix_field_;
};

class HexahedralMeshSDF : public DenseSDF {
 public:
  explicit HexahedralMeshSDF(const index_t structure_type,
                             const HexahedralMatMesh &hexahedral_matmesh, double radius,
                             const SDFSampleDomain &sdf_domain);

  explicit HexahedralMeshSDF() = default;

 private:
  static MatMesh2 GetMicroStructureBeamMesh(const HexahedralMatMesh &hexahedral_matmesh,
                                            const MatMesh2 &microstructure_matmesh);

 public:
  MatMesh2 beam_matmesh_;
};

class DenseExpressionSDF : public DenseSDF {
 public:
  using ExpressionFunction = std::function<double(double x, double y, double z)>;

 public:
  explicit DenseExpressionSDF(const ExpressionFunction &tpms_expression,
                              const SDFSampleDomain &sdf_domain);

  explicit DenseExpressionSDF() = default;

  ExpressionFunction exp_tpms_;
};

class FieldDrivenDenseExpressionSDF : public DenseSDF {
 public:
  using Expression_1 = std::function<double(double x, double y, double z)>;
  using Expression_2 = std::function<double(double x, double y, double z, double coeff)>;

 public:                             
  explicit FieldDrivenDenseExpressionSDF(const Expression_2 &tpms_expression,
                                         double tpms_coeff,
                                         const Expression_1 &field_expression,
                                         double field_coeff,
                                         const SDFSampleDomain &sdf_domain);

  explicit FieldDrivenDenseExpressionSDF(const Expression_2 &tpms_expression,
                                         double tpms_coeff,
                                         const ScalarField &field_matrix,
                                         double field_coeff,
                                         const SDFSampleDomain &sdf_domain);

  explicit FieldDrivenDenseExpressionSDF() = default;

  Expression_1 exp_field_;
  Expression_2 exp_tpms_;
  ScalarField matrix_field_;
};

}  // namespace sha
}  // namespace da