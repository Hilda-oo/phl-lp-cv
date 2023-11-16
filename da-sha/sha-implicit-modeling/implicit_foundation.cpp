#include "implicit.h"

#include <igl/marching_cubes.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/signed_distance.h>
#include <igl/writeOBJ.h>
#include <fstream>
#include <iostream>
#include <string>

#include "utility.h"

namespace da {
namespace sha {
// ---------------- ScalarField --------------------
ScalarField::ScalarField(const DataType &values, const SDFSampleDomain &sdf_domain)
    : values_(values), sdf_sample_domain_(sdf_domain) {}
ScalarField::ScalarField(const SDFSampleDomain &sdf_domain)
    : values_(DataType::Zero(0)), sdf_sample_domain_(sdf_domain) {}

index_t ScalarField::ComputeArrayIndex(index_t idx_i, index_t idx_j, index_t idx_k) const {
  return ((idx_k * sdf_sample_domain_.num_samples.y()) + idx_j) *
             sdf_sample_domain_.num_samples.x() +
         idx_i;
}

double ScalarField::StandardSample(double x, double y, double z) const {
  x            = (sdf_sample_domain_.num_samples.x() - 1) * x;
  y            = (sdf_sample_domain_.num_samples.y() - 1) * y;
  z            = (sdf_sample_domain_.num_samples.z() - 1) * z;
  int bound[6] = {static_cast<int>(x),     static_cast<int>(x) + 1, static_cast<int>(y),
                  static_cast<int>(y) + 1, static_cast<int>(z),     static_cast<int>(z) + 1};
  double v[8];
  v[0] = values_(ComputeArrayIndex(bound[0], bound[2], bound[4]), 0);
  v[1] = values_(ComputeArrayIndex(bound[1], bound[2], bound[4]), 0);
  v[2] = values_(ComputeArrayIndex(bound[0], bound[3], bound[4]), 0);
  v[3] = values_(ComputeArrayIndex(bound[1], bound[3], bound[4]), 0);
  v[4] = values_(ComputeArrayIndex(bound[0], bound[2], bound[5]), 0);
  v[5] = values_(ComputeArrayIndex(bound[1], bound[2], bound[5]), 0);
  v[6] = values_(ComputeArrayIndex(bound[0], bound[3], bound[5]), 0);
  v[7] = values_(ComputeArrayIndex(bound[1], bound[3], bound[5]), 0);
  return LinearInterpolateFor3D(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], bound[0], bound[1],
                                x, bound[2], bound[3], y, bound[4], bound[5], z);
}

double ScalarField::Sample(double x, double y, double z) const {
  x          = x - sdf_sample_domain_.domain.min().x();
  y          = y - sdf_sample_domain_.domain.min().y();
  z          = z - sdf_sample_domain_.domain.min().z();
  auto scale = sdf_sample_domain_.domain.sizes();
  x          = x / scale.x();
  y          = y / scale.y();
  z          = z / scale.z();
  return StandardSample(x, y, z);
}

void ScalarField::DumpTo(const std::string &path) {
  std::ofstream out(path);
  out.precision(64);
  Assert(out.is_open());

  out << sdf_sample_domain_.domain.min().x() << " " << sdf_sample_domain_.domain.min().y() << " "
      << sdf_sample_domain_.domain.min().z() << std::endl;
  out << sdf_sample_domain_.domain.max().x() << " " << sdf_sample_domain_.domain.max().y() << " "
      << sdf_sample_domain_.domain.max().z() << std::endl;
  out << sdf_sample_domain_.num_samples.x() << " " << sdf_sample_domain_.num_samples.y() << " "
      << sdf_sample_domain_.num_samples.z() << std::endl;
  out << values_.size() << std::endl;
  out.precision(12);
  for (int i = 0; i < values_.size(); ++i) {
    out << values_(i) << std::endl;
  }
}

void ScalarField::LoadFrom(const std::string &path) {
  std::ifstream in(path);
  Assert(in.is_open());

  int value_size;
  in >> sdf_sample_domain_.domain.min().x() >> sdf_sample_domain_.domain.min().y() >>
      sdf_sample_domain_.domain.min().z();
  in >> sdf_sample_domain_.domain.max().x() >> sdf_sample_domain_.domain.max().y() >>
      sdf_sample_domain_.domain.max().z();
  in >> sdf_sample_domain_.num_samples.x() >> sdf_sample_domain_.num_samples.y() >>
      sdf_sample_domain_.num_samples.z();
  in >> value_size;
  values_.resize(value_size);
  for (int i = 0; i < value_size; ++i) {
    in >> values_(i);
  }
}
// ---------------- ScalarField --------------------

// ---------------- DenseSDF --------------------
DenseSDF::DenseSDF(const DataType &value, const SDFSampleDomain &sdf_domain)
    : ScalarField(value, sdf_domain) {}

DenseSDF::DenseSDF(const SDFSampleDomain &sdf_domain) : DenseSDF(DataType::Zero(0), sdf_domain) {}

MatMesh3 DenseSDF::GenerateMeshByMarchingCubes(double isovalue) const {
  Eigen::MatrixXd sample_points;
  MatMesh3 matmesh;

  sample_points = GetSamplePoints();

  igl::marching_cubes(values_, sample_points, sdf_sample_domain_.num_samples(0),
                      sdf_sample_domain_.num_samples(1), sdf_sample_domain_.num_samples(2), 0,
                      matmesh.mat_coordinates, matmesh.mat_faces);

  return matmesh;
}

void DenseSDF::ExportObj(const std::string &filename) const {
  MatMesh3 matmesh = this->GenerateMeshByMarchingCubes(0);
  igl::writeOBJ(filename, matmesh.mat_coordinates, matmesh.mat_faces);
}

Eigen::MatrixXd DenseSDF::GetSamplePoints() const {
  return GetSamplePoints(sdf_sample_domain_.num_samples, sdf_sample_domain_.domain);
}

Eigen::MatrixXd DenseSDF::GetSamplePoints(const Eigen::Vector3i &size,
                                          const Eigen::AlignedBox3d &box) {
  Eigen::MatrixXd sample_points;
  sample_points.resize(size.x() * size.y() * size.z(), 3);

  for (index_t k = 0; k < size(2); ++k) {
    for (index_t j = 0; j < size(1); ++j) {
      for (index_t i = 0; i < size(0); ++i) {
        sample_points(i + j * size(0) + k * size(0) * size(1), 0) =
            box.min()(0) + i * box.sizes()(0) / size(0);
        sample_points(i + j * size(0) + k * size(0) * size(1), 1) =
            box.min()(1) + j * box.sizes()(1) / size(1);
        sample_points(i + j * size(0) + k * size(0) * size(1), 2) =
            box.min()(2) + k * box.sizes()(2) / size(2);
      }
    }
  }
  return sample_points;
}

double &DenseSDF::At(index_t idx_i, index_t idx_j, index_t idx_k) {
  return values_(idx_i + idx_j * sdf_sample_domain_.num_samples(0) +
                 idx_k * sdf_sample_domain_.num_samples(0) * sdf_sample_domain_.num_samples(1));
}

double &DenseSDF::operator()(index_t idx_i, index_t idx_j, index_t idx_k) {
  return At(idx_i, idx_j, idx_k);
}

const double &DenseSDF::operator()(index_t idx_i, index_t idx_j, index_t idx_k) const {
  return At(idx_i, idx_j, idx_k);
}

const double &DenseSDF::At(index_t idx_i, index_t idx_j, index_t idx_k) const {
  return values_(idx_i + idx_j * sdf_sample_domain_.num_samples(0) +
                 idx_k * sdf_sample_domain_.num_samples(0) * sdf_sample_domain_.num_samples(1));
}

DenseSDF DenseSDF::Union(const DenseSDF &other) const {
  if (values_.size() == 0) return other;
  Assert(sdf_sample_domain_ == other.sdf_sample_domain_);
  Assert(values_.rows() == other.values_.rows() && values_.cols() == other.values_.cols());
  return DenseSDF(values_.cwiseMax(other.values_), sdf_sample_domain_);
}
DenseSDF DenseSDF::Intersect(const DenseSDF &other) const {
  if (values_.size() == 0) return *this;
  if (other.values_.size() == 0) return other;
  Assert(sdf_sample_domain_ == other.sdf_sample_domain_,
         fmt::format("Domains are not same: ({}, {}, {}) and ({}, {}, {})",
                     sdf_sample_domain_.domain.sizes().x(), sdf_sample_domain_.domain.sizes().y(),
                     sdf_sample_domain_.domain.sizes().z(),
                     other.sdf_sample_domain_.domain.sizes().x(),
                     other.sdf_sample_domain_.domain.sizes().y(),
                     other.sdf_sample_domain_.domain.sizes().z()));
  Assert(values_.rows() == other.values_.rows() && values_.cols() == other.values_.cols());
  return DenseSDF(values_.cwiseMin(other.values_), sdf_sample_domain_);
}
DenseSDF DenseSDF::Complement() const { return DenseSDF(-values_, sdf_sample_domain_); }

DenseSDF DenseSDF::Difference(const DenseSDF &other) const {
  Assert(sdf_sample_domain_ == other.sdf_sample_domain_);
  Assert(values_.rows() == other.values_.rows() && values_.cols() == other.values_.cols());
  return Intersect(other.Complement());
}

void DenseSDF::MoveInPlace(const Eigen::Vector3i &offset, DenseSDF::PaddingType padding_type,
                           double padding) {
  DenseSDF new_sdf = Move(offset, padding_type, padding);
  this->values_    = new_sdf.values_;
}

DenseSDF DenseSDF::Move(const Eigen::Vector3i &offset, DenseSDF::PaddingType padding_type,
                        double padding) const {
  return MoveInPart({Eigen::Vector3i::Zero(), sdf_sample_domain_.num_samples}, offset, padding_type,
                    padding);
}

DenseSDF DenseSDF::Offset(double offset) {
  return DenseSDF(values_.array() + offset, sdf_sample_domain_);
}

void DenseSDF::OffsetInPlace(double offset) { values_.array() += offset; }

DenseSDF DenseSDF::MoveInPart(const Eigen::AlignedBox<int, 3> &range, const Eigen::Vector3i &offset,
                              DenseSDF::PaddingType padding_type, double padding) const {
  const int k_min = std::max(0, range.min().z());
  const int k_max = std::min(range.max().z(), sdf_sample_domain_.num_samples.z());
  const int j_min = std::max(0, range.min().y());
  const int j_max = std::min(range.max().y(), sdf_sample_domain_.num_samples.y());
  const int i_min = std::max(0, range.min().x());
  const int i_max = std::min(range.max().x(), sdf_sample_domain_.num_samples.x());

  Eigen::Vector3i shift = offset;
  shift.x() =
      ((shift.x() % sdf_sample_domain_.num_samples.x()) + sdf_sample_domain_.num_samples.x()) %
      sdf_sample_domain_.num_samples.x();
  shift.y() =
      ((shift.y() % sdf_sample_domain_.num_samples.y()) + sdf_sample_domain_.num_samples.y()) %
      sdf_sample_domain_.num_samples.y();
  shift.z() =
      ((shift.z() % sdf_sample_domain_.num_samples.z()) + sdf_sample_domain_.num_samples.z()) %
      sdf_sample_domain_.num_samples.z();
  auto new_sdf = *this;
  for (index_t k = k_min; k < k_max; ++k) {
    for (index_t j = j_min; j < j_max; ++j) {
      for (index_t i = i_min; i < i_max; ++i) {
        Eigen::Vector3i origin = Eigen::Vector3i{i, j, k} + shift;
        if (origin.x() < 0 || origin.x() >= sdf_sample_domain_.num_samples.x() || origin.y() < 0 ||
            origin.y() >= sdf_sample_domain_.num_samples.y() || origin.z() < 0 ||
            origin.z() >= sdf_sample_domain_.num_samples.z()) {
          if (padding_type == PaddingType::PADDING_ZERO)
            padding = 0;
          else if (padding_type == PaddingType::PADDING_INF)
            padding = std::numeric_limits<double>::infinity();
          else if (padding_type == PaddingType::PADDING_NEGINF)
            padding = -std::numeric_limits<double>::infinity();
          else if (padding_type == PaddingType::PADDING_SAME) {
            if (origin.x() < 0) {
              origin.x() = 0;
            } else if (origin.x() >= sdf_sample_domain_.num_samples.x()) {
              origin.x() = sdf_sample_domain_.num_samples.x() - 1;
            }
            if (origin.y() < 0) {
              origin.y() = 0;
            } else if (origin.y() >= sdf_sample_domain_.num_samples.y()) {
              origin.y() = sdf_sample_domain_.num_samples.y() - 1;
            }
            if (origin.z() < 0) {
              origin.z() = 0;
            } else if (origin.z() >= sdf_sample_domain_.num_samples.z()) {
              origin.z() = sdf_sample_domain_.num_samples.z() - 1;
            }
          }
          new_sdf.At(i, j, k) = padding;
        } else {
          new_sdf.At(i, j, k) = At(origin.x(), origin.y(), origin.z());
        }
      }
    }
  }
  return new_sdf;
}

void DenseSDF::MoveInPartInPlace(const Eigen::AlignedBox<int, 3> &range,
                                 const Eigen::Vector3i &offset, DenseSDF::PaddingType padding_type,
                                 double padding) {
  DenseSDF new_sdf = MoveInPart(range, offset, padding_type, padding);
  this->values_    = new_sdf.values_;
}
// ---------------- DenseSDF --------------------
}  // namespace sha
}  // namespace da
