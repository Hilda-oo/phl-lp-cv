#include "anisotropic_mat_utils.h"
#include <igl/AABB.h>
#include <igl/fast_winding_number.h>
#include <spdlog/spdlog.h>
#include <cstddef>
#include <random>
#include <string>
#include <vector>
#include "Eigen/src/Core/Matrix.h"
#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-surface-mesh/mesh3.h"
#include "sha-surface-mesh/sample.h"

namespace da {

AnisotropicMatWrapper::AnisotropicMatWrapper() {
  top_density_.clear();
  TT_.resize(0, 0);
  TV_.resize(0, 0);
  stress_on_point_.resize(0, 3);
}

auto AnisotropicMatWrapper::getAnisotropicMatByFem(
    const std::shared_ptr<sha::FEMTetQuasiSimulator>& fem_simulator, Eigen::MatrixXd& query_points,
    bool field_flag) -> std::vector<Eigen::Matrix3d> {
  Eigen::VectorXd QU;
  Eigen::MatrixXd Qstress;  // (nQ, 6)
  fem_simulator->postprocess(query_points, QU, Qstress);

  int query_num = static_cast<int>(query_points.rows());
  std::vector<Eigen::Matrix3d> query_field(query_num);
  if (field_flag) {
    Eigen::Matrix<double, 2, 3> stress_range;
    stress_range.row(0)              = Qstress.leftCols<3>().colwise().minCoeff();
    stress_range.row(1)              = Qstress.leftCols<3>().colwise().maxCoeff();
    Eigen::Vector3d stress_range_new = stress_range.cwiseAbs().colwise().maxCoeff();
    const double stress_scale        = 7.0;
    for (int qI = 0; qI < query_num; ++qI) {
      query_field[qI].setZero();
      query_field[qI](0, 0) = abs(Qstress(qI, 0));
      query_field[qI](1, 1) = abs(Qstress(qI, 1));
      query_field[qI](2, 2) = abs(Qstress(qI, 2));
      query_field[qI](0, 0) /= stress_range_new(0);
      query_field[qI](1, 1) /= stress_range_new(1);
      query_field[qI](2, 2) /= stress_range_new(2);
      // normalize stress
      // query_field[qI](0, 0) += stress_range_new(0);
      // query_field[qI](0, 0) /= 2.0 * stress_range_new(0);
      // query_field[qI](1, 1) += stress_range_new(1);
      // query_field[qI](1, 1) /= 2.0 * stress_range_new(1);
      // query_field[qI](2, 2) += stress_range_new(2);
      // query_field[qI](2, 2) /= 2.0 * stress_range_new(2);

      query_field[qI] *= stress_scale;
    }
  } else {
    const double displacement_range = QU.cwiseAbs().maxCoeff();
    const double displacement_scale = 5;
    for (int qI = 0; qI < query_num; ++qI) {
      query_field[qI].setZero();
      query_field[qI](0, 0) = QU(qI);
      query_field[qI](1, 1) = QU(qI);
      query_field[qI](2, 2) = QU(qI);
      // normalize displacement
      query_field[qI](0, 0) += displacement_range;
      query_field[qI](0, 0) /= 2.0 * displacement_range;
      query_field[qI](1, 1) += displacement_range;
      query_field[qI](1, 1) /= 2.0 * displacement_range;
      query_field[qI](2, 2) += displacement_range;
      query_field[qI](2, 2) /= 2.0 * displacement_range;

      query_field[qI] *= displacement_scale;
    }
  }
  return query_field;
}

auto AnisotropicMatWrapper::getAnisotropicMatByFem2(
    const std::shared_ptr<sha::FEMTetQuasiSimulator>& fem_simulator, Eigen::MatrixXd& query_points,
    bool field_flag) -> std::vector<Eigen::Matrix3d> {
  Eigen::VectorXd QU;
  Eigen::MatrixXd Qstress;  // (nQ, 6)
  fem_simulator->postprocess(query_points, QU, Qstress);
  int query_num = static_cast<int>(query_points.rows());
  std::vector<Eigen::Matrix3d> query_field(query_num);
  if (field_flag) {
    const double stress_scale = 7.0;
    for (int qI = 0; qI < query_num; ++qI) {
      Eigen::Matrix3d stressMat;
      stressMat << Qstress(qI, 0), Qstress(qI, 3), Qstress(qI, 5), Qstress(qI, 3), Qstress(qI, 1),
          Qstress(qI, 4), Qstress(qI, 5), Qstress(qI, 4), Qstress(qI, 2);
      std::pair<Eigen::MatrixXd, Eigen::MatrixXd> vec_val = eigenDecomposition(stressMat);
      Eigen::MatrixXd evecs                               = vec_val.first;
      Eigen::MatrixXd evals                               = vec_val.second;
      // std::cout<<evals << std::endl << evecs <<std::endl << evals.sum();
      Assert(evals.sum() != 0, "sum of lambda = 0!");
      // evals *= 1 / evals.sum();
      index_t evalsMax;
      evals.rowwise().sum().maxCoeff(&evalsMax);
      Eigen::Matrix3d evecsMax;
      evecsMax << evecs(0, evalsMax), 0, 0, 0, evecs(1, evalsMax), 0, 0, 0, evecs(2, evalsMax);
      query_field[qI] = evecsMax.cwiseAbs();
      query_field[qI] *= stress_scale;
    }
  } else {
    const double displacement_range = QU.cwiseAbs().maxCoeff();
    const double displacement_scale = 5;
    for (int qI = 0; qI < query_num; ++qI) {
      query_field[qI].setZero();
      query_field[qI](0, 0) = QU(qI);
      query_field[qI](1, 1) = QU(qI);
      query_field[qI](2, 2) = QU(qI);
      // normalize displacement
      query_field[qI](0, 0) += displacement_range;
      query_field[qI](0, 0) /= 2.0 * displacement_range;
      query_field[qI](1, 1) += displacement_range;
      query_field[qI](1, 1) /= 2.0 * displacement_range;
      query_field[qI](2, 2) += displacement_range;
      query_field[qI](2, 2) /= 2.0 * displacement_range;

      query_field[qI] *= displacement_scale;
    }
  }
  return query_field;
}

auto AnisotropicMatWrapper::getAnisotropicMatByTopDensity(const fs_path& field_path,
                                                          Eigen::MatrixXd& query_points,
                                                          bool field_flag)
    -> std::vector<Eigen::Matrix3d> {
  log::info("get anisotropic matrix by topopt");
  if (top_density_.empty()) {
    top_density_ = readDensityFromFile(field_path);
    log::info("Loading density field");
  }

  int num_property = top_density_.size();
  int num_point    = query_points.rows();
  std::vector<Eigen::Matrix3d> anisotropicMat(0);
  auto getIndexFromCoord = [](double x, int min, int max) -> int {
    int new_x = std::round(x + (max - min) / 2.0);
    // x = std::trunc(x + (max - min) / 2);
    // x = ceil(x + (max - min) / 2);
    // x = floor(x + (max - min) / 2);
    return new_x;
  };
  int lx = 100, ly = 30, lz = 30;
  for (index_t pI = 0; pI < num_point; pI++) {
    int x              = getIndexFromCoord(query_points.row(pI)[0], 0, lx);
    int y              = getIndexFromCoord(query_points.row(pI)[1], 0, ly);
    int z              = getIndexFromCoord(query_points.row(pI)[2], 0, lz);
    int index_property = z * lx * ly + y * lx + x;
    Eigen::Matrix3d mat;
    if (index_property < 0 || index_property > num_property) {
      mat.setZero();
    } else {
      double value = top_density_.at(index_property);
      for (index_t mI = 0; mI < 3; mI++) {
        mat(mI, mI) = value;
      }
    }
    anisotropicMat.push_back(mat);
  }
  Assert(anisotropicMat.size() == num_point);
  return anisotropicMat;
}

auto AnisotropicMatWrapper::getAnisotropicMatByTopStress(const fs_path& field_base_path,
                                                         Eigen::MatrixXd& query_points,
                                                         bool field_flag)
    -> std::vector<Eigen::Matrix3d> {
  log::info("get stress by topopt");
  if (stress_on_point_.size() == 0) {
    fs_path stressX_path = field_base_path / "stressX_field_matrix.vtk";
    fs_path stressY_path = field_base_path / "stressY_field_matrix.vtk";
    fs_path stressZ_path = field_base_path / "stressZ_field_matrix.vtk";
    Eigen::VectorXd stress_x;
    Eigen::VectorXd stress_y;
    Eigen::VectorXd stress_z;
    stress_x = readCellDataFromVTK(stressX_path);
    stress_y = readCellDataFromVTK(stressY_path);
    stress_z = readCellDataFromVTK(stressZ_path);
    Assert(stress_x.size() == stress_y.size() && stress_x.size() == stress_z.size(),
           "stress'scale does not match each other");
    stress_on_point_.resize(stress_x.size(), 3);
    stress_on_point_.col(0) = stress_x;
    stress_on_point_.col(1) = stress_y;
    stress_on_point_.col(2) = stress_z;
    sha::WriteMatrixToFile(WorkingResultDirectoryPath() / "debug/stress_on_points.txt",
                           stress_on_point_);
    // // 23/6/2 add laplace smooth
    // stress_on_point_ = fieldLaplaceSmooth(stress_on_point_, 1);

    // sha::WriteMatrixToFile(WorkingResultDirectoryPath() / "debug/laplace_stress_on_points.txt",
    //                        stress_on_point_);
    log::info("Loading stress obtained by topopt");
  }

  int num_property = stress_on_point_.rows();
  int num_point    = query_points.rows();
  std::vector<Eigen::Matrix3d> anisotropicMat(num_point);
  auto getIndexFromCoord = [](double x, int min, int max) -> int {
    int new_x = std::round(x + (max - min) / 2.0);
    // int new_x = std::trunc(x + (max - min) / 2);
    // int new_x = ceil(x + (max - min) / 2);
    // int new_x = floor(x + (max - min) / 2);
    return new_x;
  };
  int lx = 100, ly = 30, lz = 30;
  Eigen::MatrixXd query_stress_mat(num_point, 3);
  for (index_t pI = 0; pI < num_point; pI++) {
    int x              = getIndexFromCoord(query_points.row(pI)[0], 0, lx);
    int y              = getIndexFromCoord(query_points.row(pI)[1], 0, ly);
    int z              = getIndexFromCoord(query_points.row(pI)[2], 0, lz);
    y                  = y > 0 ? y - 1 : 0;
    z                  = z > 0 ? z - 1 : 0;
    int index_property = z * lx * ly + y * lx + x;
    if (index_property < 0 || index_property > num_property) {
      log::warn("index_property={},num_property={},x={},{},y={},{},z={},{}", index_property,
                num_property, query_points.row(pI)[0], x, query_points.row(pI)[1], y,
                query_points.row(pI)[2], z);
      query_stress_mat.row(pI) << 1e-5, 1e-5, 1e-5;
    } else {
      query_stress_mat.row(pI) = stress_on_point_.row(index_property);
    }
  }
  sha::WriteMatrixToFile(WorkingResultDirectoryPath() / "debug/query_stress_mat.txt",
                         query_stress_mat);
  Eigen::Vector3d stress_range;
  stress_range = query_stress_mat.cwiseAbs().leftCols<3>().colwise().maxCoeff();
  // const double stress_scale        = 0.2;  //l-r
  // const double stress_scale        = 0.08;     //l-rightTop
  const double stress_scale = 0.24;  // l-rightBottom2
  for (index_t qI = 0; qI < num_point; qI++) {
    anisotropicMat[qI].setZero();
    anisotropicMat[qI](0, 0) = query_stress_mat(qI, 0);
    anisotropicMat[qI](1, 1) = query_stress_mat(qI, 1);
    anisotropicMat[qI](2, 2) = query_stress_mat(qI, 2);
    // anisotropicMat[qI](0, 0) = query_stress_mat.cwiseAbs()(qI, 0);
    // anisotropicMat[qI](1, 1) = query_stress_mat.cwiseAbs()(qI, 1);
    // anisotropicMat[qI](2, 2) = query_stress_mat.cwiseAbs()(qI, 2);
    // normalize stress
    anisotropicMat[qI](0, 0) /= stress_range(0);
    anisotropicMat[qI](1, 1) /= stress_range(1);
    anisotropicMat[qI](2, 2) /= stress_range(2);

    anisotropicMat[qI] *= stress_scale;
  }
  return anisotropicMat;
}

void AnisotropicMatWrapper::generateSampleSeedEntry(const fs_path& mesh_path,
                                                    const fs_path& seed_path, size_t num_seed) {
  fs_path base                = WorkingAssetDirectoryPath();
  sha::MatMesh3 mesh          = sha::ReadMatMeshFromOBJ(base / mesh_path);
  Eigen::MatrixXd sample_seed = sha::SamplePointsInMeshVolumeUniformly(mesh, num_seed);
  sha::WriteMatrixToFile(base / seed_path / fmt::format("seed-{}.txt", sample_seed.rows()),
                         sample_seed);
  sha::WritePointsToVtk(base / seed_path / fmt::format("seed-{}.vtk", sample_seed.rows()),
                        sample_seed);
  log::info("generate {} seeds successful", sample_seed.rows());
}

void AnisotropicMatWrapper::generateMixSampleSeedEntry(const fs_path& mesh_path,
                                                       const fs_path& seed_path, size_t snum,
                                                       size_t xnum, size_t ynum, size_t znum) {
  fs_path base       = WorkingAssetDirectoryPath();
  sha::MatMesh3 mesh = sha::ReadMatMeshFromOBJ(base / mesh_path);
  Eigen::AlignedBox3d aligned_box;
  aligned_box.min() = mesh.mat_coordinates.colwise().minCoeff();
  aligned_box.max() = mesh.mat_coordinates.colwise().maxCoeff();
  int num_samples   = xnum * ynum * znum + snum;

  Eigen::MatrixXd sample_seed(num_samples, 3);
  double xMin   = aligned_box.min().x();
  double xMax   = aligned_box.max().x();
  double yMin   = aligned_box.min().y();
  double yMax   = aligned_box.max().y();
  double zMin   = aligned_box.min().z();
  double zMax   = aligned_box.max().z();
  double xShift = xMax - xMin;
  double yShift = yMax - yMin;
  double zShift = zMax - zMin;
  xShift /= 1.0 * (xnum - 1);
  yShift /= 1.0 * (ynum - 1);
  zShift /= 1.0 * (znum - 1);
  spdlog::info("xShift:{}, yShift:{}, zShift:{}", xShift, yShift, zShift);

  igl::FastWindingNumberBVH fwn_bvh;
  igl::fast_winding_number(mesh.mat_coordinates.template cast<float>().eval(), mesh.mat_faces, 2,
                           fwn_bvh);
  int sn = 0;
  for (index_t i = 0; i < xnum; ++i) {
    double x = xMin + xShift * i;
    for (index_t j = 0; j < ynum; ++j) {
      double y = yMin + yShift * j;
      for (index_t k = 0; k < znum; ++k) {
        double z = zMin + zShift * k;
        Eigen::Vector3d query_point;
        query_point << x, y, z;
        sample_seed.row(sn++) = query_point;
      }
    }
  }

  std::uniform_real_distribution<> pos_samplers[3] = {std::uniform_real_distribution<>(xMin, xMax),
                                                      std::uniform_real_distribution<>(yMin, yMax),
                                                      std::uniform_real_distribution<>(zMin, zMax)};
  std::mt19937 random_engine((std::random_device())());

  for (index_t sample_idx = 0; sample_idx < snum; ++sample_idx) {
    sample_seed.row(sn++) << pos_samplers[0](random_engine), pos_samplers[1](random_engine),
        pos_samplers[2](random_engine);
  }

  num_samples = sample_seed.rows();
  spdlog::info("num-samples:{},sn:{}", num_samples, sn);

  std::vector<Eigen::Vector3d> mat_sample_coordinates;
  std::map<std::string, bool> flag; 
#pragma omp parallel for
  for (index_t sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
    int num_mismatch_samples = 0;
    Eigen::MatrixXd query_point(1, 3);
    query_point.row(0) = sample_seed.row(sample_idx);
    double w           = fast_winding_number(fwn_bvh, 2, query_point.template cast<float>().eval());
    double s           = 1. - 2. * std::abs(w);
    if (s <= 0) {
      std::string str = std::to_string(query_point(0, 0)) + std::to_string(query_point(0, 1)) + std::to_string(query_point(0, 2));
      if (!flag.count(str)) {
        flag[str] = true;
        mat_sample_coordinates.push_back(query_point.row(0));
      }
    }
  }
  num_samples = mat_sample_coordinates.size();
  sample_seed.resize(0, 0);
  sample_seed.resize(num_samples, 3);
  sn = 0;
  for(auto p: mat_sample_coordinates) {
    sample_seed.row(sn++) = p;
  }

  sha::WriteMatrixToFile(base / seed_path / fmt::format("seed-{}.txt", sample_seed.rows()),
                         sample_seed);
  sha::WritePointsToVtk(base / seed_path / fmt::format("seed-{}.vtk", sample_seed.rows()),
                        sample_seed);
  log::info("generate {} seeds successful", sample_seed.rows());
}

void AnisotropicMatWrapper::generateUniformSampleSeedEntry(const fs_path& mesh_path,
                                                           const fs_path& seed_path, size_t xnum,
                                                           size_t ynum, size_t znum) {
  fs_path base       = WorkingAssetDirectoryPath();
  sha::MatMesh3 mesh = sha::ReadMatMeshFromOBJ(base / mesh_path);
  Eigen::AlignedBox3d aligned_box;
  aligned_box.min() = mesh.mat_coordinates.colwise().minCoeff();
  aligned_box.max() = mesh.mat_coordinates.colwise().maxCoeff();
  int num_samples   = xnum * ynum * znum;

  Eigen::MatrixXd sample_seed(num_samples, 3);
  double xMin   = aligned_box.min().x();
  double xMax   = aligned_box.max().x();
  double yMin   = aligned_box.min().y();
  double yMax   = aligned_box.max().y();
  double zMin   = aligned_box.min().z();
  double zMax   = aligned_box.max().z();
  double xShift = xMax - xMin;
  double yShift = yMax - yMin;
  double zShift = zMax - zMin;
  xShift /= 1.0 * (xnum - 1);
  yShift /= 1.0 * (ynum - 1);
  zShift /= 1.0 * (znum - 1);
  spdlog::info("xShift:{}, yShift:{}, zShift:{}", xShift, yShift, zShift);

  igl::FastWindingNumberBVH fwn_bvh;
  igl::fast_winding_number(mesh.mat_coordinates.template cast<float>().eval(), mesh.mat_faces, 2,
                           fwn_bvh);
  int sn = 0;
  for (index_t i = 0; i < xnum; ++i) {
    double x = xMin + xShift * i;
    for (index_t j = 0; j < ynum; ++j) {
      double y = yMin + yShift * j;
      for (index_t k = 0; k < znum; ++k) {
        double z = zMin + zShift * k;
        Eigen::Vector3d query_point;
        query_point << x, y, z;
        sample_seed.row(sn++) = query_point;
      }
    }
  }

  num_samples = sample_seed.rows();
  spdlog::info("num-samples:{},sn:{}", num_samples, sn);

  std::vector<Eigen::Vector3d> mat_sample_coordinates(0);
#pragma omp parallel for
  for (index_t sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
    int num_mismatch_samples = 0;
    Eigen::MatrixXd query_point(1, 3);
    query_point.row(0) = sample_seed.row(sample_idx);
    double w           = fast_winding_number(fwn_bvh, 2, query_point.template cast<float>().eval());
    double s           = 1. - 2. * std::abs(w);
    if (s <= 0) {
      mat_sample_coordinates.push_back(query_point.row(0));
    }
  }
  num_samples = mat_sample_coordinates.size();
  sample_seed.resize(0, 0);
  sample_seed.resize(num_samples, 3);
  for (index_t sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
    sample_seed.row(sample_idx) = mat_sample_coordinates.at(sample_idx);
  }
  sha::WriteMatrixToFile(base / seed_path / fmt::format("seed-{}.txt", sample_seed.rows()),
                         sample_seed);
  sha::WritePointsToVtk(base / seed_path / fmt::format("seed-{}.vtk", sample_seed.rows()),
                        sample_seed);
  log::info("generate {} seeds successful", sample_seed.rows());
}

auto AnisotropicMatWrapper::getAdjPoints(Eigen::Vector3d& point) -> Eigen::MatrixXd {
  std::vector<Eigen::Vector3d> adjPoint_vector(0);
  Eigen::Vector3d origin_point = point.array() - 1;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        Eigen::Vector3d temp;
        temp << i, j, k;
        Eigen::Vector3d adj_point = origin_point + temp;
        if (adj_point.minCoeff() >= 0 && adj_point != point) adjPoint_vector.push_back(adj_point);
      }
    }
  }
  Eigen::MatrixXd adjPoint_matrix(adjPoint_vector.size(), 3);
  for (int i = 0; i < adjPoint_vector.size(); ++i) {
    adjPoint_matrix.row(i) = adjPoint_vector.at(i);
  }
  return adjPoint_matrix;
}

auto AnisotropicMatWrapper::fieldLaplaceSmooth(Eigen::MatrixXd& field_maxtrix, double lmd)
    -> Eigen::MatrixXd {
  int fN = field_maxtrix.rows();
  Eigen::MatrixXd new_field(fN, 3);
  int lx = 100, ly = 30, lz = 30;
  int nfN = 0;
  for (int i = 0; i < lx; ++i) {
    for (int j = 0; j < ly; ++j) {
      for (int k = 0; k < lz; ++k) {
        Eigen::Vector3d point(i, j, k);
        Eigen::MatrixXd adjPoints = getAdjPoints(point);
        Eigen::Vector3d point_field(0, 0, 0);
        int adj_num = adjPoints.rows();
        for (int pI = 0; pI < adj_num; ++pI) {
          int fI =
              (adjPoints(pI, 2) - 1) * lx * ly + (adjPoints(pI, 1) - 1) * lx + adjPoints(pI, 0) - 1;
          Assert(fI < fN, "fI >= fN, match no field matrix");
          point_field += field_maxtrix.row(fI);
        }
        new_field.row(nfN++) = point_field / adj_num;
      }
    }
  }
  for (int fI = 0; fI < fN; ++fI) {
    new_field.row(fI) = field_maxtrix.row(fI) + lmd * (new_field.row(fI) - field_maxtrix.row(fI));
  }
  return new_field;
}

void AnisotropicMatWrapper::writeStressOrientation(std::vector<Eigen::Matrix3d> stress,
                                                   Eigen::MatrixXd points,
                                                   const fs_path& out_path) {
  std::ofstream vtk_out_stream(out_path.string());
  if (!vtk_out_stream.is_open()) {
    Terminate("File " + out_path.string() + " can not be opened.");
  }
  vtk_out_stream << "# vtk DataFile Version 3.0\n"
                    "Volume Mesh\n"
                    "ASCII\n"
                    "DATASET UNSTRUCTURED_GRID\n";
  int point_num = points.rows();
  vtk_out_stream << "POINTS " << point_num * 2 << " float\n";
  for (index_t pI = 0; pI < point_num; ++pI) {
    Eigen::Matrix3d stressMat  = stress.at(pI);
    Eigen::Vector3d startPoint = points.row(pI);
    Eigen::Vector3d endPoint;
    double norm_num =
        sqrt(pow(stressMat(0, 0), 2) + pow(stressMat(1, 1), 2) + pow(stressMat(2, 2), 2));
    double line_scale = 2.0 / (norm_num == 0 ? 1 : norm_num);
    endPoint << startPoint[0] + line_scale * stressMat(0, 0),
        startPoint[1] + line_scale * stressMat(1, 1), startPoint[2] + line_scale * stressMat(2, 2);
    vtk_out_stream << startPoint[0] << " " << startPoint[1] << " " << startPoint[2] << "\n";
    vtk_out_stream << endPoint[0] << " " << endPoint[1] << " " << endPoint[2] << "\n";
  }
  vtk_out_stream << "CELLS " << point_num * 2 << " " << point_num * 5 << "\n";
  for (index_t cI = 0; cI < point_num; ++cI) {
    vtk_out_stream << "2 " << 2 * cI << " " << 2 * cI + 1 << "\n";
  }
  for (index_t cI = 0; cI < point_num; ++cI) {
    vtk_out_stream << "1 " << 2 * cI << "\n";
  }
  vtk_out_stream << "CELL_TYPES " << point_num * 2 << "\n";
  for (index_t ctI = 0; ctI < point_num; ++ctI) {
    vtk_out_stream << "3\n";
  }
  for (index_t ctI = 0; ctI < point_num; ++ctI) {
    vtk_out_stream << "2\n";
  }
  vtk_out_stream << "CELL_DATA " << point_num * 2 << "\n";
  vtk_out_stream << "SCALARS cell_scalars double 3\n"
                    "LOOKUP_TABLE default\n";
  for (index_t csI = 0; csI < point_num; ++csI) {
    Eigen::Matrix3d stressMat = stress.at(csI);
    vtk_out_stream << stressMat(0, 0) << " " << stressMat(1, 1) << " " << stressMat(2, 2)
                   << std::endl;
  }
  for (index_t csI = 0; csI < point_num; ++csI) {
    Eigen::Matrix3d stressMat = stress.at(csI);
    vtk_out_stream << stressMat(0, 0) << " " << stressMat(1, 1) << " " << stressMat(2, 2)
                   << std::endl;
  }
}

auto AnisotropicMatWrapper::readDensityFromFile(const fs_path& field_mat_path)
    -> std::vector<double> {
  std::vector<double> vector_data(0);
  std::ifstream vector_instream(field_mat_path.string());
  double value;
  std::vector<int> scale(0);
  while (vector_instream >> value) {
    scale.push_back(value);
    if (scale.size() == 10) break;
  }
  while (vector_instream >> value) {
    vector_data.push_back(value);
  }
  Assert(vector_data.size() == scale.back(), "vector_data.size != scale.back()");
  return vector_data;
}

auto AnisotropicMatWrapper::readCellDataFromVTK(const fs_path& path) -> Eigen::VectorXd {
  std::ifstream vector_instream(path.string());
  std::string line_string;
  std::vector<std::string> split_line_string;
  size_t data_size;
  while (std::getline(vector_instream, line_string)) {
    boost::split(split_line_string, line_string, boost::is_any_of(" "));
    bool data_locate_flag = false;
    for (auto& value_string : split_line_string) {
      if (value_string == "CELL_DATA") {
        std::getline(vector_instream, line_string);
        std::getline(vector_instream, line_string);
        data_locate_flag = true;
        break;
      }
    }
    if (data_locate_flag) {
      std::istringstream(split_line_string.back()) >> data_size;
      break;
    }
  }
  std::vector<double> vector_data(data_size);
  for (int i = 0; i < data_size; i++) {
    vector_instream >> vector_data.at(i);
  }
  Eigen::VectorXd vector_value(data_size);
  std::copy_n(vector_data.data(), data_size, vector_value.data());
  return vector_value;
}

auto AnisotropicMatWrapper::eigenDecomposition(const Eigen::MatrixXd& A)
    -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
  const int m = A.rows(), n = A.cols();
  Assert(m == n, "not a nxn matrix!");
  Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver(A);
  Eigen::MatrixXcd eigenValue  = eigenSolver.eigenvalues();
  Eigen::MatrixXcd eigenVector = eigenSolver.eigenvectors();
  eigenValue.lpNorm<6>();
  return std::make_pair(eigenVector.real(), eigenValue.real());
}

}  // namespace da