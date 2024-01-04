#include "anisotropic_mat_utils.h"
#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/sample.h"

namespace da {

AnisotropicMatWrapper::AnisotropicMatWrapper() {}

auto AnisotropicMatWrapper::getAnisotropicMatByFemStress(Eigen::MatrixXd& query_points,
                                                         std::vector<Eigen::VectorXd> stress_field)
    -> std::vector<Eigen::Matrix3d> {
  int qN = static_cast<int>(query_points.rows());
  Assert(qN == stress_field.size(), "the number of points != the rows of stress_field");
  std::vector<Eigen::Matrix3d> stressQ(qN);

  for (index_t pI = 0; pI < qN; ++pI) {
    stressQ.at(pI).setIdentity();
    stressQ[pI](0, 0) = stress_field[pI](0);
    stressQ[pI](1, 1) = stress_field[pI](1);
    stressQ[pI](2, 2) = stress_field[pI](2);
  }
  log::debug("process stress successful", stressQ.size());
  return stressQ;
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