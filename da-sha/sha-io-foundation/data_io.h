#pragma once

#include "sha-base-framework/frame.h"

#include <Eigen/Eigen>
#include <boost/algorithm/string.hpp>
#include <fstream>

namespace da {
namespace sha {
template <class Scalar>
inline auto ReadVectorFromFile(const fs_path &path) -> Eigen::Vector<Scalar, -1> {
  Eigen::Vector<Scalar, -1> vector_data;
  std::vector<Scalar> vector_values;
  Scalar value;
  std::ifstream vector_instream(path.string());
  while (vector_instream >> value) {
    vector_values.push_back(value);
  }
  size_t vector_size = vector_values.size();
  vector_data.resize(vector_size);
  std::copy_n(vector_values.data(), vector_size, vector_data.data());
  return vector_data;
}

template <class Scalar>
inline void WriteVectorToFile(const fs_path &path, const Eigen::Vector<Scalar, -1> &vector) {
  std::ofstream vector_outstream(path.string());
  vector_outstream << vector << std::endl;
}

template <class Scalar>
inline auto ReadMatrixFromFile(const fs_path &path) -> Eigen::Matrix<Scalar, -1, -1> {
  Eigen::Matrix<Scalar, -1, -1> matrix_data;
  std::ifstream matrix_instream(path.string());
  std::string line_string;
  std::vector<std::string> split_line_string;
  std::vector<std::vector<double>> matrix_values;

  while (std::getline(matrix_instream, line_string)) {
    matrix_values.emplace_back();
    boost::split(split_line_string, line_string, boost::is_any_of(" "));
    for (auto &value_string : split_line_string) {
      if (value_string.empty()) continue;
      matrix_values.back().push_back(static_cast<Scalar>(std::stod(value_string)));
    }
  }
  if (matrix_values.empty()) {
    return matrix_data;
  }
  matrix_data.resize(matrix_values.size(), matrix_values[0].size());
  for (int i = 0; i < matrix_values.size(); i++) {
    for (int j = 0; j < matrix_values[0].size(); j++) {
      matrix_data(i, j) = matrix_values[i][j];
    }
  }
  return matrix_data;
}

template <class Scalar>
inline void WriteMatrixToFile(const fs_path &path, const Eigen::Matrix<Scalar, -1, -1> &matrix) {
  std::ofstream matrix_outstream(path.string());
  matrix_outstream << matrix << std::endl;
}

static const auto &ReadDoubleMatrixFromFile = ReadMatrixFromFile<double>;
static const auto &ReadIntMatrixFromFile    = ReadMatrixFromFile<int>;
static const auto &ReadDoubleVectorFromFile = ReadVectorFromFile<double>;
static const auto &ReadIntVectorFromFile    = ReadVectorFromFile<int>;
}  // namespace sha
}  // namespace da