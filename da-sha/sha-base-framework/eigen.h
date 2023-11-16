#pragma once

#include <Eigen/Eigen>
#include <vector>

namespace da {
namespace sha {
template <typename Type>
Eigen::Vector<Type, -1> ConvertStlVectorToEigenVector(const std::vector<Type> &vector) {
  Eigen::Vector<Type, -1> eigen_vector(vector.size());
  for (int idx = 0; idx < vector.size(); ++idx) {
    eigen_vector(idx) = vector[idx];
  }
  return eigen_vector;
}

template <typename Type>
std::vector<Type> ConvertEigenVectorToStlVector(const Eigen::Vector<Type, -1> &eigen_vector) {
  std::vector<Type> vector(eigen_vector.size());
  for (int idx = 0; idx < vector.size(); ++idx) {
    vector[idx] = eigen_vector(idx);
  }
  return vector;
}

}  // namespace sha
}  // namespace da