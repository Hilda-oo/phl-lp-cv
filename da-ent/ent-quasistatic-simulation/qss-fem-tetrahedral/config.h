#pragma once

#include <Eigen/Eigen>
#include <string>
#include <utility>

#include "sha-simulation-utils/boundary_conditions.h"

namespace da {

class Config {
 public:
  std::string filePath;
  std::string outputPath;

  double YM      = 1e5;
  double PR      = 0.3;
  double density = 1e3;

  std::string mshFilePath;
  std::vector<sha::DirichletBC> DirichletBCs;
  std::vector<sha::NeumannBC> NeumannBCs;

 public:
  bool loadFromJSON(const std::string& p_filePath);
  void backUpConfig(const std::string& p_filePath);
};

}  // namespace da
