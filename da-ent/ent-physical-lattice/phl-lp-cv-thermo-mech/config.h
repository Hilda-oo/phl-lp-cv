#pragma once

#include "sha-fem-thermoelastic/ThermoelasticWrapper.h"
#include <string>
#include <vector>

namespace da {

class Config {

 public:
  std::string filePath;
  std::string outputPath;
  std::string meshFilePath;
  std::string seedsPath;
  std::string backgroundCellsPath;
  std::string cellTetsPath;
  std::string cellPolyhedronPath;

  std::string condition;

  double YM      = 1e5;
  double PR      = 0.3;
  double TC;  //thermal conductivity
  double TEC; //thermal expansion coefficient

  double radius[3];
  double shell;
  int cellNum;
  
  std::shared_ptr<sha::CtrlPara> para = std::make_shared<sha::CtrlPara>();

  std::vector<sha::DirichletBC> thermalDirichletBCs;
  std::vector<sha::NeumannBC> thermalNeumannBCs;
  std::vector<sha::DirichletBC> mechanicalDirichletBCs;
  std::vector<sha::NeumannBC> mechanicalNeumannBCs;

 public:
  bool loadFromJSON(const std::string& p_filePath);
  void backUpConfig(const std::string& p_filePath);
};

}  // namespace da
