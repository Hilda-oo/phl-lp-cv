#pragma once

#include <Eigen/Eigen>
#include <string>
#include <utility>
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Geometry/AlignedBox.h"


namespace da {

struct Neu{
  Eigen::AlignedBox3d box;
  Eigen::Vector3d val;   
  Neu(const Eigen::Vector3d& min,const Eigen::Vector3d& max,const Eigen::Vector3d& val)
  :box(min,max),val(val){}
};
struct Dir{
  Eigen::AlignedBox3d box;
  Eigen::Vector3i dir;
  Dir(const Eigen::Vector3d& min,const Eigen::Vector3d& max,const Eigen::Vector3i& dir)
  :box(min,max),dir(dir){}
};

class Config {
 public:
  std::string filePath;
  std::string outputPath;
  std::string visualObjPath;

  double YM      = 1e5;
  double PR      = 0.3;

  double max_loop =200;
  double volfrac=0.15;
  double penal=1.0;
  double r_min=1.5;

  int len_x;
  int len_y;
  int len_z;

  std::vector<Neu> v_Neu;
  std::vector<Dir> v_Dir;


 public:
  bool loadFromJSON(const std::string& p_filePath);
  void backUpConfig(const std::string& p_filePath);
};

}  // namespace da
