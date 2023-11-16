#pragma once

#include <fstream>
#include <Eigen/Eigen>

namespace da::sha {

template<typename T>
void WriteMatrix(const std::string &path, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &M){
  std::ofstream ofs(path);
  for (int i = 0; i < M.rows(); ++i) {
    for (int j = 0; j < M.cols(); ++j) {
      ofs << M(i, j) << " ";
    }
    ofs << std::endl;
  }
  ofs.close();
}

template<typename T>
void ReadMatrix(const std::string &path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &M){
  std::vector<std::vector<T>> data;
  std::ifstream ifs(path);
  std::string line;
  int row = 0;
  int col = 0;
  while(std::getline(ifs, line)){
    std::stringstream ss(line);
    std::vector<T> row_data;
    col = 0;
    T v;
    while(ss >> v){
      col++;
      row_data.push_back(v);
    }
    data.push_back(row_data);
    row++;
  }

  M.resize(row, col);
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      M(i, j) = data[i][j];
    }
  }
  ifs.close();
}

void WriteTriVTK(const std::string &path, const Eigen::MatrixXd &V, const Eigen::MatrixXi &T,
                        const std::vector<double> &cell_data = {}, const std::vector<double> &v_data = {});

void WriteTetVTK(const std::string &path, const Eigen::MatrixXd &V, const Eigen::MatrixXi &T,
                        const std::vector<double> &cell_data = {}, const std::vector<double> &v_data = {});

void WritePntVTK(const std::string &path, const Eigen::MatrixXd &V);

// find (vertex indices of) surface triangle from tetrahedral
void FindSurfTriForTet(const Eigen::MatrixXi& TT, Eigen::MatrixXi& SF);

// read TV, TT, SF from .msh file
bool ReadTetMesh(const std::string& filePath, Eigen::MatrixXd& TV, Eigen::MatrixXi& TT,
                        Eigen::MatrixXi& SF);

// write .obj file
void WriteOBJ(const std::string &path, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);

}  // namespace da::sha
