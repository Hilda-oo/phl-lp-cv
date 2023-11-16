//
// Created by cflin on 4/20/23.
//

#ifndef TOP3D_UTIL_H
#define TOP3D_UTIL_H

#include <igl/adjacency_list.h>
#include <igl/avg_edge_length.h>
#include <igl/per_vertex_normals.h>
#include <igl/voxel_grid.h>
#include <igl/writeOBJ.h>
#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include "TensorWrapper.h"
#include "sha-base-framework/declarations.h"
#include "sha-base-framework/frame.h"
#include "sha-io-foundation/mesh_io.h"
#include "sha-simulation-utils/io_utils.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-volume-mesh/mesh.h"
namespace da::sha {
namespace top {
using Tensor3d = TensorWrapper<double, 3>;
using Tensor3i = TensorWrapper<int, 3>;
using Eigen::all;
using SpMat = Eigen::SparseMatrix<double>;

template <typename Scalar>
inline std::vector<Eigen::Triplet<Scalar>> Vec2Triplet(const Eigen::VectorXi &I,
                                                       const Eigen::VectorXi &J,
                                                       const Eigen::Matrix<Scalar, -1, 1> &V) {
  std::vector<Eigen::Triplet<Scalar>> v_tri;
  for (int i = 0; i < I.size(); ++i) {
    v_tri.push_back({I(i), J(i), V(i)});
  }
  return v_tri;
}

inline void write_tensor3d(const std::string &save_path, const Tensor3d &t3,
                           const Eigen::Vector3d &bd_min, const Eigen::Vector3d &bd_max) {
  std::ofstream of(save_path);
  of << bd_min.x() << '\t' << bd_min.y() << '\t' << bd_min.z() << std::endl;
  of << bd_max.x() << '\t' << bd_max.y() << '\t' << bd_max.z() << std::endl;
  of << t3.dimension(0) << '\t' << t3.dimension(1) << '\t' << t3.dimension(2) << std::endl;
  of << t3.size() << std::endl;
  for (int k = 0; k < t3.dimension(2); ++k) {
    for (int j = 0; j < t3.dimension(1); ++j) {
      for (int i = 0; i < t3.dimension(0); ++i) {
        of << t3(i, j, k) << '\n';
      }
    }
  }
  of.close();
  spdlog::info("write tensor to: " + save_path);
}
inline void WriteVectorXd(const std::string &save_path, const Eigen::VectorXd &vxd) {
  std::ofstream of(save_path);
  for (int i = 0; i < vxd.size(); ++i) {
    of << vxd(i) << std::endl;
  }
  of.close();
  spdlog::info("write  VectorXd to: " + save_path);
}

inline Tensor3d read_tensor3d(const std::string &read_path) {
  std::ifstream iif(read_path);
  int lx, ly, lz;
  iif >> lx >> ly >> lz;
  Tensor3d t3(lx, ly, lz);
  for (int k = 0; k < t3.dimension(2); ++k) {
    for (int j = 0; j < t3.dimension(1); ++j) {
      for (int i = 0; i < t3.dimension(0); ++i) {
        iif >> t3(i, j, k);
      }
    }
  }
  return t3;
}
class Mesh;
void WriteTensorToVtk(const fs_path &file_path, const Tensor3d &t3, std::shared_ptr<Mesh> sp_mesh);

inline void clear_dir(const std::filesystem::path &path) {
  for (const auto &entry : std::filesystem::recursive_directory_iterator(path)) {
    if (entry.is_regular_file()) {
      std::filesystem::remove(entry.path());
    }
  }
  std::string the_path = path;
  spdlog::info("clear dir : {}", the_path);
}

inline Eigen::VectorXi SetDifference(const Eigen::VectorXi &ordered_a,
                                     const Eigen::VectorXi &ordered_b) {
  Eigen::VectorXi ret(ordered_a.size());
  auto it = std::set_difference(ordered_a.data(), ordered_a.data() + ordered_a.size(),
                                ordered_b.data(), ordered_b.data() + ordered_b.size(), ret.data());
  ret.conservativeResize(std::distance(ret.data(), it));
  return ret;
}
inline Eigen::VectorXd FillDensityByTensor(const MatMesh3 &top_space_obj_mesh,
                                           const Tensor3d &rho_field) {
  Eigen::VectorXd rho_each_vertex = rho_field(top_space_obj_mesh.mat_coordinates.cast<int>());
  return rho_each_vertex;
}
class Top3d;
class Boundary;
std::pair<MatMesh3, Eigen::VectorXd> GetMeshVertexPropty(const fs_path &fs_obj,
                                                    const Tensor3d &ten_rho_field, Boundary &r_bdr,bool offset_flg=false);

// }

}  // namespace top
}  // namespace da::sha

#endif  // TOP3D_UTIL_H
