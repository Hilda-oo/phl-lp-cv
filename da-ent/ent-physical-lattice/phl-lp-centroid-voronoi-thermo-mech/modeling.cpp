#include "modeling.h"

#include "Eigen/src/Core/Matrix.h"
#include "sha-base-framework/declarations.h"
#include "utilis.h"

#include <igl/signed_distance.h>
#include "sha-io-foundation/mesh_io.h"
#include "sha-voronoi-foundation/fast_voronoi.h"

#include <igl/centroid.h>
#include <oneapi/tbb.h>

namespace da {

ModelAlgorithm::ModelAlgorithm(const MatMesh3 &mesh,
                               const sha::NestedBackgroundMesh &nested_background_mesh,
                               bool use_shell, double shell_thickness)
    : mesh_(mesh),  //fine mesh
      background_mesh_(nested_background_mesh),
      use_shell_(use_shell),
      shell_thickness_(shell_thickness) {
  sharp_angle_ = 11;

  signed_distances_of_micro_vertices_.clear();
  signed_distances_of_micro_vertices_.resize(nested_background_mesh.nested_cells_.size());
  for (index_t cell_idx = 0; cell_idx < nested_background_mesh.nested_cells_.size(); ++cell_idx) {
    Eigen::MatrixXi I;
    Eigen::MatrixXd C;
    Eigen::MatrixXd N;
    igl::signed_distance(
        nested_background_mesh.nested_cells_.at(cell_idx).tetrahedrons.mat_coordinates,
        mesh.mat_coordinates, mesh.mat_faces, igl::SIGNED_DISTANCE_TYPE_DEFAULT,
        signed_distances_of_micro_vertices_.at(cell_idx), I, C, N);
  }
  eps = nested_background_mesh.ComputeMinMicroTetEdgeLength();  // micro cell tet min edge length
}

void ModelAlgorithm::Update(const Eigen::MatrixXd &mat_variables) {
  map_voronoi_beam_idx_to_cell_indices.clear();

  current_seeds_    = mat_variables.leftCols(3);
  current_radiuses_ = mat_variables.col(3);
  // voronoi_diagram_ =
  //     CreateRestrictedVoronoiDiagramFromMesh(mesh_, current_seeds_, 0, sharp_angle_, true);
  voronoi_diagram_ =
      FastCreateRestrictedVoronoiDiagramFromMesh(mesh_, current_seeds_, sharp_angle_);

  //output voronoi cell
  // for (index_t cellIdx = 0; cellIdx < voronoi_diagram_.cells.size(); cellIdx++) {
  //   sha::WriteMatMesh3ToVtk((WorkingResultDirectoryPath() / "voronoiCell" /
  //                            fmt::format("voronoi_triangle_mesh{}.vtk", cellIdx))
  //                               .string(),
  //                           voronoi_diagram_.cells.at(cellIdx).cell_triangle_mesh);
  // }
  voronoi_beams_mesh_ = ComputeRelatedEdgesFromVoronoiDiagram(voronoi_diagram_, mesh_.AlignedBox(),
                                                              map_voronoi_beam_idx_to_cell_indices);
  // std::map<int, int> sta;
  // for (auto &cell : map_beam_idx_to_cell_indices) {
  //   sta[cell.size()] += 1;
  // }
  // for (auto [k, v] : sta) {
  //   std::cout << k << ": " << v << std::endl;
  // }
  // exit(5);
  voronoi_beams_radiuses_.resize(voronoi_beams_mesh_.NumBeams());

  for (index_t beam_idx = 0; beam_idx < voronoi_beams_mesh_.NumBeams(); ++beam_idx) {
    double beam_radius = boost::accumulate(map_voronoi_beam_idx_to_cell_indices.at(beam_idx), 0.0,
                                           [&](double total_radius, index_t cell_idx) {
                                             return total_radius + current_radiuses_(cell_idx);
                                           });
    voronoi_beams_radiuses_(beam_idx) =
        beam_radius / map_voronoi_beam_idx_to_cell_indices.at(beam_idx).size();
  }
  log::info("voronoi_beams_mesh_.beams = {}", voronoi_beams_mesh_.NumBeams());
}
/*
void ModelAlgorithm::Update(const Eigen::MatrixXd &mat_variables) {
  map_voronoi_beam_idx_to_cell_indices.clear();

  current_seeds_    = mat_variables.leftCols(3);
  current_radiuses_ = mat_variables.col(3);
  // voronoi_diagram_ =
  //     CreateRestrictedVoronoiDiagramFromMesh(mesh_, current_seeds_, 0, sharp_angle_, true);
  voronoi_diagram_ =
      FastCreateRestrictedVoronoiDiagramFromMesh(mesh_, current_seeds_, sharp_angle_);

  //output voronoi cell
  // for (index_t cellIdx = 0; cellIdx < voronoi_diagram_.cells.size(); cellIdx++) {
  //   sha::WriteMatMesh3ToVtk((WorkingResultDirectoryPath() / "voronoiCell" /
  //                            fmt::format("voronoi_triangle_mesh{}.vtk", cellIdx))
  //                               .string(),
  //                           voronoi_diagram_.cells.at(cellIdx).cell_triangle_mesh);
  // }
  voronoi_beams_mesh_ = ComputeRelatedEdgesFromVoronoiDiagram(voronoi_diagram_, mesh_.AlignedBox(),
                                                              map_voronoi_beam_idx_to_cell_indices);
  // std::map<int, int> sta;
  // for (auto &cell : map_beam_idx_to_cell_indices) {
  //   sta[cell.size()] += 1;
  // }
  // for (auto [k, v] : sta) {
  //   std::cout << k << ": " << v << std::endl;
  // }
  // exit(5);
  voronoi_beams_radiuses_.resize(voronoi_beams_mesh_.NumBeams());

  for (index_t beam_idx = 0; beam_idx < voronoi_beams_mesh_.NumBeams(); ++beam_idx) {
    // double beam_radius = boost::accumulate(map_voronoi_beam_idx_to_cell_indices.at(beam_idx), 0.0,
    //                                        [&](double total_radius, index_t cell_idx) {
    //                                          return total_radius + current_radiuses_(cell_idx);
    //                                        });

    int lx = 100, ly = 30, lz = 30;
    int num_property = top_density_.size();

    auto getIndexFromCoord = [](double x, int min, int max) -> int {
      int new_x = std::round(x + (max - min) / 2.0);
      // x = std::trunc(x + (max - min) / 2);
      // x = ceil(x + (max - min) / 2);
      // x = floor(x + (max - min) / 2);
      return new_x;
    };

    auto getVerticeDensity = [&](Eigen::Vector3d v) -> double {
      int x              = getIndexFromCoord(v[0], 0, lx);
      int y              = getIndexFromCoord(v[1], 0, ly);
      int z              = getIndexFromCoord(v[2], 0, lz);
      int index_property = z * lx * ly + y * lx + x;
      double value = 0.0;
      if (index_property >= 0 && index_property <= num_property) {
        value = top_density_.at(index_property);
      }
      return value;
    };
  
    double beam_radius = boost::accumulate(map_voronoi_beam_idx_to_cell_indices.at(beam_idx), 0.0,
                                           [&](double total_radius, index_t cell_idx) {
                                            auto &tet = background_mesh_.nested_cells_[cell_idx].tetrahedrons;
                                            auto &TV  = tet.mat_coordinates;
                                            for (int vi = 0; vi < 4; vi++) {
                                              total_radius += getVerticeDensity(TV.row(vi));
                                            }
                                            double result = total_radius / 4.0;
                                            current_radiuses_(cell_idx) = result;
                                            return result;
                                           });
    voronoi_beams_radiuses_(beam_idx) =
        beam_radius / map_voronoi_beam_idx_to_cell_indices.at(beam_idx).size();
  }
  double min_density = voronoi_beams_radiuses_.minCoeff();
  double max_density = voronoi_beams_radiuses_.maxCoeff();
  double mean_density = (max_density - min_density) / (voronoi_beams_mesh_.NumBeams() * 1.0);
  double mean_radius = (max_radius_ - min_radius_) / voronoi_beams_mesh_.NumBeams();
  for (index_t beam_idx = 0; beam_idx < voronoi_beams_mesh_.NumBeams(); ++beam_idx) {
    voronoi_beams_radiuses_(beam_idx) = min_radius_ + (voronoi_beams_radiuses_(beam_idx) - min_density) / mean_density * mean_radius;
  }
  log::info("voronoi_beams_mesh_.beams = {}", voronoi_beams_mesh_.NumBeams());
}*/

double ModelAlgorithm::ComputeMinDistance(const Eigen::Vector3d &x) {
  auto phij = [](const Eigen::Vector3d &a, const Eigen::Vector3d &b, double radius,
                 const Eigen::Vector3d &x) -> double {
    Eigen::Vector3d ab = b - a;
    Eigen::Vector3d ax = x - a;
    if (ax.dot(ab) <= 0.0) return ax.norm();
    Eigen::Vector3d bx = x - b;
    if (bx.dot(ab) >= 0.0) return bx.norm();
    return (ab.cross(ax)).norm() / ab.norm() - radius;
  };
  const int num_beams = voronoi_beams_mesh_.NumBeams();
  std::vector<double> phis(voronoi_beams_mesh_.NumBeams());
  oneapi::tbb::parallel_for(0, num_beams, 1, [&](int beam_idx) {
    phis.at(beam_idx) =
        -phij(voronoi_beams_mesh_.mat_coordinates.row(voronoi_beams_mesh_.mat_beams(beam_idx, 0)),
              voronoi_beams_mesh_.mat_coordinates.row(voronoi_beams_mesh_.mat_beams(beam_idx, 1)),
              voronoi_beams_radiuses_(beam_idx), x);
  });
  return *std::max_element(phis.begin(), phis.end());
}

auto ModelAlgorithm::ComputeRhos() -> std::vector<Eigen::VectorXd> {
  const size_t num_macro_cells = background_mesh_.nested_cells_.size();
  std::vector<Eigen::VectorXd> rhos(num_macro_cells);

  for (index_t cell_idx = 0; cell_idx < num_macro_cells; ++cell_idx) {
    auto &tet = background_mesh_.nested_cells_[cell_idx].tetrahedrons;
    auto &TV  = tet.mat_coordinates;
    auto &TT  = tet.mat_tetrahedrons;
    Eigen::VectorXd tet_rho;
    tet_rho.resize(TT.rows());

    auto S = signed_distances_of_micro_vertices_[cell_idx];  // distance of each macro-tet-vertex to
                                                             // closest micro-vertex
    if (use_shell_) {
      for (index_t idx = 0; idx < S.rows(); ++idx) {
        if (S(idx) > 0) {
          S(idx) = Heaviside(0, eps);  // out of shell, return H(0), most complicated
        } else {
          double dis_to_shell = std::abs(S(idx));
          if (dis_to_shell <= shell_thickness_) {  // inside shell,return 1
            S(idx) = Heaviside(1, eps);
          } else {
            double dis_to_rod = ComputeMinDistance(TV.row(idx));
            S(idx)            = Heaviside(std::max(-dis_to_shell, dis_to_rod), eps);
          }
        }
      }
    } else {
      for (index_t idx = 0; idx < S.rows(); ++idx) {
        if (S(idx) > 0) {
          S(idx) = Heaviside(0, eps);
        } else {
          S(idx) = Heaviside(ComputeMinDistance(TV.row(idx)), eps);
        }
      }
    }

    for (index_t idx = 0; idx < TT.rows(); ++idx) {
      double value = (S(TT(idx, 0)) + S(TT(idx, 1)) + S(TT(idx, 2)) + S(TT(idx, 3))) / 4.0;
      tet_rho(idx) = value;
    }
    rhos[cell_idx] = tet_rho;
  }
  return rhos;
}

auto ModelAlgorithm::ComputeRhos(const Eigen::MatrixXd &P) -> Eigen::VectorXd {
  Eigen::VectorXd S;
  Eigen::MatrixXi I;
  Eigen::MatrixXd C;
  Eigen::MatrixXd N;
  igl::signed_distance(P, mesh_.mat_coordinates, mesh_.mat_faces, igl::SIGNED_DISTANCE_TYPE_DEFAULT,
                       S, I, C, N);

  Eigen::VectorXd rhos(P.rows());
  if (use_shell_) {
    for (int i = 0; i < S.rows(); ++i) {
      if (S(i) > 0) {
        rhos(i) = Heaviside(0, eps);
      } else {
        double dis_to_shell = abs(S(i));
        if (dis_to_shell <= shell_thickness_) {
          rhos(i) = Heaviside(1, eps);
        } else {
          double dis_to_rod = ComputeMinDistance(P.row(i));
          rhos(i)           = Heaviside(std::max(-dis_to_shell, dis_to_rod), eps);
        }
      }
    }
  } else {
    for (int i = 0; i < S.rows(); ++i) {
      if (S(i) > 0) {
        rhos(i) = Heaviside(0, eps);
      } else {
        rhos(i) = Heaviside(ComputeMinDistance(P.row(i)), eps);
      }
    }
  }
  return rhos;
}

void ModelAlgorithm::UpdateFD(const Eigen::MatrixXd &mat_variables) {
  this->current_seeds_    = mat_variables.leftCols<3>();
  this->current_radiuses_ = mat_variables.col(3);

  const size_t num_macro_cells = background_mesh_.nested_cells_.size();
  if (INIT_sorted_v_to_seeds_to_seeds_id.empty()) {
    Eigen::VectorXi linspace(current_seeds_.rows());
    linspace.setLinSpaced(0, current_seeds_.rows());
    INIT_sorted_v_to_seeds_to_seeds_id.resize(num_macro_cells);
    for (index_t cid = 0; cid < num_macro_cells; ++cid) {
      INIT_sorted_v_to_seeds_to_seeds_id[cid].resize(
          background_mesh_.nested_cells_.at(cid).tetrahedrons.NumVertices(), current_seeds_.rows());
      INIT_sorted_v_to_seeds_to_seeds_id[cid].rowwise() = linspace.transpose();
    }
  }
  seeds_5_distance_temp.resize(num_macro_cells);
}

auto ModelAlgorithm::ComputeRhosFD() -> std::vector<Eigen::VectorXd> {
  const size_t num_macro_cells = background_mesh_.nested_cells_.size();
  std::vector<Eigen::VectorXd> rhos(num_macro_cells);

#pragma omp parallel for
  for (index_t cell_idx = 0; cell_idx < num_macro_cells; ++cell_idx) {
    rhos[cell_idx] = ComputeRhosFDInSingleCell(cell_idx);
  }
  current_rhos = rhos;
  return rhos;
}

auto ModelAlgorithm::ComputeRhosFD(Eigen::VectorXi &flag, const int seed_id,
                                   const Eigen::Vector3d &seed_position, double radius,
                                   const Eigen::Vector3d &search_range)
    -> std::vector<Eigen::VectorXd> {
  const size_t num_macro_cells = background_mesh_.nested_cells_.size();

  Eigen::MatrixXd tmp_seeds         = current_seeds_;
  Eigen::VectorXd tmp_radiuses      = current_radiuses_;
  std::vector<Eigen::VectorXd> rhos = current_rhos;
  flag.setZero(rhos.size());
  Eigen::AlignedBox3d seed_box(seed_position - search_range, seed_position + search_range);
  current_seeds_.row(seed_id) = seed_position.transpose();
  current_radiuses_(seed_id)  = radius;

#pragma omp parallel for
  for (index_t cell_idx = 0; cell_idx < num_macro_cells; ++cell_idx) {
    if (background_mesh_.nested_cells_.at(cell_idx).cell_box.intersects(seed_box)) {
      rhos[cell_idx] = ComputeRhosFDInSingleCell(cell_idx, seed_id);
      flag(cell_idx) = 1;
    } else {
      flag(cell_idx) = 0;
    }
  }
  // restore
  current_seeds_    = tmp_seeds;
  current_radiuses_ = tmp_radiuses;
  return rhos;
}

auto ModelAlgorithm::ComputeRhosFDInSingleCell(int macro_idx, int updated_seed_id)
    -> Eigen::VectorXd {
  // step1. 逐顶点计算到种子点的距离
  const auto &TV = background_mesh_.nested_cells_[macro_idx].tetrahedrons.mat_coordinates;
  const auto &TT = background_mesh_.nested_cells_[macro_idx].tetrahedrons.mat_tetrahedrons;

  Eigen::VectorXd distV(TV.rows());

  Eigen::MatrixXd v_to_seeds_one_iter;  // 四面体顶点 P 到种子点的距离 <顶点数量，种子点数量>
  {
    if (updated_seed_id == -1) {
      v_to_seeds_one_iter.resize(TV.rows(), current_seeds_.rows());
      for (int i = 0; i < TV.rows(); i++) {
        for (int j = 0; j < current_seeds_.rows(); ++j) {
          v_to_seeds_one_iter(i, j) = (TV.row(i) - current_seeds_.row(j)).norm();
        }
      }
    } else {
      v_to_seeds_one_iter = this->seeds_5_distance_temp[macro_idx].first;
      for (int i = 0; i < TV.rows(); i++) {
        v_to_seeds_one_iter(i, 4) = (TV.row(i) - current_seeds_.row(updated_seed_id)).norm();
      }
    }
  }

  // step2. 逐顶点计算:
  // step2.1. 在一定范围里面找到距离最近的几个点 set<Vector3d> seeds
  //           TODO: 先计算最近的试试
  Eigen::MatrixXi nearest_seeds_id;  // 距离顶点 P 最近的三个种子点索引
  {
#if 0
        Eigen::MatrixXd sorted_v_to_seeds;
        Eigen::MatrixXi sorted_v_to_seeds_to_seeds_id;
        igl::sort(v_to_seeds_one_iter, 2, true, sorted_v_to_seeds, sorted_v_to_seeds_to_seeds_id);
        nearest_seeds_id = sorted_v_to_seeds_to_seeds_id.leftCols<3>();
#endif
    if (updated_seed_id == -1) {
      Eigen::MatrixXi sorted_v_to_seeds_to_seeds_id;
      sorted_v_to_seeds_to_seeds_id = INIT_sorted_v_to_seeds_to_seeds_id[macro_idx];
      for (int i = 0; i < TV.rows(); i++) {
        // 部分冒泡排序，不用top k或者堆排序避免多余运算
        for (int tt = 0; tt < 4; ++tt) {
          for (int j = current_seeds_.rows() - 1; j >= 1; --j) {
            if (v_to_seeds_one_iter(i, j) < v_to_seeds_one_iter(i, j - 1)) {
              std::swap(v_to_seeds_one_iter(i, j), v_to_seeds_one_iter(i, j - 1));
              std::swap(sorted_v_to_seeds_to_seeds_id(i, j),
                        sorted_v_to_seeds_to_seeds_id(i, j - 1));
            }
          }
        }
      }
      nearest_seeds_id = sorted_v_to_seeds_to_seeds_id.leftCols<3>();
      // 缓存前五列，前1列用于排序，后1列用于放值
      seeds_5_distance_temp[macro_idx].first  = v_to_seeds_one_iter.leftCols<5>();
      seeds_5_distance_temp[macro_idx].second = sorted_v_to_seeds_to_seeds_id.leftCols<5>();
    } else {
      Eigen::MatrixXi sorted_v_to_seeds_to_seeds_id = seeds_5_distance_temp[macro_idx].second;
      sorted_v_to_seeds_to_seeds_id.col(4) =
          Eigen::VectorXi::Constant(sorted_v_to_seeds_to_seeds_id.rows(), updated_seed_id);
      for (int i = 0; i < TV.rows(); i++) {
        int should_loc_id = 3;
        for (int j = 0; j < 3; ++j) {
          if (updated_seed_id == sorted_v_to_seeds_to_seeds_id(i, j)) {
            should_loc_id = j;
            break;
          }
        }
        // should_loc_id == 3   不是前三个移动得到的
        sorted_v_to_seeds_to_seeds_id(i, should_loc_id) = updated_seed_id;
        v_to_seeds_one_iter(i, should_loc_id)           = v_to_seeds_one_iter(i, 4);
        for (int tt = 0; tt < 3; ++tt) {
          for (int j = 3; j >= 1; --j) {
            if (v_to_seeds_one_iter(i, j) < v_to_seeds_one_iter(i, j - 1)) {
              std::swap(v_to_seeds_one_iter(i, j), v_to_seeds_one_iter(i, j - 1));
              std::swap(sorted_v_to_seeds_to_seeds_id(i, j),
                        sorted_v_to_seeds_to_seeds_id(i, j - 1));
            }
          }
        }
      }
      nearest_seeds_id = sorted_v_to_seeds_to_seeds_id.leftCols<3>();
    }
  }

  // step2.2. seeds 中每三个种子点对计算外心，并且计算种子点对的平均半径
  Eigen::MatrixXd seeds_circumcenters;
  Eigen::VectorXd average_radius;
  {
    seeds_circumcenters.resize(nearest_seeds_id.rows(), 3);
    average_radius.resize(nearest_seeds_id.rows());
    for (int i = 0; i < nearest_seeds_id.rows(); i++) {
      seeds_circumcenters.row(i) = ComputeTriangularCircumcenter(
          current_seeds_.row(nearest_seeds_id(i, 0)), current_seeds_.row(nearest_seeds_id(i, 1)),
          current_seeds_.row(nearest_seeds_id(i, 2)));
      average_radius(i) =
          (current_radiuses_(nearest_seeds_id(i, 0)) + current_radiuses_(nearest_seeds_id(i, 1)) +
           current_radiuses_(nearest_seeds_id(i, 2))) /
          3;
    }
  }

  // step2.3. 计算三角形构成的平面投影上顶点到外心的距离
  {
    for (int i = 0; i < TV.rows(); i++) {
      Eigen::Vector3d e1 =
          current_seeds_.row(nearest_seeds_id(i, 1)) - current_seeds_.row(nearest_seeds_id(i, 0));
      Eigen::Vector3d e2 =
          current_seeds_.row(nearest_seeds_id(i, 2)) - current_seeds_.row(nearest_seeds_id(i, 0));
      Eigen::Vector3d n = e1.cross(e2).normalized();
      Eigen::Vector3d v = TV.row(i) - seeds_circumcenters.row(i);
      distV(i)          = std::abs(v.cross(n).norm()) - average_radius(i);
    }
  }

  // step2.4. 取最小的距离作为 dis
  auto S = signed_distances_of_micro_vertices_[macro_idx];
  if (use_shell_) {
    for (int i = 0; i < TV.rows(); ++i) {
      if (S(i) > 0) {
        S(i) = Heaviside(0, eps);
      } else {
        double dis_to_shell = abs(S(i));
        if (dis_to_shell <= shell_thickness_) {
          S(i) = Heaviside(1, eps);
        } else {
          double dis_to_rod = -distV(i);
          S(i)              = Heaviside(std::max(-dis_to_shell, dis_to_rod), eps);
        }
      }
    }
  } else {
    for (int i = 0; i < TV.rows(); ++i) {
      if (S(i) > 0) {
        S(i) = Heaviside(0, eps);
      } else {
        S(i) = Heaviside(-distV(i), eps);
      }
    }
  }

  Eigen::VectorXd tet_rho;
  tet_rho.resize(TT.rows());
  for (int i = 0; i < TT.rows(); ++i) {
    double v   = (S(TT(i, 0)) + S(TT(i, 1)) + S(TT(i, 2)) + S(TT(i, 3))) / 4.0;
    tet_rho(i) = v;
  }
  return tet_rho;
}

void ModelAlgorithm::ComputeCellCentroids(Eigen::MatrixXd &mat_centroids,
                                          Eigen::VectorXi &flag) const {
  const size_t num_cells = voronoi_diagram_.cells.size();
  mat_centroids.resize(num_cells, 3);
  flag.resize(num_cells);

  oneapi::tbb::parallel_for(0, static_cast<int>(num_cells), 1, [&](int cell_idx) {
    auto &cell = voronoi_diagram_.cells.at(cell_idx);
    if (cell.cell_triangle_mesh.NumFaces() == 0) {
      flag(cell_idx) = 0;
    } else {
      flag(cell_idx) = 1;
      Eigen::Vector3d centroid;
      igl::centroid(cell.cell_triangle_mesh.mat_coordinates, cell.cell_triangle_mesh.mat_faces,
                    centroid);
      mat_centroids.row(cell_idx) = centroid;
    }
  });
}
}  // namespace da