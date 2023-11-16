#pragma once

#include <Eigen/Core>

/**
 * Created by kwp
 */

namespace cage {
struct Mesh {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
};

struct LevelInfo {
  int num_faces;
  bool adaptive = true;
};

enum class EnergyType { None, DispStep, DispInitial, Volume, SurfARAP, VolARAP };

auto nested_cages(const Mesh &mesh0, int quad_order, const std::vector<LevelInfo> &level_info,
                  EnergyType energy_inflation = EnergyType::None,
                  EnergyType energy_final     = EnergyType::Volume) -> std::vector<Mesh>;
}  // namespace cage