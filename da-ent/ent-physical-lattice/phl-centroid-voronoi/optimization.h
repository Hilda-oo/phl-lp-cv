#pragma once

#include <Eigen/Eigen>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "modeling.h"

#include "sha-simulation-3d/CBN.h"
#include "sha-surface-mesh/matmesh.h"

namespace da {
class CVTOptimizer {
 public:
  explicit CVTOptimizer(const Eigen::AlignedBox3d &design_domain,
                          const std::shared_ptr<ModelAlgorithm> &modeling,
                          const std::shared_ptr<sha::CBNSimulator> &simulation,
                          double init_radius,
                          const std::pair<double, double> radius_range, const double scalar_E,
                          const double volfrac, const Eigen::MatrixXd &mat_init_seeds);

  void EvaluateFiniteDifference(
      std::vector<std::vector<Eigen::VectorXd>> &dH,
      std::vector<Eigen::VectorXi> &mat_varibale_to_effected_macro_indices);

  // Compliance
  double EvaluateEnergyC(const Eigen::VectorXd &displacements,
                         const std::vector<Eigen::VectorXd> &rhos,
                         const std::vector<std::vector<Eigen::VectorXd>> &dH,
                         const std::vector<Eigen::VectorXi> &mat_variable_to_effected_macro_indices,
                         Eigen::VectorXd &dC) const;
  // Volume Frac
  double EvaluateVolume(const std::vector<Eigen::VectorXd> &rhos,
                        const std::vector<std::vector<Eigen::VectorXd>> &dH,
                        const std::vector<Eigen::VectorXi> &mat_variable_to_effected_macro_indices,
                        Eigen::VectorXd &dV) const;
  // CVT energy
  double EvaluateEnergyE(Eigen::VectorXd &dE);

  // Angle energy
  double EvaluateEnergyR(Eigen::VectorXd &dR);

  using IterationFunctionType =
      std::function<void(index_t iteration_idx, const Eigen::MatrixXd &mat_variables, double C,
                         const Eigen::VectorXd &dC, double E, const Eigen::VectorXd &dE, double V,
                         const Eigen::VectorXd &dV)>;

  void Optimize(Eigen::MatrixXd &mat_seeds_result, Eigen::VectorXd &radiuses_result,
                size_t num_iterations, IterationFunctionType IterationFunction = nullptr);

  auto GetModeling() { return modeling_; }
  auto GetSimulation() { return simulation_; }

 protected:
  Eigen::AlignedBox3d design_domain_;
  std::shared_ptr<ModelAlgorithm> modeling_;
  std::shared_ptr<sha::CBNSimulator> simulation_;

  Eigen::MatrixXd mat_variables_;
  std::pair<Eigen::VectorXd, Eigen::VectorXd> variables_bounds_;
  const double scalar_E_;
  const double volfrac_;

 protected:
  const size_t num_seeds_;
  const size_t num_cols_;
  const size_t num_variables_;

 private:
  double finite_difference_step_;
  Eigen::Vector3d finding_vision_;
};
}  // namespace da