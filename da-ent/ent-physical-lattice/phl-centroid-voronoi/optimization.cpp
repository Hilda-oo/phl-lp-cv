#include "optimization.h"

#include <fmt/format.h>
#include <mma/MMASolver.h>

#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include <oneapi/tbb.h>

namespace da {

CVTOptimizer::CVTOptimizer(const Eigen::AlignedBox3d &design_domain,
                           const std::shared_ptr<ModelAlgorithm> &modeling,
                           const std::shared_ptr<sha::CBNSimulator> &simulation, double init_radius,
                           const std::pair<double, double> radius_range, double scalar_E,
                           double volfrac, const Eigen::MatrixXd &mat_init_seeds)
    : design_domain_(design_domain),
      modeling_(modeling),
      simulation_(simulation),
      scalar_E_(scalar_E),
      volfrac_(volfrac),
      num_seeds_(mat_init_seeds.rows()),
      num_cols_(4),
      num_variables_(mat_init_seeds.rows() * 4) {
  // init variables
  mat_variables_ = mat_init_seeds;
  mat_variables_.conservativeResize(num_seeds_, num_cols_);
  mat_variables_.col(num_cols_ - 1).setConstant(init_radius);

  // init bounds
  variables_bounds_.first.resize(num_variables_);
  variables_bounds_.second.resize(num_variables_);
  variables_bounds_.first(Eigen::seq(0, Eigen::last, num_cols_))
      .setConstant(design_domain_.min().x());
  variables_bounds_.first(Eigen::seq(1, Eigen::last, num_cols_))
      .setConstant(design_domain_.min().y());
  variables_bounds_.first(Eigen::seq(2, Eigen::last, num_cols_))
      .setConstant(design_domain_.min().z());
  variables_bounds_.first(Eigen::seq(3, Eigen::last, num_cols_)).setConstant(radius_range.first);

  variables_bounds_.second(Eigen::seq(0, Eigen::last, num_cols_))
      .setConstant(design_domain_.max().x());
  variables_bounds_.second(Eigen::seq(1, Eigen::last, num_cols_))
      .setConstant(design_domain_.max().y());
  variables_bounds_.second(Eigen::seq(2, Eigen::last, num_cols_))
      .setConstant(design_domain_.max().z());
  variables_bounds_.second(Eigen::seq(3, Eigen::last, num_cols_)).setConstant(radius_range.second);

  //
  finite_difference_step_ = simulation_->nested_background_->ComputeAverageMicroTetEdgeLength();
  log::info("finite_difference_step = {}", finite_difference_step_);

  //
  finding_vision_(0) = design_domain_.sizes().x() / std::pow(num_seeds_, 1.0 / 3) * 3.0 / 2.0;
  finding_vision_(1) = design_domain_.sizes().y() / std::pow(num_seeds_, 1.0 / 3) * 3.0 / 2.0;
  finding_vision_(2) = design_domain_.sizes().z() / std::pow(num_seeds_, 1.0 / 3) * 3.0 / 2.0;
  spdlog::debug("finding_vision: {:.3f} {:.3f} {:.3f}", finding_vision_(0), finding_vision_(1),
                finding_vision_(2));
}

void CVTOptimizer::Optimize(Eigen::MatrixXd &mat_seeds_result, Eigen::VectorXd &radiuses_result,
                            size_t num_iterations, IterationFunctionType IterationFunction) {
  // init
  const int dgt0 = 5;
  double objVr5  = 1.0;

  size_t num_constrants = 1;
  auto mma              = std::make_shared<MMASolver>(num_variables_, num_constrants);

  std::vector<double> OBJ;

  modeling_->Update(mat_variables_);

  log::info("Start iterations: {}", num_iterations);

  if (IterationFunction != nullptr) {
    IterationFunction(0, mat_variables_, 0, Eigen::VectorXd(), 0, Eigen::VectorXd(), 0,
                      Eigen::VectorXd());
  }

  // iterations
  for (int iteration_idx = 1; iteration_idx <= num_iterations; ++iteration_idx) {
    // modeling
    std::vector<Eigen::VectorXd> rhos = modeling_->ComputeRhos();

    // simulation
    Eigen::VectorXd displacements = simulation_->Simulate(rhos);

    // compute differential
    std::vector<std::vector<Eigen::VectorXd>> dH;
    std::vector<Eigen::VectorXi> mat_variable_to_effected_macro_indices;
    EvaluateFiniteDifference(dH, mat_variable_to_effected_macro_indices);

    double C, V, E;
    Eigen::VectorXd dC, dV, dE;

    C = EvaluateEnergyC(displacements, rhos, dH, mat_variable_to_effected_macro_indices, dC);
    V = EvaluateVolume(rhos, dH, mat_variable_to_effected_macro_indices, dV);
    E = EvaluateEnergyE(dE);

    log::info("mean: dC:{}, dV:{}, dE:{}", dC.mean(), dV.mean(), dE.mean());

    double dgt_c = dgt0 - static_cast<int>(log10(dC.cwiseAbs().maxCoeff()));
    double dgt_v = dgt0 - static_cast<int>(log10(dV.cwiseAbs().maxCoeff()));
    double dgt_e = dgt0 - static_cast<int>(log10(dE.cwiseAbs().maxCoeff()));
    double tmp_c = std::pow(10.0, dgt_c);
    double tmp_v = std::pow(10.0, dgt_v);
    double tmp_e = std::pow(10.0, dgt_e);
    dC           = (dC * tmp_c).array().round() / tmp_c;
    dV           = (dV * tmp_v).array().round() / tmp_v;
    dE           = (dE * tmp_e).array().round() / tmp_e;

    // normalization
    dC /= dC.maxCoeff();
    dV /= dV.maxCoeff();
    dE /= dE.maxCoeff();

    Eigen::VectorXd variables_tmp = mat_variables_.transpose().reshaped(num_variables_, 1);
    double f0val                  = C * (1.0 - scalar_E_) + E * scalar_E_;
    Eigen::VectorXd df0dx         = -dC * (1.0 - scalar_E_) + dE * scalar_E_;
    double fval                   = V / volfrac_ - 1;
    Eigen::VectorXd dfdx          = dV / volfrac_;

    spdlog::info("fval: {}, f0val: {}", fval, f0val);

    spdlog::info("mma update");
    mma->Update(variables_tmp.data(), df0dx.data(), &fval, dfdx.data(),
                variables_bounds_.first.data(), variables_bounds_.second.data());
    mat_variables_ = variables_tmp.reshaped(num_cols_, num_seeds_).transpose();

    // update modeling
    modeling_->Update(mat_variables_);  // update design variable

    // compute f0val
    OBJ.emplace_back(f0val);
    if (iteration_idx >= 5 && (V - volfrac_) / volfrac_ < 1.0e-3) {
      double mean_ = 0.0;
      for (int i_ = iteration_idx - 5; i_ < iteration_idx; ++i_) {
        mean_ += OBJ[i_];
      }
      mean_ /= 5.0;
      double max_ = 0.0;
      for (int i_ = iteration_idx - 5; i_ < iteration_idx; ++i_) {
        max_ = std::max(max_, abs(OBJ[i_] - mean_));
      }
      objVr5 = abs(max_ / mean_);
    }

    if (IterationFunction != nullptr) {
      IterationFunction(iteration_idx, mat_variables_, C, dC, E, dE, V, dV);
    }

    spdlog::critical("Optimization iter# {}, C = {:.6}, E = {:.6f}, V = {:.3f}, ch = {:.4f}",
                     iteration_idx, C, E, V, objVr5);
  }

  mat_seeds_result = mat_variables_.leftCols(3);
  radiuses_result  = mat_variables_.col(3);
}

void CVTOptimizer::EvaluateFiniteDifference(
    std::vector<std::vector<Eigen::VectorXd>> &dH,
    std::vector<Eigen::VectorXi> &mat_varibale_to_effected_macro_indices) {
  spdlog::info("compute dH");
  dH.resize(num_variables_);
  // boost::progress_display show_progress(nCell);
  double time_rho = 0.0;

  mat_varibale_to_effected_macro_indices.clear();
  mat_varibale_to_effected_macro_indices.resize(num_variables_);
  std::vector<Eigen::VectorXi> effected_flags(num_variables_);
  std::vector<int> mat_varibale_to_effected_macro_num(num_variables_);

  modeling_->UpdateFD(mat_variables_);
  modeling_->ComputeRhosFD();
  for (int xI = 0; xI < num_seeds_; ++xI) {
    for (int xJ = 0; xJ < num_cols_; ++xJ) {
      int varI = xI * num_cols_ + xJ;

      Eigen::Vector4d X_minus_row = mat_variables_.row(xI).transpose();
      X_minus_row(xJ) -= finite_difference_step_;
      std::vector<Eigen::VectorXd> rho_minus = modeling_->ComputeRhosFD(
          effected_flags[varI], xI, X_minus_row.head<3>(), X_minus_row(3), finding_vision_);
      Assert(rho_minus.size() == simulation_->nEle);

      Eigen::Vector4d X_plus_row = mat_variables_.row(xI).transpose();
      X_plus_row(xJ) += finite_difference_step_;
      std::vector<Eigen::VectorXd> rho_plus = modeling_->ComputeRhosFD(
          effected_flags[varI], xI, X_plus_row.head<3>(), X_plus_row(3), finding_vision_);
      Assert(rho_plus.size() == simulation_->nEle);

      mat_varibale_to_effected_macro_indices[varI].resize(simulation_->nEle);
      int cnt = 0;
      for (int macI = 0; macI < simulation_->nEle; ++macI) {
        if (effected_flags[varI](macI)) {
          mat_varibale_to_effected_macro_indices[varI](cnt++) = macI;
        }
      }
      mat_varibale_to_effected_macro_indices[varI].conservativeResize(cnt);
      mat_varibale_to_effected_macro_num[varI] = cnt;

      dH[varI].resize(mat_varibale_to_effected_macro_num[varI]);
      for (int i = 0; i < mat_varibale_to_effected_macro_num[varI]; ++i) {
        int macI    = mat_varibale_to_effected_macro_indices[varI](i);
        dH[varI][i] = (rho_plus[macI] - rho_minus[macI]) / (2 * finite_difference_step_);
        Assert(
            dH[varI][i].size() ==
            simulation_->nested_background_->nested_cells_.at(macI).tetrahedrons.NumTetrahedrons());
      }
    }
    // ++show_progress;
  }
  std::cout << "time cost for rho: " << time_rho << std::endl;
}

double CVTOptimizer::EvaluateVolume(
    const std::vector<Eigen::VectorXd> &rhos, const std::vector<std::vector<Eigen::VectorXd>> &dH,
    const std::vector<Eigen::VectorXi> &mat_variable_to_effected_macro_indices,
    Eigen::VectorXd &dV) const {
  double Vol = 0;
  for (int macI = 0; macI < simulation_->nEle; ++macI) {
    Vol += (simulation_->mic_Vol[macI].array() * rhos[macI].array()).sum();
  }
  Vol /= simulation_->Vol0;

  dV.setZero(num_variables_);
  oneapi::tbb::parallel_for(0, static_cast<int>(num_variables_), 1, [&](int xI_) {
    const std::vector<Eigen::VectorXd> &dH_x = dH[xI_];
    for (int i = 0; i < mat_variable_to_effected_macro_indices.at(xI_).size(); ++i) {
      int macI = mat_variable_to_effected_macro_indices.at(xI_)(i);

      dV[xI_] += ((simulation_->mic_Vol[macI] / simulation_->Vol0).array() * dH_x[i].array()).sum();
    }
  });

  return Vol;
}

double CVTOptimizer::EvaluateEnergyC(
    const Eigen::VectorXd &displacements, const std::vector<Eigen::VectorXd> &rhos,
    const std::vector<std::vector<Eigen::VectorXd>> &dH,
    const std::vector<Eigen::VectorXi> &mat_variable_to_effected_macro_indices,
    Eigen::VectorXd &dC) const {
  double energy_C = simulation_->load.dot(displacements);

  // compute dCdH
  std::vector<Eigen::VectorXd> dCdH(simulation_->nEle);
  for (int macI = 0; macI < simulation_->nEle; ++macI) {
    const int num_tetrahedrons =
        simulation_->nested_background_->nested_cells_.at(macI).tetrahedrons.NumTetrahedrons();
    dCdH[macI].resize(num_tetrahedrons);

    oneapi::tbb::parallel_for(0, num_tetrahedrons, 1, [&](int micI) {
      double d_YM_rho = simulation_->penaltyYM *
                        pow(rhos[macI](micI), simulation_->penaltyYM - 1.0) *
                        (simulation_->YM1 - simulation_->YM0);
      const Eigen::MatrixXd &M2 =
          simulation_->elementM2[macI](Eigen::seq(micI * 12, (micI + 1) * 12 - 1), Eigen::all);
      Eigen::Vector<double, 12> M2_mul_U = M2 * simulation_->elementU[macI];
      dCdH[macI](micI) =
          d_YM_rho * M2_mul_U.transpose() * simulation_->mic_Ke[macI][micI] * M2_mul_U;
    });
  }

  dC.setZero(num_variables_);
  oneapi::tbb::parallel_for(0, static_cast<int>(num_variables_), 1, [&](int xI_) {
    const std::vector<Eigen::VectorXd> &dH_x = dH[xI_];

    for (int i = 0; i < mat_variable_to_effected_macro_indices.at(xI_).size(); ++i) {
      int macI = mat_variable_to_effected_macro_indices.at(xI_)(i);
      dC[xI_] += (dCdH[macI].array() * dH_x[i].array()).sum();
    }
  });
  return energy_C;
}

double CVTOptimizer::EvaluateEnergyE(Eigen::VectorXd &dE) {
  double E            = 0;
  constexpr int pNorm = 4;
  Eigen::MatrixXd mat_centroids;
  Eigen::VectorXi seed_valid_flags;
  modeling_->ComputeCellCentroids(mat_centroids, seed_valid_flags);
  Assert(mat_centroids.rows() == num_seeds_ && mat_centroids.cols() == 3);
  Assert(seed_valid_flags.size() == num_seeds_);
  dE.setZero(num_variables_);
  for (int cI = 0; cI < num_seeds_; ++cI) {
    if (seed_valid_flags(cI)) {
      dE(cI * num_cols_)     = std::pow(mat_variables_(cI, 0) - mat_centroids(cI, 0), pNorm - 1);
      dE(cI * num_cols_ + 1) = std::pow(mat_variables_(cI, 1) - mat_centroids(cI, 1), pNorm - 1);
      dE(cI * num_cols_ + 2) = std::pow(mat_variables_(cI, 2) - mat_centroids(cI, 2), pNorm - 1);
      E += std::pow(mat_variables_(cI, 0) - mat_centroids(cI, 0), pNorm) +
           std::pow(mat_variables_(cI, 1) - mat_centroids(cI, 1), pNorm) +
           std::pow(mat_variables_(cI, 2) - mat_centroids(cI, 2), pNorm);
    }
  }
  // double E = dE.squaredNorm();
  dE *= pNorm;
  return E;
}

double CVTOptimizer::EvaluateEnergyR(Eigen::VectorXd &dR) {
  double R = 0;
  dR.setZero(num_variables_);
  const auto &beam_mesh    = modeling_->voronoi_beams_mesh_;
  double total_beam_length = 0;
  for (index_t beam_idx = 0; beam_idx < beam_mesh.NumBeams(); ++beam_idx) {
    if (modeling_->map_voronoi_beam_idx_to_cell_indices[beam_idx].size() != 3) continue;
    double beam_length = (beam_mesh.mat_coordinates.row(beam_mesh.mat_beams(beam_idx, 0)) -
                          beam_mesh.mat_coordinates.row(beam_mesh.mat_beams(beam_idx, 1)))
                             .norm();
    total_beam_length += beam_length;
  }

  for (index_t beam_idx = 0; beam_idx < beam_mesh.NumBeams(); ++beam_idx) {
    if (modeling_->map_voronoi_beam_idx_to_cell_indices[beam_idx].size() != 3) continue;
    double beam_length = (beam_mesh.mat_coordinates.row(beam_mesh.mat_beams(beam_idx, 0)) -
                          beam_mesh.mat_coordinates.row(beam_mesh.mat_beams(beam_idx, 1)))
                             .norm();
    std::vector<index_t> cell_indices(
        modeling_->map_voronoi_beam_idx_to_cell_indices[beam_idx].begin(),
        modeling_->map_voronoi_beam_idx_to_cell_indices[beam_idx].end());
    // cell_indices.size() == 3
    index_t vtx_a = cell_indices.at(0);
    index_t vtx_b = cell_indices.at(1);
    index_t vtx_c = cell_indices.at(2);

    const auto &coord_a = beam_mesh.mat_coordinates.row(vtx_a);
    const auto &coord_b = beam_mesh.mat_coordinates.row(vtx_b);
    const auto &coord_c = beam_mesh.mat_coordinates.row(vtx_c);
    // const Eigen::Vector3d normal   = (coord_a - coord_b).cross(coord_a - coord_c).normalized();

    autodiff::VectorXvar ad_values(9);
    ad_values << coord_a.x(), coord_a.y(), coord_a.z(),  //
        coord_b.x(), coord_b.y(), coord_b.z(),           //
        coord_c.x(), coord_c.y(), coord_c.z();           //

    Eigen::Vector3<autodiff::var> a(ad_values(0), ad_values(1), ad_values(2)),
        b(ad_values(3), ad_values(4), ad_values(5)), c(ad_values(6), ad_values(7), ad_values(8));
    Eigen::Vector3<autodiff::var> normal = (a - b).cross(a - c).normalized();
    autodiff::var normal_z               = abs(normal.z());  // (0, 1)

    constexpr double kAlpha = 1;
    constexpr double kEps   = 1e-6;
    autodiff::var penalty   = (beam_length / total_beam_length) * -(log10(normal_z + kEps));
    // autodiff::var penalty = (beam_length / total_beam_length) * (1 / (normal_z + 1.0) - 0.5);
    // autodiff::var penalty = (beam_length / total_beam_length) * (exp(2 * kAlpha * normal_z) - 1)
    // /
    //                         (exp(2 * kAlpha * normal_z) + 1);

    auto gradient = autodiff::gradient(penalty, ad_values);

    R += penalty.expr->val;
    dR(vtx_a * num_cols_ + 0) += gradient(0);
    dR(vtx_a * num_cols_ + 1) += gradient(1);
    dR(vtx_a * num_cols_ + 2) += gradient(2);

    dR(vtx_b * num_cols_ + 0) += gradient(3);
    dR(vtx_b * num_cols_ + 1) += gradient(4);
    dR(vtx_b * num_cols_ + 2) += gradient(5);

    dR(vtx_c * num_cols_ + 0) += gradient(6);
    dR(vtx_c * num_cols_ + 1) += gradient(7);
    dR(vtx_c * num_cols_ + 2) += gradient(8);
  }
  return R;
}
}  // namespace da