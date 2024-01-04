#include "optimization.h"
#include <mma/MMASolver.h>
#include <spdlog/spdlog.h>
#include <autodiff/reverse/var/eigen.hpp>
#include "Eigen/src/Core/Matrix.h"
#include "sha-io-foundation/data_io.h"

namespace da {

LpCVTOptimizer::LpCVTOptimizer(
    const MatMesh3 &mesh, int p, const Eigen::AlignedBox3d &design_domain,
    const std::shared_ptr<ModelAlgorithm> &modeling,
    const std::shared_ptr<sha::ThermoelasticWrapper> &simulation,
    const std::shared_ptr<da::AnisotropicMatWrapper> anisotropic_mat_wrapper, double init_radius,
    const std::pair<double, double> radius_range, const double scalar_E, const double volfrac,
    const Eigen::MatrixXd &mat_init_seeds)
    : design_domain_(design_domain),
      modeling_(modeling),
      simulation_(simulation),
      anisotropic_mat_wrapper_(anisotropic_mat_wrapper),
      scalar_E_(scalar_E),
      volfrac_(volfrac),
      lp_CVT_wrapper_(std::make_shared<LpNormCVTWrapper>(mesh, p)),
      p_(p),
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
  finite_difference_step_ = simulation_->ComputeAverageMicroTetEdgeLength();
  log::info("finite_difference_step = {}", finite_difference_step_);

  //
  finding_vision_(0) = design_domain_.sizes().x() / std::pow(num_seeds_, 1.0 / 3) * 3.0 / 2.0;
  finding_vision_(1) = design_domain_.sizes().y() / std::pow(num_seeds_, 1.0 / 3) * 3.0 / 2.0;
  finding_vision_(2) = design_domain_.sizes().z() / std::pow(num_seeds_, 1.0 / 3) * 3.0 / 2.0;
  spdlog::debug("finding_vision: {:.3f} {:.3f} {:.3f}", finding_vision_(0), finding_vision_(1),
                finding_vision_(2));
  // radius oo
  modeling_->min_radius_ = radius_range.first;
  modeling_->max_radius_ = radius_range.second;
}

void LpCVTOptimizer::Optimize(Eigen::MatrixXd &mat_seeds_result, Eigen::VectorXd &radiuses_result,
                              size_t num_iterations, size_t mode,
                              IterationFunctionType IterationFunction) {
  // init
  const int dgt0 = 5;
  double objVr5  = 1.0;

  size_t num_constrants = 1;
  std::shared_ptr<MMASolver> mma;

  std::vector<double> OBJ;

  modeling_->Update(mat_variables_);

  log::info("Start iterations: {}", num_iterations);

  if (IterationFunction != nullptr) {
    IterationFunction(0, mat_variables_, 0, Eigen::VectorXd(), 0, Eigen::VectorXd(), 0,
                      Eigen::VectorXd());
  }

  bool not_last_iter_flag = true;
  Eigen::VectorXd changeVec;
  changeVec.resize(100);
  // iterations
  for (int iteration_idx = 1, ch_num_stop = 3;
       iteration_idx <= num_iterations && not_last_iter_flag; ++iteration_idx) {
    if (iteration_idx == num_iterations || ch_num_stop == 0) not_last_iter_flag = false;

    // modeling
    std::vector<Eigen::VectorXd> rhos = modeling_->ComputeRhos();

    // precompute for thermo-elastic simulation
    Eigen::VectorXd rhos_para;
    int size = rhos.size();
    std::vector<double> rhos_vec(0);
    for (int rI = 0; rI < size; ++rI) {
      Eigen::VectorXd rho = rhos.at(rI);
      int row             = rho.rows();
      for (int ri = 0; ri < row; ++ri) {
        rhos_vec.push_back(rho(ri));
      }
    }
    size = rhos_vec.size();
    rhos_para.resize(size);
    std::copy_n(rhos_vec.data(), size, rhos_para.data());

    double C, V, E;
    Eigen::VectorXd dC, dT, dV, dE;

    // simulation
    Eigen::VectorXd dCdH;
    Eigen::MatrixXd dTdH;
    simulation_->simulate(rhos_para, dCdH, dTdH, C);

    // compute differential
    std::vector<std::vector<Eigen::VectorXd>> dH;
    std::vector<Eigen::VectorXi> mat_variable_to_effected_macro_indices;
    EvaluateFiniteDifference(dH, mat_variable_to_effected_macro_indices);

    V  = EvaluateVolume(rhos, dH, mat_variable_to_effected_macro_indices, dV);
    E  = EvaluetaEnergyL(dE, mode);
    dC = EvaluateDC(dCdH, rhos, dH, mat_variable_to_effected_macro_indices);
#ifndef MECH_ONLY
    dT = EvaluateDT(dTdH, rhos, dH, mat_variable_to_effected_macro_indices);

    spdlog::debug("prepare for mma");

    dC /= dC.cwiseAbs().maxCoeff();
    dE /= dE.cwiseAbs().maxCoeff();
    log::info("mean: dC:{}, dV:{}, dE:{}, dT:{}", dC.mean(), dV.mean(), dE.mean(), dT.mean());
    log::info("C = {:.6}, E = {:.6}, V = {:.3f}", C, E, V);

    if (iteration_idx == 1) {
      num_constrants = 1 + dT.cols();
      mma            = std::make_shared<MMASolver>(num_variables_, num_constrants);
    }

    // double f0val = C;
    // double f0val = C * 10e-12 * (1.0 - scalar_E_) + E * scalar_E_; //cube206
    double f0val = C * (1.0 - scalar_E_) + E * scalar_E_;

    // Eigen::VectorXd df0dx = dC;
    Eigen::VectorXd df0dx = dC * (1.0 - scalar_E_) + dE * scalar_E_;
    auto setDofT =
        simulation_->getSetDofThermalU().array() / simulation_->getContrlPara()->T_limit - 1;
    Eigen::VectorXd fval =
        (Eigen::VectorXd(num_constrants) << (V / num_variables_ - volfrac_), setDofT).finished();
    Eigen::VectorXd dv_constraint = 1.0 / num_variables_ * dV;
    Eigen::MatrixXd dt_constraint = 1.0 / simulation_->getContrlPara()->T_limit * dT;
    Eigen::MatrixXd dfdx =
        (Eigen::MatrixXd(num_variables_, num_constrants) << dv_constraint, dt_constraint)
            .finished()
            .transpose();
#else
    spdlog::debug("prepare for mma");

    dC /= dC.cwiseAbs().maxCoeff();
    dE /= dE.cwiseAbs().maxCoeff();
    log::info("mean: dC:{}, dV:{}, dE:{}", dC.mean(), dV.mean(), dE.mean());
    log::info("C = {:.6}, E = {:.6}, V = {:.3f}", C, E, V);

    if (iteration_idx == 1) {
      num_constrants = 1;
      mma            = std::make_shared<MMASolver>(num_variables_, num_constrants);
    }

    // double f0val = C;
    // double f0val = C * 10e-12 * (1.0 - scalar_E_) + E * scalar_E_; //cube206
    double f0val = C * (1.0 - scalar_E_) + E * scalar_E_;

    // Eigen::VectorXd df0dx = dC;
    Eigen::VectorXd df0dx = dC * (1.0 - scalar_E_) + dE * scalar_E_;
    Eigen::VectorXd fval =
        (Eigen::VectorXd(num_constrants) << V / num_variables_ - volfrac_).finished();

    Eigen::VectorXd dv_constraint = 1.0 / num_variables_ * dV;
    Eigen::MatrixXd dfdx =
        (Eigen::MatrixXd(num_variables_, num_constrants) << dv_constraint).finished().transpose();
#endif
    spdlog::info("fval: {}, f0val: {}", fval.sum(), f0val);

    Eigen::VectorXd variables_tmp = mat_variables_.transpose().reshaped(num_variables_, 1);

    spdlog::info("mma update");
    mma->Update(variables_tmp.data(), df0dx.data(), fval.data(), dfdx.data(),
                variables_bounds_.first.data(), variables_bounds_.second.data());

    mat_variables_ = variables_tmp.reshaped(num_cols_, num_seeds_).transpose();

    // update modeling
    spdlog::info("modeling");
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

    if (objVr5 <= 1e-3)
      ch_num_stop--;
    else
      ch_num_stop = 3;

    if (IterationFunction != nullptr) {
      IterationFunction(iteration_idx, mat_variables_, C, dC, E, dE, V, dV);
    }

    spdlog::critical("Optimization iter# {}, C = {:.6}, E = {:.6}, V = {:.3f}, ch = {:.7f}",
                     iteration_idx, C, E, V, objVr5);
    changeVec[iteration_idx - 1] = objVr5;
  }
  mat_seeds_result = mat_variables_.leftCols(3);
  radiuses_result  = mat_variables_.col(3);
  sha::WriteVectorToFile(WorkingResultDirectoryPath() / "ch.txt", changeVec);
}

void LpCVTOptimizer::EvaluateFiniteDifference(
    std::vector<std::vector<Eigen::VectorXd>> &dH,
    std::vector<Eigen::VectorXi> &mat_varibale_to_effected_macro_indices) {
  spdlog::debug("compute dH");
  dH.resize(num_variables_);
  // boost::progress_display show_progress(nCell);
  double time_rho = 0.0;

  mat_varibale_to_effected_macro_indices.clear();
  mat_varibale_to_effected_macro_indices.resize(num_variables_);
  std::vector<Eigen::VectorXi> effected_flags(num_variables_);
  std::vector<int> mat_varibale_to_effected_macro_num(num_variables_);

  modeling_->UpdateFD(mat_variables_);
  modeling_->ComputeRhosFD();

  int macro_cell_num = simulation_->getMacroCellNum();
  for (int xI = 0; xI < num_seeds_; ++xI) {
    for (int xJ = 0; xJ < num_cols_; ++xJ) {
      int varI = xI * num_cols_ + xJ;

      Eigen::Vector4d X_minus_row = mat_variables_.row(xI).transpose();
      X_minus_row(xJ) -= finite_difference_step_;
      std::vector<Eigen::VectorXd> rho_minus = modeling_->ComputeRhosFD(
          effected_flags[varI], xI, X_minus_row.head<3>(), X_minus_row(3), finding_vision_);
      Assert(rho_minus.size() == macro_cell_num);

      Eigen::Vector4d X_plus_row = mat_variables_.row(xI).transpose();
      X_plus_row(xJ) += finite_difference_step_;
      std::vector<Eigen::VectorXd> rho_plus = modeling_->ComputeRhosFD(
          effected_flags[varI], xI, X_plus_row.head<3>(), X_plus_row(3), finding_vision_);
      Assert(rho_plus.size() == macro_cell_num);

      mat_varibale_to_effected_macro_indices[varI].resize(macro_cell_num);
      int cnt = 0;
      for (int macI = 0; macI < macro_cell_num; ++macI) {
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

double LpCVTOptimizer::EvaluateVolume(
    const std::vector<Eigen::VectorXd> &rhos, const std::vector<std::vector<Eigen::VectorXd>> &dH,
    const std::vector<Eigen::VectorXi> &mat_variable_to_effected_macro_indices,
    Eigen::VectorXd &dV) const {
  spdlog::debug("compute dV");
  Eigen::VectorXd all_mic_vol               = simulation_->getAllTetVol();
  std::vector<Eigen::VectorXi> macId2tetsId = simulation_->getMacIdMaptetsId();
  int macN                                  = simulation_->getMacroCellNum();
  double Vol = 0., Vol0 = all_mic_vol.sum();
  for (int macI = 0; macI < macN; ++macI) {
    Eigen::VectorXd mic_vol = all_mic_vol(macId2tetsId[macI]);
    Vol += (mic_vol.array() * rhos[macI].array()).sum();
  }
  Vol /= Vol0;

  dV.setZero(num_variables_);
  oneapi::tbb::parallel_for(0, static_cast<int>(num_variables_), 1, [&](int xI_) {
    const std::vector<Eigen::VectorXd> &dH_x = dH[xI_];
    for (int i = 0; i < mat_variable_to_effected_macro_indices.at(xI_).size(); ++i) {
      int macI                = mat_variable_to_effected_macro_indices.at(xI_)(i);
      Eigen::VectorXd mic_vol = all_mic_vol(macId2tetsId[macI]);
      dV[xI_] += ((mic_vol / Vol0).array() * dH_x[i].array()).sum();
    }
  });

  return Vol;
}

Eigen::VectorXd LpCVTOptimizer::EvaluateDC(
    const Eigen::VectorXd &dCdrho, const std::vector<Eigen::VectorXd> &rhos,
    const std::vector<std::vector<Eigen::VectorXd>> &dH,
    const std::vector<Eigen::VectorXi> &mat_variable_to_effected_macro_indices) const {
  spdlog::debug("compute dC");
  // compute dCdH
  int macN = simulation_->getMacroCellNum();
  std::vector<Eigen::VectorXd> dCdH(macN);
  auto macId2tetsId = simulation_->getMacIdMaptetsId();
  for (int macI = 0; macI < macN; ++macI) {
    const int num_tetrahedrons =
        simulation_->nested_background_->nested_cells_.at(macI).tetrahedrons.NumTetrahedrons();
    dCdH[macI].resize(num_tetrahedrons);

    Eigen::VectorXi simId = macId2tetsId[macI];
    Assert(simId.size() == num_tetrahedrons, "macId2tetsId is wrong!");

    oneapi::tbb::parallel_for(0, num_tetrahedrons, 1,
                              [&](int micI) { dCdH[macI](micI) = dCdrho(simId(micI)); });
  }
  Eigen::VectorXd dC;
  dC.setZero(num_variables_);
  oneapi::tbb::parallel_for(0, static_cast<int>(num_variables_), 1, [&](int xI_) {
    const std::vector<Eigen::VectorXd> &dH_x = dH[xI_];

    for (int i = 0; i < mat_variable_to_effected_macro_indices.at(xI_).size(); ++i) {
      int macI = mat_variable_to_effected_macro_indices.at(xI_)(i);
      dC[xI_] += (0.5 * dCdH[macI].array() * dH_x[i].array()).sum();
    }
  });
  return dC;
}

Eigen::MatrixXd LpCVTOptimizer::EvaluateDT(
    const Eigen::MatrixXd &dTdrho, const std::vector<Eigen::VectorXd> &rhos,
    const std::vector<std::vector<Eigen::VectorXd>> &dH,
    const std::vector<Eigen::VectorXi> &mat_variable_to_effected_macro_indices) const {
  spdlog::debug("compute dT");
  // compute dTdH
  int macN      = simulation_->getMacroCellNum();
  int setdofNum = dTdrho.cols();
  std::vector<Eigen::MatrixXd> dTdH(macN);
  auto macId2tetsId = simulation_->getMacIdMaptetsId();
  for (int macI = 0; macI < macN; ++macI) {
    const int num_tetrahedrons =
        simulation_->nested_background_->nested_cells_.at(macI).tetrahedrons.NumTetrahedrons();
    dTdH[macI].resize(num_tetrahedrons, setdofNum);

    Eigen::VectorXi simId = macId2tetsId[macI];
    Assert(simId.size() == num_tetrahedrons, "macId2tetsId is wrong!");

    oneapi::tbb::parallel_for(0, num_tetrahedrons, 1, [&](int micI) {
      dTdH[macI](micI, Eigen::all) = dTdrho(simId(micI), Eigen::all);
    });
  }
  Eigen::MatrixXd dT;
  dT.setZero(num_variables_, setdofNum);

  oneapi::tbb::parallel_for(0, static_cast<int>(num_variables_), 1, [&](int xI_) {
    const std::vector<Eigen::VectorXd> &dH_x = dH[xI_];

    for (int i = 0; i < mat_variable_to_effected_macro_indices.at(xI_).size(); ++i) {
      int macI = mat_variable_to_effected_macro_indices.at(xI_)(i);
      dT.row(xI_) += 0.5 * dTdH[macI].transpose() * dH_x[i];
    }
  });
  return dT;
}

double LpCVTOptimizer::EvaluateEnergyR(Eigen::VectorXd &dR) {
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

// Lp CVT Energy
double LpCVTOptimizer::EvaluetaEnergyL(Eigen::VectorXd &dL, size_t mode) {
  spdlog::debug("compute dL");
  // get seeds for stress field computation
  Eigen::MatrixXd query_seeds = lp_CVT_wrapper_->GetQuerySeeds(this->mat_variables_.leftCols(3));

  std::vector<Eigen::Matrix3d> field_matrix;
  switch (mode) {
    case 1:
      // with the resulted stress field of the thermo-elastic fem.
      field_matrix = anisotropic_mat_wrapper_->getAnisotropicMatByFemStress(
          query_seeds, simulation_->queryStress(query_seeds));
      break;
    default:
      // generate identity matrix for each query seed
      int qN = query_seeds.rows();
      for (index_t qI = 0; qI < qN; ++qI) {
        field_matrix.push_back(Eigen::Matrix3d::Identity());
      }
      break;
  }
  Eigen::VectorXd gradients;
  double f =
      lp_CVT_wrapper_->EvaluateLpCVT(this->mat_variables_.leftCols(3), field_matrix, gradients);

  dL.resize(num_variables_);
  dL.setZero();
  for (index_t idx = 0; idx < num_seeds_; ++idx) {
    dL(idx * num_cols_ + 0) = gradients(idx * 3 + 0);
    dL(idx * num_cols_ + 1) = gradients(idx * 3 + 1);
    dL(idx * num_cols_ + 2) = gradients(idx * 3 + 2);
  }
  return f;
}
}  // namespace da