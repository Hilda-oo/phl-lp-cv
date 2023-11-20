#include "ThermoelasticWrapper.h"
#include "mma/MMASolver.h"
#include "util.h"
namespace da::sha {
void ThermoelasticWrapper::simulate(Eigen::VectorXd p_rhos) {

  // preCompute
  double change = 1.0;
  double E0_m = sp_therMech_sim_->E_;
  double E_min = E0_m * sp_para_->E_factor;
  double lambda0 = sp_thermal_sim_->thermal_conductivity_;
  double lambda_min = lambda0 * sp_para_->E_factor;
  double alpha0 = sp_therMech_sim_->thermal_expansion_coefficient_;
  double alpha_min = alpha0 * sp_para_->E_factor;
  std::vector<int> v_dof(sp_thermal_sim_->set_dofs_to_load_.begin(),
                         sp_thermal_sim_->set_dofs_to_load_.end());
  spdlog::info("end Precompute");

    // process material interpolation before the simulation
    sp_thermal_sim_->setMatrialTC(
        CalLambda_Vec(p_rhos, lambda_min, lambda0, sp_para_->R_lambda));
    Eigen::VectorXd p_E_vec = CalE_Vec(p_rhos, E_min, E0_m, sp_para_->R_E);
    Eigen::VectorXd p_beta_vec =
        CalBeta_Vec(p_rhos, E0_m, alpha0, sp_para_->R_beta);
    sp_therMech_sim_->setMaterialParam(p_E_vec, p_beta_vec);
    // simulation
    sp_thermal_sim_->simulation();
    Eigen::VectorXd T = sp_thermal_sim_->U_;
    //// Fth
    sp_therMech_sim_->preCompute(T, sp_para_->T_ref, sp_thermal_sim_->eDof_);

      spdlog::info(
          "||Fth|| / ||Fm||: {}, ||Fm||: {}, ||Fth||: {}",
          sp_therMech_sim_->Fth_.norm() / sp_therMech_sim_->load_.norm(),
          sp_therMech_sim_->load_.norm(), sp_therMech_sim_->Fth_.norm());
    sp_therMech_sim_->simulation();

    // sensitivity
    // compliance
    c_ = EvaluateEnergyC(p_rhos, E_min, E0_m, alpha0, lambda_min,
                               lambda0, dc_drho_);
    // constrain

    // dT_drho
    dT_drho_ =
        EvaluateTemperatureConstrain(p_rhos, lambda_min, lambda0, v_dof);
}

double ThermoelasticWrapper::EvaluateEnergyC(const Eigen::VectorXd &p_rhos,
                                         double E_min, double E0_m,
                                         double alpha0, double lambda_min,
                                         double lambda0,
                                         Eigen::VectorXd &dC) const {

  Eigen::VectorXd ce(sp_therMech_sim_->GetNumEles());
  for (int i = 0; i < sp_therMech_sim_->GetNumEles(); ++i) {
    Eigen::VectorXi dofs_in_ele_i = sp_therMech_sim_->GetMapEleId2DofsVec()[i];
    Eigen::VectorXd Ue = sp_therMech_sim_->U_(dofs_in_ele_i);
    ce(i) = Ue.transpose() * sp_therMech_sim_->GetElementK(i) * Ue;
  }
  // double energy_C = sp_therMech_sim_->F_all_.dot(sp_therMech_sim_->U_);
  double energy_C = ce.sum();

  // lambda_m
  Eigen::VectorXd lambda_m = -sp_therMech_sim_->U_;
  // dFth_drho
  Eigen::SparseMatrix<double> dFth_drho(sp_therMech_sim_->GetNumEles(),
                                        sp_therMech_sim_->GetNumDofs());
  //每个元素dFth_drho 12x1
  Eigen::VectorXd v_dFth_drho(i_dFth_drho_.size());
  for (int i = 0; i < sp_thermal_sim_->GetNumEles(); ++i) {
    Eigen::VectorXi dofs_th = sp_thermal_sim_->GetMapEleId2DofsVec()[i];
    double Te = sp_thermal_sim_->U_(dofs_th).mean();
    Eigen::VectorXd ele_dFth_drho =
        CalDBetaDrho(p_rhos(i), E0_m, alpha0, sp_para_->R_beta) *
        (Te - sp_para_->T_ref) * sp_therMech_sim_->GetElementB(i).transpose() *
        Inted_; // 12x1
    assert(ele_dFth_drho.size() == 12);
    v_dFth_drho(Eigen::seqN(i * ele_dFth_drho.size(), ele_dFth_drho.size())) =
        ele_dFth_drho;
  }
  auto v_dFth_drho_tri = Vec2Triplet(i_dFth_drho_, j_dFth_drho_, v_dFth_drho);
  dFth_drho.setFromTriplets(v_dFth_drho_tri.begin(), v_dFth_drho_tri.end());

  // dFth^T_dT
  Eigen::SparseMatrix<double> dFth_dT(sp_thermal_sim_->GetNumDofs(),
                                      sp_therMech_sim_->GetNumDofs());
  Eigen::VectorXd v_dFth_dT(i_dFth_dT_.size());
  for (int i = 0; i < sp_thermal_sim_->GetNumEles(); ++i) {
    double beta_rho = CalBeta(p_rhos(i), E0_m, alpha0, sp_para_->R_beta);
    Eigen::MatrixXd ele_dFth_dT =
        Eigen::VectorXd::Ones(sp_thermal_sim_->Get_DOFS_EACH_ELE()) * 1.0 /
        sp_thermal_sim_->Get_DOFS_EACH_ELE() * beta_rho *
        (sp_therMech_sim_->GetElementB(i).transpose() * Inted_).transpose();
    assert(ele_dFth_dT.rows() == 4 && ele_dFth_dT.cols() == 12);
    v_dFth_dT(Eigen::seqN(i * ele_dFth_dT.size(), ele_dFth_dT.size())) =
        ele_dFth_dT.reshaped();
  }
  auto v_dFth_dT_tri = Vec2Triplet(i_dFth_dT_, j_dFth_dT_, v_dFth_dT);
  dFth_dT.setFromTriplets(v_dFth_dT_tri.begin(), v_dFth_dT_tri.end());

  // lambda_t
  Eigen::VectorXd rhs = dFth_dT * 2 * lambda_m;
  for (auto dof_value : sp_thermal_sim_->v_dofs_to_set_) {
    auto [dof, value] = dof_value;
    rhs(dof) = sp_thermal_sim_->K_(dof, dof) * value;
  }
  Eigen::VectorXd lambda_t = sp_thermal_sim_->solver_.solve(rhs);

  // lambda_m_Mul_dKm_drho_Mul_U
  Eigen::VectorXd lambda_m_Mul_dKm_drho_Mul_U =
      -CalDEDrho_Vec(p_rhos, E_min, E0_m, sp_para_->R_E).array() *
      ce.array() / sp_therMech_sim_->material_int_E_.array();

  // lambda_t_Mul_dKt_drho_Mul_T
  Eigen::VectorXd lambda_t_Mul_Kt_Mul_Te(sp_thermal_sim_->GetNumEles());
  for (int i = 0; i < sp_thermal_sim_->GetNumEles(); ++i) {
    Eigen::VectorXi dofs_in_ele_i = sp_thermal_sim_->GetMapEleId2DofsVec()[i];
    Eigen::VectorXd Te = sp_thermal_sim_->U_(dofs_in_ele_i);
    Eigen::VectorXd lambda_t_e = lambda_t(dofs_in_ele_i);
    lambda_t_Mul_Kt_Mul_Te(i) =
        lambda_t_e.transpose() * sp_thermal_sim_->GetElementK(i) * Te;
  }
  Eigen::VectorXd lambda_t_Mul_dKt_drho_Mul_T =
      CalDlambdaDrho_Vec(p_rhos, lambda_min, lambda0, sp_para_->R_lambda)
          .array() *
      lambda_t_Mul_Kt_Mul_Te.array() /
      sp_thermal_sim_->material_int_TC_.array();

  dC = lambda_t_Mul_dKt_drho_Mul_T + lambda_m_Mul_dKm_drho_Mul_U +
       2 * Eigen::VectorXd(dFth_drho * sp_therMech_sim_->U_);
  return energy_C;
}

auto ThermoelasticWrapper::EvaluateTemperatureConstrain(
    const Eigen::VectorXd &p_rhos, double lambda_min, double lambda0,
    const std::vector<int> &v_dof) -> Eigen::MatrixXd {

  //      dofs of limited T
  std::map<int, std::vector<LimitedDof>> map_ele2Limit;
  Eigen::MatrixXi ele2dof_map = sp_thermal_sim_->GetMapEleId2DofsMat();
  //      loop ele2dof_map
  for (int i = 0; i < ele2dof_map.rows(); ++i) {
    for (int j = 0; j < ele2dof_map.cols(); ++j) {
      for (int k = 0; k < v_dof.size(); ++k) {
        if (ele2dof_map(i, j) == v_dof[k]) {
          if (map_ele2Limit.find(i) == map_ele2Limit.end()) {
            map_ele2Limit[i] = {LimitedDof(v_dof[k], k, j)};
          } else {
            map_ele2Limit[i].push_back(LimitedDof(v_dof[k], k, j));
          }
        }
      }
    }
  }
  auto CalDKth_Mul_T__For_DTi_Drhoj =
      [&](int rho_i) -> Eigen::SparseVector<double> {
    Eigen::VectorXi dofs_in_ele = sp_thermal_sim_->GetMapEleId2DofsVec()[rho_i];
    Eigen::VectorXd ele_T = sp_thermal_sim_->U_(dofs_in_ele);
    Eigen::SparseVector<double> sp_dKth_Mul_T(sp_thermal_sim_->GetNumDofs());
    Eigen::VectorXd dKe_th_Mul_T = CalDlambdaDrho(p_rhos(rho_i), lambda_min,
                                                  lambda0, sp_para_->R_lambda) *
                                   sp_thermal_sim_->GetElementK(rho_i) /
                                   sp_thermal_sim_->material_int_TC_(rho_i) *
                                   ele_T;
    for (int i = 0; i < dofs_in_ele.size(); ++i) {
      sp_dKth_Mul_T.coeffRef(dofs_in_ele(i)) = dKe_th_Mul_T(i);
    }
    return sp_dKth_Mul_T;
  };
  auto CalDTi_Drhoj =
      [&](int Tdof,
          const Eigen::SparseVector<double> &sp_dKth_Mul_T) -> double {
    Eigen::VectorXd Li = Eigen::VectorXd::Zero(sp_dKth_Mul_T.rows());
    Li(Tdof) = -1;
    for (auto dof_value : sp_thermal_sim_->v_dofs_to_set_) {
      auto [dof, value] = dof_value;
      Li(dof) = sp_thermal_sim_->K_(dof, dof) * value;
    }
    Eigen::VectorXd lambda_i = sp_thermal_sim_->solver_.solve(Li);
    return lambda_i.transpose() * sp_dKth_Mul_T;
  };
  Eigen::MatrixXd dT = Eigen::MatrixXd::Zero(
      sp_thermal_sim_->GetNumEles(), sp_thermal_sim_->set_dofs_to_load_.size());
  for (auto it = map_ele2Limit.begin(); it != map_ele2Limit.end(); ++it) {
    auto [ele_id, v_limited] = *it;
    auto sp_dKth_Mul_T = CalDKth_Mul_T__For_DTi_Drhoj(ele_id);
    for (auto &limited : v_limited) {
      dT(ele_id, limited.idx_of_load_dof) =
          CalDTi_Drhoj(limited.dof, sp_dKth_Mul_T);
    }
  }
  return dT;
}

} // namespace da::sha