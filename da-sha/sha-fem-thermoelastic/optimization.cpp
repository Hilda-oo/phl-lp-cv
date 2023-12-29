#include "optimization.h"
#include "mma/MMASolver.h"
#include "util.h"
namespace da::sha {
void ThermoelasticOpt::optimize() {
  Eigen::VectorXd xPhys_col(sp_therMech_sim_->GetNumEles());
  xPhys_col.setConstant(sp_para_->volfrac / sp_para_->volfrac);

  // preCompute
  int loop = 0;
  double change = 1.0;
  double E0_m = sp_therMech_sim_->E_;
  double E_min = E0_m * sp_para_->E_factor;
  double lambda0 = sp_thermal_sim_->thermal_conductivity_;
  double lambda_min = lambda0 * sp_para_->E_factor;
  double alpha0 = sp_therMech_sim_->thermal_expansion_coefficient_;
  double alpha_min = alpha0 * sp_para_->E_factor;
  std::vector<int> v_dof(sp_thermal_sim_->set_dofs_to_load_.begin(),
                         sp_thermal_sim_->set_dofs_to_load_.end());
  spdlog::debug("end Precompute");

  // start iteration
  while (change > sp_para_->tol_x && loop < sp_para_->max_loop) {
    ++loop;
    // process material interpolation before the simulation
    sp_thermal_sim_->setMatrialTC(
        CalLambda_Vec(xPhys_col, lambda_min, lambda0, sp_para_->R_lambda));
    Eigen::VectorXd p_E_vec = CalE_Vec(xPhys_col, E_min, E0_m, sp_para_->R_E);
    Eigen::VectorXd p_beta_vec =
        CalBeta_Vec(xPhys_col, E0_m, alpha0, sp_para_->R_beta);
    sp_therMech_sim_->setMaterialParam(p_E_vec, p_beta_vec);
    // simulation
    sp_thermal_sim_->simulation();
    Eigen::VectorXd T = sp_thermal_sim_->U_;
    //// Fth
    sp_therMech_sim_->preCompute(T, sp_para_->T_ref, sp_thermal_sim_->eDof_);

    if (loop == 1 || loop == sp_para_->max_loop)
      spdlog::info(
          "||Fth|| / ||Fm||: {}, ||Fm||: {}, ||Fth||: {}",
          sp_therMech_sim_->Fth_.norm() / sp_therMech_sim_->load_.norm(),
          sp_therMech_sim_->load_.norm(), sp_therMech_sim_->Fth_.norm());
    sp_therMech_sim_->simulation();
    // dv
    Eigen::VectorXd dv(sp_therMech_sim_->GetNumEles()); // diff from lin
    dv = sp_therMech_sim_->GetV();
    // volume
    double v = xPhys_col.sum();
    // sensitivity
    Eigen::VectorXd dc_drho;
    // compliance
    double c = EvaluateEnergyC(xPhys_col, E_min, E0_m, alpha0, lambda_min,
                               lambda0, dc_drho);
    // constrain

    // dT_drho
    Eigen::MatrixXd dT_drho =
        EvaluateTemperatureConstrain(xPhys_col, lambda_min, lambda0, v_dof);
// mma solver
#define SENSITIVITY_SCALE_COEF 200
    size_t num_constraints =
        1 + dT_drho.cols(); // volume and temperature constraints
    size_t num_variables = sp_therMech_sim_->GetNumEles();
    auto mma = std::make_shared<MMASolver>(num_variables, num_constraints);
    Eigen::VectorXd variables_tmp = xPhys_col;
    double f0val = c;
    Eigen::VectorXd df0dx =
        dc_drho / dc_drho.cwiseAbs().maxCoeff() * SENSITIVITY_SCALE_COEF;

    Eigen::VectorXd fval = (Eigen::VectorXd(num_constraints)
                                << (v / num_variables - sp_para_->volfrac),
                            T(v_dof).array() / sp_para_->T_limit - 1)
                               .finished() *
                           SENSITIVITY_SCALE_COEF;
    Eigen::VectorXd dv_constraint = 1.0 / num_variables * dv;
    Eigen::MatrixXd dt_constraint = 1.0 / sp_para_->T_limit * dT_drho;
    Eigen::MatrixXd dfdx =
        (Eigen::MatrixXd(num_variables, num_constraints) << dv_constraint,
         dt_constraint)
            .finished().transpose() *
        SENSITIVITY_SCALE_COEF;

    static Eigen::VectorXd low_bounds = Eigen::VectorXd::Zero(num_variables);
    static Eigen::VectorXd up_bounds = Eigen::VectorXd::Ones(num_variables);

    mma->Update(variables_tmp.data(), df0dx.data(), fval.data(), dfdx.data(),
                low_bounds.data(), up_bounds.data());

    change = (variables_tmp - xPhys_col).cwiseAbs().maxCoeff();
    xPhys_col = variables_tmp;

    spdlog::critical("Iter: {:3d}, Comp: {:.3e}, Vol: {:.2f}, Change: {:f}",
                     loop, c, v, change);
  }
  // result
  rhos_ = xPhys_col;
}

double ThermoelasticOpt::EvaluateEnergyC(const Eigen::VectorXd &xPhys_col,
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
        CalDBetaDrho(xPhys_col(i), E0_m, alpha0, sp_para_->R_beta) *
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
    double beta_rho = CalBeta(xPhys_col(i), E0_m, alpha0, sp_para_->R_beta);
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
    rhs(dof) = sp_thermal_sim_->K_spMat_.coeff(dof, dof) * value;
  }
  Eigen::VectorXd lambda_t = sp_thermal_sim_->solver_.solve(rhs);

  // lambda_m_Mul_dKm_drho_Mul_U
  Eigen::VectorXd lambda_m_Mul_dKm_drho_Mul_U =
      -CalDEDrho_Vec(xPhys_col, E_min, E0_m, sp_para_->R_E).array() *
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
      CalDlambdaDrho_Vec(xPhys_col, lambda_min, lambda0, sp_para_->R_lambda)
          .array() *
      lambda_t_Mul_Kt_Mul_Te.array() /
      sp_thermal_sim_->material_int_TC_.array();

  dC = lambda_t_Mul_dKt_drho_Mul_T + lambda_m_Mul_dKm_drho_Mul_U +
       2 * Eigen::VectorXd(dFth_drho * sp_therMech_sim_->U_);
  return energy_C;
}

auto ThermoelasticOpt::EvaluateTemperatureConstrain(
    const Eigen::VectorXd &xPhys_col, double lambda_min, double lambda0,
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
    Eigen::VectorXd dKe_th_Mul_T = CalDlambdaDrho(xPhys_col(rho_i), lambda_min,
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
      Li(dof) = sp_thermal_sim_->K_spMat_.coeff(dof, dof) * value;
    }
    Eigen::VectorXd lambda_i = sp_thermal_sim_->solver_.solve(Li);
    // auto solver_th = sp_thermal_sim_->GetLinSysSolver();
    // solver_th->Solve(Li, lambda_i);
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