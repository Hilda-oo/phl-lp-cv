#pragma once

#include "sha-fem-thermoelastic/HeatSimulation.h"
#include "sha-fem-thermoelastic/ThermoelasticSim.h"
#include <algorithm>
#include <oneapi/tbb/parallel_for.h>
#include <unsupported/Eigen/src/KroneckerProduct/KroneckerTensorProduct.h>
#include <vector>

namespace da::sha {
struct CtrlPara {
  double volfrac = 0.5;
  double penal = 1.0;
  int max_loop = 100;
  double r_min = 2.0;
  double T_ref = 295;
  double T_limit = 325;
  double tol_x = 0.00001;
  double E_factor = 1e-9;
  double R_E = 8;
  double R_lambda = 0;
  double R_beta = 0;
};

struct LimitedDof {
  int dof;
  int idx_of_load_dof;
  int idx_in_ele;

  LimitedDof(int dof, int idx_of_load_dof, int idx_in_ele)
      : dof(dof), idx_of_load_dof(idx_of_load_dof), idx_in_ele(idx_in_ele) {}
};

class ThermoelasticOpt {
private:
  std::shared_ptr<HeatSimulation> sp_thermal_sim_;
  std::shared_ptr<ThermoelasticSim> sp_therMech_sim_;
  std::shared_ptr<CtrlPara> sp_para_;
  Eigen::VectorXd Inted_;
  Eigen::VectorXi i_dFth_dT_, j_dFth_dT_;
  // i:i: j:每个元素自由度的索引
  Eigen::VectorXi i_dFth_drho_, j_dFth_drho_;
  Eigen::VectorXd rhos_;

public:
  ThermoelasticOpt(std::shared_ptr<HeatSimulation> p_thermal_sim,
                   std::shared_ptr<ThermoelasticSim> p_therMech_sim,
                   std::shared_ptr<CtrlPara> p_para)
      : sp_thermal_sim_(p_thermal_sim), 
        sp_therMech_sim_(p_therMech_sim),
        sp_para_(p_para) {
    preCompute();
  }

  // Compliance
  double EvaluateEnergyC(const Eigen::VectorXd &xPhys_col, double E_min,
                         double E0_m, double alpha0, double lambda_min,
                         double lambda0, Eigen::VectorXd &dC) const;

  auto EvaluateTemperatureConstrain(const Eigen::VectorXd &xPhys_col,
                                    double lambda_min, double lambda0,
                                    const std::vector<int> &v_dof)
      -> Eigen::MatrixXd;

  void optimize();

public:
  auto getRhos() -> std::vector<double> {
    int size = rhos_.size();
    std::vector<double> rhos_vec(size);
    std::copy_n(rhos_.data(), size, rhos_vec.data());
    return rhos_vec;
  }

  auto getTemperature() -> std::vector<double> {
    Eigen::VectorXd temp = sp_thermal_sim_->U_;
    int size = temp.size();
    std::vector<double> temp_vec(size);
    std::copy_n(temp.data(), size, temp_vec.data());
    return temp_vec;
  }

  auto getStress() -> std::vector<std::vector<double>> {
    int eleN = sp_therMech_sim_->GetNumEles();
    std::vector<std::vector<double>> stress_vec(eleN);
    tbb::parallel_for(0, eleN, 1, [&](int eI) {
      Eigen::VectorXd stress;
      stress.resize(6);
      auto eleId2MechDof = sp_therMech_sim_->GetMapEleId2DofsVec()[eI];
      auto eleId2TherDof = sp_thermal_sim_->GetMapEleId2DofsVec()[eI];
      Eigen::VectorXd Ue = sp_therMech_sim_->U_(eleId2MechDof);
      double Te = sp_therMech_sim_->U_(eleId2TherDof).mean();
      Eigen::MatrixXd Di = sp_therMech_sim_->GetDI(eI);
      stress =
          Di * sp_therMech_sim_->GetElementB(eI) * Ue -
          Di * sp_therMech_sim_->thermal_expansion_coefficient_ *
              (Te - sp_para_->T_ref) *
              (Eigen::VectorXd(6) << 1, 1, 1, 0, 0, 0).finished().transpose();
      stress_vec[eI].resize(3);
      std::copy_n(stress.topRows(3).data(), 3, stress_vec[eI].data());
    });
    return stress_vec;
  }

private:
  void preCompute() {
    i_dFth_dT_ =
        Eigen::KroneckerProduct(
            sp_thermal_sim_->GetMapEleId2DofsMat(),
            Eigen::VectorXi::Ones(sp_therMech_sim_->Get_DOFS_EACH_ELE()))
            .transpose()
            .reshaped();
    j_dFth_dT_ =
        Eigen::KroneckerProduct(
            sp_therMech_sim_->GetMapEleId2DofsMat(),
            Eigen::RowVectorXi::Ones(sp_thermal_sim_->Get_DOFS_EACH_ELE()))
            .transpose()
            .reshaped();

    i_dFth_drho_ =
        (Eigen::VectorXi::LinSpaced(sp_therMech_sim_->GetNumEles(), 0,
                                    sp_therMech_sim_->GetNumEles()) *
         Eigen::RowVectorXi::Ones(sp_therMech_sim_->Get_DOFS_EACH_ELE()))
            .transpose()
            .reshaped();
    j_dFth_drho_ =
        sp_therMech_sim_->GetMapEleId2DofsMat().transpose().reshaped();

    Inted_ = sp_therMech_sim_->GetD0() / sp_therMech_sim_->E_ *
             (Eigen::VectorXd(6) << 1, 1, 1, 0, 0, 0).finished();
  }
};
} // namespace da::sha
