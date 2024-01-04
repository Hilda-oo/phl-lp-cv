#include "ThermoelasticWrapper.h"
#include <igl/AABB.h>
#include <oneapi/tbb/parallel_for.h>
#include <spdlog/spdlog.h>
#include "Eigen/src/Core/Matrix.h"
#include "igl/in_element.h"
#include "mma/MMASolver.h"
#include "sha-io-foundation/data_io.h"
#include "sha-simulation-utils/io_utils.h"
#include "sha-topology-optimization-3d/Util.h"
#include "util.h"

namespace da::sha {
void ThermoelasticWrapper::simulate(Eigen::VectorXd p_rhos, Eigen::VectorXd &dc_drho,
                                    Eigen::MatrixXd &dT_drho, double &C) {
  // int size1 = p_rhos.size();
  // std::vector<double> p_rhos_vec(size1);
  // std::copy_n(p_rhos.data(), size1, p_rhos_vec.data());
  // fs_path p_rhos_vtk_path =
  //     "/home/oo/project/vscode/designauto/exe/Workplace/results/thermo-elastic/cube/rho.vtk";
  // WriteTetVTK(p_rhos_vtk_path.string(), sp_thermal_sim_->TV_, sp_thermal_sim_->TT_, p_rhos_vec);
  // spdlog::debug("write T norm vtk to: {}", p_rhos_vtk_path.c_str());
  // WriteVectorToFile(
  //     "/home/oo/project/vscode/designauto/exe/Workplace/results/thermo-elastic/cube/rho.txt",
  //     p_rhos);
  // preCompute
  double change     = 1.0;
  double E0_m       = sp_therMech_sim_->E_;
  double E_min      = E0_m * sp_para_->E_factor;
  double lambda0    = sp_thermal_sim_->thermal_conductivity_;
  double lambda_min = lambda0 * sp_para_->E_factor;
  double alpha0     = sp_therMech_sim_->thermal_expansion_coefficient_;
  double alpha_min  = alpha0 * sp_para_->E_factor;
  std::vector<int> v_dof(sp_thermal_sim_->set_dofs_to_load_.begin(),
                         sp_thermal_sim_->set_dofs_to_load_.end());
  spdlog::debug("end Precompute");
#ifndef MECH_ONLY
  // process material interpolation before the simulation
  sp_thermal_sim_->setMatrialTC(CalLambda_Vec(p_rhos, lambda_min, lambda0, sp_para_->R_lambda));
#endif
  Eigen::VectorXd p_E_vec    = CalE_Vec(p_rhos, E_min, E0_m, sp_para_->R_E);
  Eigen::VectorXd p_beta_vec = CalBeta_Vec(p_rhos, E0_m, alpha0, sp_para_->R_beta);
  sp_therMech_sim_->setMaterialParam(p_E_vec, p_beta_vec);
// simulation
#ifndef MECH_ONLY
  sp_thermal_sim_->simulation();
  Eigen::VectorXd T = sp_thermal_sim_->U_;
  int size          = T.size();
  std::vector<double> T_vec(size);
  std::copy_n(T.data(), size, T_vec.data());
  fs_path T_vtk_path =
      "/home/oo/project/vscode/designauto/exe/Workplace/results/thermo-elastic/T.vtk";
  WriteTetVTK(T_vtk_path.string(), sp_thermal_sim_->TV_, sp_thermal_sim_->TT_, {}, T_vec);
  spdlog::debug("write T norm vtk to: {}", T_vtk_path.c_str());
  //// Fth
  sp_therMech_sim_->preCompute(T, sp_para_->T_ref, sp_thermal_sim_->eDof_);

  spdlog::info("||Fth|| / ||Fm||: {}, ||Fm||: {}, ||Fth||: {}",
               sp_therMech_sim_->Fth_.norm() / sp_therMech_sim_->load_.norm(),
               sp_therMech_sim_->load_.norm(), sp_therMech_sim_->Fth_.norm());
#endif

  sp_therMech_sim_->simulation();

  // sensitivity
  // compliance
  C = EvaluateEnergyC(p_rhos, E_min, E0_m, alpha0, lambda_min, lambda0, dc_drho);
  // constrain
#ifndef MECH_ONLY
  // dT_drho
  dT_drho = EvaluateTemperatureConstrain(p_rhos, lambda_min, lambda0, v_dof);
#endif
  rhos_ = p_rhos;
}

double ThermoelasticWrapper::EvaluateEnergyC(const Eigen::VectorXd &p_rhos, double E_min,
                                             double E0_m, double alpha0, double lambda_min,
                                             double lambda0, Eigen::VectorXd &dC) const {
  spdlog::debug("compute dCdH");
  Eigen::VectorXd ce(sp_therMech_sim_->nEle_);
  oneapi::tbb::parallel_for(0, static_cast<int>(sp_therMech_sim_->nEle_), 1, [&](int i) {
    // for (int i = 0; i < sp_therMech_sim_->nEle_; ++i) {
    Eigen::VectorXd Ue = sp_therMech_sim_->U_(sp_therMech_sim_->eDof_[i]);
    ce(i)              = Ue.transpose() * sp_therMech_sim_->eleKe_[i] * Ue;
    //会导致运行时间变长
    // Eigen::VectorXi dofs_in_ele_i = sp_therMech_sim_->GetMapEleId2DofsVec()[i];
    // Eigen::VectorXd Ue            = sp_therMech_sim_->U_(dofs_in_ele_i);
    // ce(i)                         = Ue.transpose() * sp_therMech_sim_->GetElementK(i) * Ue;
    // }
  });

  spdlog::debug("sum");
  // double energy_C = sp_therMech_sim_->F_all_.dot(sp_therMech_sim_->U_);
  double energy_C = ce.sum();
  spdlog::debug("sum:{}", energy_C);
  // lambda_m
  Eigen::VectorXd lambda_m = -sp_therMech_sim_->U_;
  // dFth_drho
  Eigen::SparseMatrix<double> dFth_drho(sp_therMech_sim_->nEle_, sp_therMech_sim_->nDof_);
  spdlog::debug("每个元素dFth_drho");
  //每个元素dFth_drho 12x1
  Eigen::VectorXd v_dFth_drho(i_dFth_drho_.size());
  for (int i = 0; i < sp_thermal_sim_->nEle_; ++i) {
    // Eigen::VectorXi dofs_th       = sp_thermal_sim_->GetMapEleId2DofsVec()[i];
    // double Te                     = sp_thermal_sim_->U_(dofs_th).mean();
    // Eigen::VectorXd ele_dFth_drho = CalDBetaDrho(p_rhos(i), E0_m, alpha0, sp_para_->R_beta) *
    //                                 (Te - sp_para_->T_ref) *
    //                                 sp_therMech_sim_->GetElementB(i).transpose() * Inted_;  //
    //                                 12x1
    double Te                     = sp_thermal_sim_->U_(sp_thermal_sim_->eDof_[i]).mean();
    Eigen::VectorXd ele_dFth_drho = CalDBetaDrho(p_rhos(i), E0_m, alpha0, sp_para_->R_beta) *
                                    (Te - sp_para_->T_ref) *
                                    sp_therMech_sim_->eleBe_[i].transpose() * Inted_;  // 12x1
    assert(ele_dFth_drho.size() == 12);
    v_dFth_drho(Eigen::seqN(i * ele_dFth_drho.size(), ele_dFth_drho.size())) = ele_dFth_drho;
  }
  auto v_dFth_drho_tri = Vec2Triplet(i_dFth_drho_, j_dFth_drho_, v_dFth_drho);
  dFth_drho.setFromTriplets(v_dFth_drho_tri.begin(), v_dFth_drho_tri.end());

  // dFth^T_dT
  spdlog::debug("dFth^T_dT");
  Eigen::SparseMatrix<double> dFth_dT(sp_thermal_sim_->nDof_, sp_therMech_sim_->nDof_);
  Eigen::VectorXd v_dFth_dT(i_dFth_dT_.size());
  for (int i = 0; i < sp_thermal_sim_->nEle_; ++i) {
    double beta_rho             = CalBeta(p_rhos(i), E0_m, alpha0, sp_para_->R_beta);
    Eigen::MatrixXd ele_dFth_dT = Eigen::VectorXd::Ones(sp_thermal_sim_->eleDofNum_) * 1.0 /
                                  sp_thermal_sim_->eleDofNum_ * beta_rho *
                                  (sp_therMech_sim_->eleBe_[i].transpose() * Inted_).transpose();
    assert(ele_dFth_dT.rows() == 4 && ele_dFth_dT.cols() == 12);
    v_dFth_dT(Eigen::seqN(i * ele_dFth_dT.size(), ele_dFth_dT.size())) = ele_dFth_dT.reshaped();
  }
  auto v_dFth_dT_tri = Vec2Triplet(i_dFth_dT_, j_dFth_dT_, v_dFth_dT);
  dFth_dT.setFromTriplets(v_dFth_dT_tri.begin(), v_dFth_dT_tri.end());

  // lambda_t
  spdlog::debug("lambda_t");
  Eigen::VectorXd rhs = dFth_dT * 2 * lambda_m;
  for (auto dof_value : sp_thermal_sim_->v_dofs_to_set_) {
    auto [dof, value] = dof_value;
    rhs(dof)          = sp_thermal_sim_->K_spMat_.coeff(dof, dof) * value;
  }
  Eigen::VectorXd lambda_t = sp_thermal_sim_->solver_.solve(rhs);

  // lambda_m_Mul_dKm_drho_Mul_U
  spdlog::debug("lambda_m_Mul_dKm_drho_Mul_U");
  Eigen::VectorXd lambda_m_Mul_dKm_drho_Mul_U =
      -CalDEDrho_Vec(p_rhos, E_min, E0_m, sp_para_->R_E).array() * ce.array() /
      sp_therMech_sim_->material_int_E_.array();

  // lambda_t_Mul_dKt_drho_Mul_T
  spdlog::debug("lambda_t_Mul_dKt_drho_Mul_T");
  Eigen::VectorXd lambda_t_Mul_Kt_Mul_Te(sp_thermal_sim_->nEle_);
  for (int i = 0; i < sp_thermal_sim_->nEle_; ++i) {
    Eigen::VectorXd Te         = sp_thermal_sim_->U_(sp_thermal_sim_->eDof_[i]);
    Eigen::VectorXd lambda_t_e = lambda_t(sp_thermal_sim_->eDof_[i]);
    lambda_t_Mul_Kt_Mul_Te(i)  = lambda_t_e.transpose() * sp_thermal_sim_->eleKe_[i] * Te;
    // too slow
    //  Eigen::VectorXi dofs_in_ele_i = sp_thermal_sim_->GetMapEleId2DofsVec()[i];
    //  Eigen::VectorXd Te            = sp_thermal_sim_->U_(dofs_in_ele_i);
    //  Eigen::VectorXd lambda_t_e    = lambda_t(dofs_in_ele_i);
    //  lambda_t_Mul_Kt_Mul_Te(i)     = lambda_t_e.transpose() * sp_thermal_sim_->GetElementK(i) *
    //  Te;
  }
  Eigen::VectorXd lambda_t_Mul_dKt_drho_Mul_T =
      CalDlambdaDrho_Vec(p_rhos, lambda_min, lambda0, sp_para_->R_lambda).array() *
      lambda_t_Mul_Kt_Mul_Te.array() / sp_thermal_sim_->material_int_TC_.array();

  dC = lambda_t_Mul_dKt_drho_Mul_T + lambda_m_Mul_dKm_drho_Mul_U +
       2 * Eigen::VectorXd(dFth_drho * sp_therMech_sim_->U_);
  return energy_C;
}

auto ThermoelasticWrapper::EvaluateTemperatureConstrain(const Eigen::VectorXd &p_rhos,
                                                        double lambda_min, double lambda0,
                                                        const std::vector<int> &v_dof)
    -> Eigen::MatrixXd {
  spdlog::debug("compute dTdH");
  //      dofs of limited T
  std::map<int, std::vector<LimitedDof>> map_ele2Limit;
  Eigen::MatrixXi ele2dof_map = sp_thermal_sim_->GetMapEleId2DofsMat();
  //      loop ele2dof_map
  spdlog::debug("loop ele2dof_map");
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
  // auto CalDKth_Mul_T_For_DTi_Drhoj = [&](int rho_i) -> Eigen::SparseVector<double> {
  //   Eigen::VectorXi dofs_in_ele = sp_thermal_sim_->eDof_[rho_i];
  //   Eigen::VectorXd ele_T       = sp_thermal_sim_->U_(dofs_in_ele);
  //   Eigen::SparseVector<double> sp_dKth_Mul_T(sp_thermal_sim_->nDof_);
  //   Eigen::VectorXd dKe_th_Mul_T =
  //       CalDlambdaDrho(p_rhos(rho_i), lambda_min, lambda0, sp_para_->R_lambda) *
  //       sp_thermal_sim_->eleKe_[rho_i] / sp_thermal_sim_->material_int_TC_(rho_i) * ele_T;

  //   //too slow
  //   // Eigen::VectorXi dofs_in_ele = sp_thermal_sim_->GetMapEleId2DofsVec()[rho_i];
  //   // Eigen::VectorXd ele_T       = sp_thermal_sim_->U_(dofs_in_ele);
  //   // Eigen::SparseVector<double> sp_dKth_Mul_T(sp_thermal_sim_->nDof_);
  //   // Eigen::VectorXd dKe_th_Mul_T =
  //   //     CalDlambdaDrho(p_rhos(rho_i), lambda_min, lambda0, sp_para_->R_lambda) *
  //   //     sp_thermal_sim_->GetElementK(rho_i) / sp_thermal_sim_->material_int_TC_(rho_i) *
  //   ele_T; for (int i = 0; i < dofs_in_ele.size(); ++i) {
  //     sp_dKth_Mul_T.coeffRef(dofs_in_ele(i)) = dKe_th_Mul_T(i);
  //   }
  //   return sp_dKth_Mul_T;
  // };
  // auto CalDTi_Drhoj = [&](int Tdof, const Eigen::SparseVector<double> &sp_dKth_Mul_T) -> double {
  //   Eigen::VectorXd Li = Eigen::VectorXd::Zero(sp_dKth_Mul_T.rows());
  //   Li(Tdof)           = -1;
  //   for (auto dof_value : sp_thermal_sim_->v_dofs_to_set_) {
  //     auto [dof, value] = dof_value;
  //     Li(dof)           = sp_thermal_sim_->K_spMat_.coeff(dof, dof) * value;
  //   }
  //   Eigen::VectorXd lambda_i = sp_thermal_sim_->solver_.solve(Li);
  //   return lambda_i.transpose() * sp_dKth_Mul_T;
  // };
  Eigen::MatrixXd dT =
      Eigen::MatrixXd::Zero(sp_thermal_sim_->nEle_, sp_thermal_sim_->set_dofs_to_load_.size());
  spdlog::debug("loop dT");
  // for (auto it = map_ele2Limit.begin(); it != map_ele2Limit.end(); ++it) {
  //   auto [ele_id, v_limited] = *it;
  //   auto sp_dKth_Mul_T       = CalDKth_Mul_T_For_DTi_Drhoj(ele_id);
  //   for (auto &limited : v_limited) {
  //     dT(ele_id, limited.idx_of_load_dof) = CalDTi_Drhoj(limited.dof, sp_dKth_Mul_T);
  //   }
  // }

  for (auto it = map_ele2Limit.begin(); it != map_ele2Limit.end(); ++it) {
    auto [ele_id, v_limited]    = *it;
    Eigen::VectorXi dofs_in_ele = sp_thermal_sim_->eDof_[ele_id];
    Eigen::VectorXd ele_T       = sp_thermal_sim_->U_(dofs_in_ele);
    Eigen::SparseVector<double> sp_dKth_Mul_T(sp_thermal_sim_->nDof_);
    Eigen::VectorXd dKe_th_Mul_T =
        CalDlambdaDrho(p_rhos(ele_id), lambda_min, lambda0, sp_para_->R_lambda) *
        sp_thermal_sim_->eleKe_[ele_id] / sp_thermal_sim_->material_int_TC_(ele_id) * ele_T;
    for (int i = 0; i < dofs_in_ele.size(); ++i) {
      sp_dKth_Mul_T.coeffRef(dofs_in_ele(i)) = dKe_th_Mul_T(i);
    }
    for (auto &limited : v_limited) {
      Eigen::VectorXd Li = Eigen::VectorXd::Zero(sp_dKth_Mul_T.rows());
      Li(limited.dof)    = -1;
      for (auto dof_value : sp_thermal_sim_->v_dofs_to_set_) {
        auto [dof, value] = dof_value;
        Li(dof)           = sp_thermal_sim_->K_spMat_.coeff(dof, dof) * value;
      }
      Eigen::VectorXd lambda_i            = sp_thermal_sim_->solver_.solve(Li);
      dT(ele_id, limited.idx_of_load_dof) = lambda_i.transpose() * sp_dKth_Mul_T;
    }
  }
  return dT;
}

auto ThermoelasticWrapper::getCellStress() -> std::vector<std::vector<double>> {
  int eleN = sp_therMech_sim_->nEle_;
  std::vector<std::vector<double>> stress_vec(eleN);
  tbb::parallel_for(0, eleN, 1, [&](int eI) {
    Eigen::VectorXd stress;
    stress.resize(6);
    // auto eleId2MechDof = sp_therMech_sim_->GetMapEleId2DofsVec()[eI];
    // auto eleId2TherDof = sp_thermal_sim_->GetMapEleId2DofsVec()[eI];
    auto eleId2MechDof = sp_therMech_sim_->eDof_[eI];
    auto eleId2TherDof = sp_thermal_sim_->eDof_[eI];
    Eigen::VectorXd Ue = sp_therMech_sim_->U_(eleId2MechDof);
    double Te          = sp_thermal_sim_->U_(eleId2TherDof).mean();
    Eigen::MatrixXd Di = sp_therMech_sim_->GetDI(eI);
    // stress             = Di * sp_therMech_sim_->GetElementB(eI) * Ue -
    //          Di * sp_therMech_sim_->thermal_expansion_coefficient_ * (Te - sp_para_->T_ref) *
    //              (Eigen::VectorXd(6) << 1, 1, 1, 0, 0, 0).finished().transpose();
    stress = Di * sp_therMech_sim_->eleBe_[eI] * Ue -
             Di * sp_therMech_sim_->thermal_expansion_coefficient_ * (Te - sp_para_->T_ref) *
                 (Eigen::VectorXd(6) << 1, 1, 1, 0, 0, 0).finished().transpose();
    stress_vec[eI].resize(3);
    std::copy_n(stress.topRows(3).data(), 3, stress_vec[eI].data());
  });
  return stress_vec;
}

auto ThermoelasticWrapper::queryStress(Eigen::MatrixXd &query_points)
    -> std::vector<Eigen::VectorXd> {
  spdlog::debug("query stress");
  int qN = static_cast<int>(query_points.rows());
  std::vector<Eigen::VectorXd> stress_query(qN);
  igl::AABB<Eigen::MatrixXd, 3> tree;
  Eigen::MatrixXi TT = sp_therMech_sim_->TT_;
  Eigen::MatrixXd TV = sp_therMech_sim_->TV_;
  tree.init(TV, TT);
  Eigen::VectorXi tetI;
  igl::in_element(TV, TT, query_points, tree, tetI);

  int unlocate_point_cnt = 0;
  tbb::parallel_for(0, qN, 1, [&](int qI) {
    Eigen::VectorXd stress;
    stress.resize(6);
    int eI = tetI(qI);
    if (eI == -1) {
      stress << 1, 1, 1, 0, 0, 0;
      unlocate_point_cnt++;
      spdlog::debug("point {} is out of tets", qI);
    } else {
      auto eleId2MechDof = sp_therMech_sim_->eDof_[eI];
      auto eleId2TherDof = sp_thermal_sim_->eDof_[eI];
      Eigen::VectorXd Ue = sp_therMech_sim_->U_(eleId2MechDof);
      double Te          = sp_therMech_sim_->U_(eleId2TherDof).mean();
      Eigen::MatrixXd Di = sp_therMech_sim_->GetDI(eI);
      stress             = Di * sp_therMech_sim_->eleBe_[eI] * Ue -
               Di * sp_therMech_sim_->thermal_expansion_coefficient_ * (Te - sp_para_->T_ref) *
                   (Eigen::VectorXd(6) << 1, 1, 1, 0, 0, 0).finished().transpose();
    }
    stress_query[qI].resize(6);
    std::copy_n(stress.data(), 6, stress_query[qI].data());
  });
  log::info("the point out of tets: {}/{}", unlocate_point_cnt, qN);
  return stress_query;
}

void ThermoelasticWrapper::extractResult(fs_path out_path) {
  fs_path base_path = out_path;
  //  extract rho (txt or vtk)
  fs_path rho_vtk_path = base_path / "rho.vtk";
  sha::WriteTetVTK(rho_vtk_path.string(), sp_thermal_sim_->TV_, sp_thermal_sim_->TT_, getRhos());
  spdlog::info("write density vtk to: {}", rho_vtk_path.c_str());

  // extract temperature(vtk)
  Eigen::VectorXd T = sp_thermal_sim_->U_;
  int size          = T.size();
  std::vector<double> T_vec(size);
  std::copy_n(T.data(), size, T_vec.data());
  fs_path T_vtk_path = base_path / "T.vtk";

  WriteTetVTK(T_vtk_path.string(), sp_thermal_sim_->TV_, sp_thermal_sim_->TT_, {}, T_vec);
  spdlog::debug("write T norm vtk to: {}", T_vtk_path.c_str());
  // extract stress field(vtk)
  auto stress = getCellStress();
  spdlog::debug("getStress");
  int nCell = stress.size();
  std::vector<double> vonStress(nCell);
  std::vector<double> xStress(nCell);
  std::vector<double> yStress(nCell);
  std::vector<double> zStress(nCell);
  tbb::parallel_for(0, nCell, 1, [&](int i) {
    double s1 = stress[i][0];
    double s2 = stress[i][1];
    double s3 = stress[i][2];
    vonStress[i] =
        std::sqrt(0.5 * (std::pow((s1 - s2), 2) + std::pow((s2 - s3), 2) + std::pow((s3 - s1), 2)));
    xStress[i] = s1;
    yStress[i] = s2;
    zStress[i] = s3;
  });

  fs_path stressVon_vtk_path = base_path / "stressVon.vtk";
  WriteTetVTK(stressVon_vtk_path.string(), sp_therMech_sim_->TV_, sp_therMech_sim_->TT_, vonStress);
  spdlog::debug("write stressVon vtk to: {}", stressVon_vtk_path.c_str());

  fs_path stressX_vtk_path = base_path / "stressX.vtk";
  WriteTetVTK(stressX_vtk_path.string(), sp_therMech_sim_->TV_, sp_therMech_sim_->TT_, xStress);
  spdlog::debug("write stressX vtk to: {}", stressX_vtk_path.c_str());

  fs_path stressY_vtk_path = base_path / "stressY.vtk";
  WriteTetVTK(stressY_vtk_path.string(), sp_therMech_sim_->TV_, sp_therMech_sim_->TT_, yStress);
  spdlog::debug("write stressY vtk to: {}", stressY_vtk_path.c_str());

  fs_path stressZ_vtk_path = base_path / "stressZ.vtk";
  WriteTetVTK(stressZ_vtk_path.string(), sp_therMech_sim_->TV_, sp_therMech_sim_->TT_, zStress);
  spdlog::debug("write stressZ vtk to: {}", stressZ_vtk_path.c_str());
}

}  // namespace da::sha