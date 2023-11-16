//
// Created by cflin on 4/20/23.
//

#include "Top3d.h"
#include <spdlog/spdlog.h>
#include <cassert>
#include "Eigen/src/Core/Matrix.h"
#include "sha-topology-optimization-3d/Util.h"
namespace da::sha {
namespace top {
Tensor3d Top3d::TopOptMainLoop() {
  Eigen::VectorXd xPhys_col(sp_mesh_->GetNumEles());
  Eigen::VectorXi chosen_ele_id(sp_mesh_->GetChosenEleIdx());
  bool flg_chosen = chosen_ele_id.size() != 0;
  if (!flg_chosen) {
    // no part chosen
    xPhys_col.setConstant(sp_para_->volfrac);
  } else {
    // pick chosen part to sim
    xPhys_col = sp_mesh_->GetInitEleRho();
    xPhys_col(chosen_ele_id).setConstant(sp_para_->volfrac);
  }

  int loop      = 0;
  double change = 1.0;
  double E0     = sp_material_->YM;
  double Emin   = sp_material_->YM * sp_para_->E_factor;

  // precompute
  Eigen::VectorXd dv(sp_mesh_->GetNumEles());
  dv.setOnes();
  dv = H_ * (dv.array() / Hs_.array()).matrix().eval();

  Eigen::VectorXd ele_to_write =
      Eigen::VectorXd::Zero(sp_mesh_->GetLx() * sp_mesh_->GetLy() * sp_mesh_->GetLz());
  Eigen::VectorXi pixel_idx = sp_mesh_->GetPixelIdx();
  spdlog::info("end precompute");
  //        // clear output dir
  //        clear_dir(CMAKE_SOURCE_DIR "/output");

#ifdef USE_SUITESPARSE
  spdlog::info("using suitesparse solver");
#else
  spdlog::warn("using Eigen built-in direct solver!");
#endif
  // start iteration
  while (change > sp_para_->tol_x && loop < sp_para_->max_loop) {
    ++loop;
    Eigen::VectorXd sK =
        (sKe_ * (Emin + xPhys_col.array().pow(sp_para_->penal) * (E0 - Emin)).matrix().transpose())
            .reshaped();
    auto v_tri = Vec2Triplet(iK_, jK_, sK);
    K_.setFromTriplets(v_tri.begin(), v_tri.end());
    IntroduceFixedDofs(K_, F_);
#ifdef USE_SUITESPARSE
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
#else
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
#endif
    solver.compute(K_);
    U_ = solver.solve(F_);
    // compliance
    Eigen::VectorXd ce(sp_mesh_->GetNumEles());
    for (int i = 0; i < sp_mesh_->GetNumEles(); ++i) {
      Eigen::VectorXi dofs_in_ele_i = sp_mesh_->MapEleId2Dofs(i);
      Eigen::VectorXd Ue            = U_(dofs_in_ele_i);
      ce(i)                         = Ue.transpose() * Ke_ * Ue;
    }
    double c =
        ce.transpose() * (Emin + xPhys_col.array().pow(sp_para_->penal) * (E0 - Emin)).matrix();
    double v = flg_chosen ? xPhys_col(chosen_ele_id).sum() : xPhys_col.sum();

    Eigen::VectorXd dc =
        -sp_para_->penal * (E0 - Emin) * xPhys_col.array().pow(sp_para_->penal - 1.0) * ce.array();

    // mma solver
    size_t num_constrants             = 1;
    size_t num_variables              = flg_chosen ? chosen_ele_id.size() : sp_mesh_->GetNumEles();
    auto mma                          = std::make_shared<MMASolver>(num_variables, num_constrants);
    Eigen::VectorXd variables_tmp     = flg_chosen ? xPhys_col(chosen_ele_id) : xPhys_col;
    double f0val                      = c;
    Eigen::VectorXd df0dx             = flg_chosen
                                            ? dc(chosen_ele_id).eval() / dc(chosen_ele_id).cwiseAbs().maxCoeff()
                                            : dc / dc.cwiseAbs().maxCoeff();
    double fval                       = v - num_variables * sp_para_->volfrac;
    Eigen::VectorXd dfdx              = flg_chosen ? dv(chosen_ele_id) : dv;
    static Eigen::VectorXd low_bounds = Eigen::VectorXd::Zero(num_variables);
    static Eigen::VectorXd up_bounds  = Eigen::VectorXd::Ones(num_variables);

    //                spdlog::info("mma update");
    mma->Update(variables_tmp.data(), df0dx.data(), &fval, dfdx.data(), low_bounds.data(),
                up_bounds.data());
    if (flg_chosen) {
      change                   = (variables_tmp - xPhys_col(chosen_ele_id)).cwiseAbs().maxCoeff();
      xPhys_col(chosen_ele_id) = variables_tmp;
    } else {
      change    = (variables_tmp - xPhys_col).cwiseAbs().maxCoeff();
      xPhys_col = variables_tmp;
    }

    spdlog::critical("Iter: {:3d}, Comp: {:.3e}, Vol: {:.2f}, Change: {:f}", loop, c, v, change);
#ifdef WRITE_TENSOR_IN_LOOP
    // extract vtk
    ele_to_write(pixel_idx) = xPhys_col;
    Tensor3d ten_xPhys_to_write(sp_mesh_->GetLx() * sp_mesh_->GetLy() * sp_mesh_->GetLz(), 1, 1);
    for (int i = 0; i < ele_to_write.size(); ++i) {
      ten_xPhys_to_write(i, 0, 0) = ele_to_write(i);
    }
    ten_xPhys_to_write = ten_xPhys_to_write.reshape(Eigen::array<Eigen::DenseIndex, 3>{
        sp_mesh_->GetLx(), sp_mesh_->GetLy(), sp_mesh_->GetLz()});
    top::WriteTensorToVtk(
        da::WorkingResultDirectoryPath() / ("field_matrix" + std::to_string(loop) + ".vtk"),
        ten_xPhys_to_write, sp_mesh_);
#endif
  }
  // result
  rho_ = xPhys_col;
  // set 0 to rho of unchosen part
  assert(xPhys_col.size());
  Eigen::VectorXi continue_idx =
      Eigen::VectorXi::LinSpaced(xPhys_col.size(), 0, xPhys_col.size() - 1);
  Eigen::VectorXi unchosen_idx =flg_chosen? SetDifference(continue_idx, chosen_ele_id): Eigen::VectorXi();
  {
    xPhys_col(unchosen_idx).setZero();
    ele_to_write(pixel_idx) = xPhys_col;
    Tensor3d ten_xPhys_to_write(sp_mesh_->GetLx() * sp_mesh_->GetLy() * sp_mesh_->GetLz(), 1, 1);
    for (int i = 0; i < ele_to_write.size(); ++i) {
      ten_xPhys_to_write(i, 0, 0) = ele_to_write(i);
    }
    ten_xPhys_to_write     = ten_xPhys_to_write.reshape(Eigen::array<Eigen::DenseIndex, 3>{
        sp_mesh_->GetLx(), sp_mesh_->GetLy(), sp_mesh_->GetLz()});
    rho_field_zero_filled_ = ten_xPhys_to_write;
  }

  {
    xPhys_col(unchosen_idx).setOnes();
    ele_to_write(pixel_idx) = xPhys_col;
    Tensor3d ten_xPhys_to_write(sp_mesh_->GetLx() * sp_mesh_->GetLy() * sp_mesh_->GetLz(), 1, 1);
    for (int i = 0; i < ele_to_write.size(); ++i) {
      ten_xPhys_to_write(i, 0, 0) = ele_to_write(i);
    }
    ten_xPhys_to_write    = ten_xPhys_to_write.reshape(Eigen::array<Eigen::DenseIndex, 3>{
        sp_mesh_->GetLx(), sp_mesh_->GetLy(), sp_mesh_->GetLz()});
    rho_field_one_filled_ = ten_xPhys_to_write;
  }

  return rho_field_zero_filled_;
}

std::vector<Tensor3d> Top3d::GetTensorOfStress(const Eigen::VectorXd &which_col_of_stress) {
  Eigen::VectorXd ele_to_write =
      Eigen::VectorXd::Zero(sp_mesh_->GetLx() * sp_mesh_->GetLy() * sp_mesh_->GetLz());
  Eigen::VectorXi pixel_idx = sp_mesh_->GetPixelIdx();
  // stress
  Eigen::MatrixXd mat_stress(sp_mesh_->GetNumEles(), 6);
  Eigen::Matrix<double, 6, 24> B;
  sp_material_->computeB(0.5, 0.5, 0.5, {0, 0, 0}, B);
  for (int i = 0; i < sp_mesh_->GetNumEles(); ++i) {
    Eigen::VectorXi dofs_in_ele_i = sp_mesh_->MapEleId2Dofs(i);
    Eigen::VectorXd Ue            = U_(dofs_in_ele_i);
    mat_stress.row(i)             = rho_(i) * sp_material_->D * B * Ue;
  }
  // fill
  std::vector<Tensor3d> vt;
  for (int i = 0; i < which_col_of_stress.size(); ++i) {
    ele_to_write(pixel_idx) = mat_stress.col(which_col_of_stress(i));
    vt.push_back(GetTensorFromCol(ele_to_write));
  }
  return vt;
}

Tensor3d Top3d::GetTensorFromCol(const Eigen::VectorXd &proprty_col) {
  Tensor3d ten_prop_to_write(sp_mesh_->GetLx() * sp_mesh_->GetLy() * sp_mesh_->GetLz(), 1, 1);
  assert(proprty_col.size() == ten_prop_to_write.size());
  for (int i = 0; i < proprty_col.size(); ++i) {
    ten_prop_to_write(i, 0, 0) = proprty_col(i);
  }
  ten_prop_to_write = ten_prop_to_write.reshape(
      Eigen::array<Eigen::DenseIndex, 3>{sp_mesh_->GetLx(), sp_mesh_->GetLy(), sp_mesh_->GetLz()});
  return ten_prop_to_write;
}

void Top3d::Precompute() {
  Eigen::MatrixXi mat_ele2dofs = sp_mesh_->GetEleId2DofsMap();
  int dofs_each_ele            = sp_mesh_->Get_NUM_NODES_EACH_ELE() *
                      sp_mesh_->Get_DOFS_EACH_NODE();  // 24 for mathe; 8 for heat
  iK_ = Eigen::KroneckerProduct(mat_ele2dofs, Eigen::VectorXi::Ones(dofs_each_ele))
            .transpose()
            .reshaped();
  jK_ = Eigen::KroneckerProduct(mat_ele2dofs, Eigen::RowVectorXi::Ones(dofs_each_ele))
            .transpose()
            .reshaped();
  if (sp_mesh_->Get_DOFS_EACH_NODE() == 1) {
    Ke_ = sp_material_->computeHeatKe(0.5, 0.5, 0.5, sp_material_->k0);
  } else if (sp_mesh_->Get_DOFS_EACH_NODE() == 3) {
    // mathetical condition
    Eigen::Matrix<double, 24, 24> Ke;  // TODO: fixme
    sp_material_->computeKe(0.5, 0.5, 0.5, sp_material_->D, Ke);
    Ke_ = Ke;
  } else {
    spdlog::warn("wrong dofs!");
    exit(-1);
  }

  sKe_ = Ke_.reshaped();

  // precompute filter
  Eigen::VectorXi iH = Eigen::VectorXi::Ones(
      sp_mesh_->GetNumEles() * std::pow(2.0 * (std::ceil(sp_para_->r_min) - 1.0) + 1, 3));
  Eigen::VectorXi jH = iH;
  Eigen::VectorXd sH(iH.size());
  sH.setZero();
  int cnt = 0;
  Hs_     = Eigen::VectorXd(sp_mesh_->GetNumEles());

  int delta = std::ceil(sp_para_->r_min) - 1;
  for (int k = 0; k < sp_mesh_->GetLz(); ++k) {
    for (int j = 0; j < sp_mesh_->GetLy(); ++j) {
      for (int i = 0; i < sp_mesh_->GetLx(); ++i) {
        int ele_id0 = sp_mesh_->MapEleCoord2Id((Eigen::MatrixXi(1, 3) << i, j, k).finished())(0);
        if (ele_id0 == -1) {
          continue;
        }
        for (int k2 = std::max(k - delta, 0); k2 <= std::min(k + delta, sp_mesh_->GetLz() - 1);
             ++k2) {
          for (int j2 = std::max(j - delta, 0); j2 <= std::min(j + delta, sp_mesh_->GetLy() - 1);
               ++j2) {
            for (int i2 = std::max(i - delta, 0); i2 <= std::min(i + delta, sp_mesh_->GetLx() - 1);
                 ++i2) {
              int ele_id1 =
                  sp_mesh_->MapEleCoord2Id((Eigen::MatrixXi(1, 3) << i2, j2, k2).finished())(0);
              if (ele_id1 == -1) {
                continue;
              }
              iH(cnt) = ele_id0;
              jH(cnt) = ele_id1;
              sH(cnt) =
                  std::max(0.0, sp_para_->r_min - Eigen::Vector3d(i - i2, j - j2, k - k2).norm());
              Hs_(ele_id0) += sH(cnt);
              ++cnt;
            }
          }
        }
      }
    }
  }
  std::vector<Eigen::Triplet<double>> v_tri = Vec2Triplet(iH, jH, sH);
  H_                                        = SpMat(sp_mesh_->GetNumEles(), sp_mesh_->GetNumEles());
  H_.setFromTriplets(v_tri.begin(), v_tri.end());
}
}  // namespace top
}  // namespace da::sha