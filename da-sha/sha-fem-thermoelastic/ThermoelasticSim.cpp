#include "ThermoelasticSim.h"
#include <Eigen/src/Core/Matrix.h>
#include <oneapi/tbb/parallel_for.h>
#include "sha-base-framework/frame.h"
#include "sha-simulation-utils/io_utils.h"
#include "sha-simulation-utils/other_utils.h"
#include "sha-simulation-utils/shape_function_utils.h"

namespace da::sha {

void ThermoelasticSim::simulation() {
  computeK();
  solve();
}

void ThermoelasticSim::computeFeatures() {
  // eDof
  eDof_.resize(nEle_);
  tbb::parallel_for(0, nEle_, 1, [&](int eI) {
    Eigen::VectorXi TT_I =
        TT_.row(eI); //每个元素的点索引，计算每个元素对应的自由度的索引
    eDof_[eI](Eigen::seq(0, Eigen::last, 3)) = TT_I * 3;
    eDof_[eI](Eigen::seq(1, Eigen::last, 3)) = (TT_I * 3).array() + 1;
    eDof_[eI](Eigen::seq(2, Eigen::last, 3)) = (TT_I * 3).array() + 2;
  });

  // vNeighbor
  vNeighbor_.resize(0);
  vNeighbor_.resize(nN_);
  for (int eI = 0; eI < nEle_; ++eI) {
    const Eigen::Matrix<int, 1, 4> &eleVInd = TT_.row(eI);
    for (const auto &nI : eleVInd) {
      vNeighbor_[nI].insert(eleVInd.begin(), eleVInd.end());
    }
  }
  for (int nI = 0; nI < nN_; ++nI) {  // remove itself
    vNeighbor_[nI].erase(nI);
  }

  // vFLoc
  vFLoc_.resize(0);
  vFLoc_.resize(nN_);
  for (int eI = 0; eI < nEle_; eI++) {
    for (int _nI = 0; _nI < eleNodeNum_; ++_nI) {
      const int &nI = TT_(eI, _nI);
      vFLoc_[nI].insert(std::make_pair(eI, _nI));
    }
  }
}

void ThermoelasticSim::computeK() {  // assembly stiffness matrix
  spdlog::debug("assembly stiffness matrix");

  eleKe_.resize(nEle_);
  eleBe_.resize(nEle_);
  vol_.resize(nEle_);
  std::vector<Eigen::VectorXi> vInds(nEle_);
  tbb::parallel_for(0, nEle_, 1, [&](int eI) {
    // eleKe
    Eigen::Matrix<double, 4, 3> X = TV_(TT_.row(eI), Eigen::all);
    Eigen::Matrix<double, 12, 12> eleKe_I;
    double vol;
    ComputeKeForTet(X, D_, eleKe_I, vol);
    eleKe_[eI] = eleKe_I.array() / E_ * material_int_E_(eI);
    vol_[eI]   = vol;
    // conserve Be
    Eigen::Matrix<double, 6, 12> eleBe_I;
    ComputeBForTet(X, eleBe_I);
    eleBe_[eI] = eleBe_I;
    // vInds
    vInds[eI].resize(eleNodeNum_);
    for (int _nI = 0; _nI < eleNodeNum_; ++_nI) {
      int nI         = TT_(eI, _nI);
      vInds[eI](_nI) = isDBC_(nI) ? (-nI - 1) : nI;
    }
  });
  computeGlobalK();
}

void ThermoelasticSim::solve() {
  spdlog::debug("solve U");
  F_.resize(0, 1);
  F_.resize(Fth_.size(), 1);
  for (int i = 0; i < nDof_; i++) {
    F_.coeffRef(i, 0) = load_(i) + Fth_(i);
  }
  addDBCForLoad();
  solver_.compute(K_spMat_);
  U_ = solver_.solve(F_);
  spdlog::info("compliance Cm = {:e}", F_.col(0).dot(U_));
  // TV1 = TV + U.reshaped(3, nN).transpose();
  // Utils::writeTetVTK(outputPath + "deformed.vtk", TV1, TT);
  spdlog::info("U.mean()={:e},U.max={:e},U.min={:e}", U_.mean(), U_.maxCoeff(), U_.minCoeff());
}

void ThermoelasticSim::setBC() {
  spdlog::debug("set Boundary Conditions");

  // DBC
  int nDBC = 0;
  DBC_nI_.resize(nN_);
  isDBC_.setZero(nN_);
  v_dofs_to_set_.resize(0);
  int DBCNum = (int)DirichletBCs_.size();
  for (int nI = 0; nI < nN_; ++nI) {
    Eigen::Vector3d p = TV_.row(nI);
    for (int _i = 0; _i < DBCNum; ++_i) {
      if (DirichletBCs_[_i].inDBC(p)) {
        DBC_nI_(nDBC) = nI;
        isDBC_(nI)    = 1;
        ++nDBC;
        v_dofs_to_set_.emplace_back(std::make_pair(nI, 0));
        break;
      }
    }
  }
  DBC_nI_.conservativeResize(nDBC);
  WriteOBJ((WorkingResultDirectoryPath() / "thermo-elastic/DBCV.obj").string(),
           TV_(DBC_nI_, Eigen::all), Eigen::VectorXi::LinSpaced(nDBC, 0, nDBC - 1));

  // NBC
  load_.resize(0);
  load_.setZero(nDof_);

  int nNBC = 0;
  Eigen::VectorXi NBC_nI(nN_);
  int NBCNum = (int)NeumannBCs_.size();
  for (int nI = 0; nI < nN_; ++nI) {
    Eigen::Vector3d p = TV_.row(nI);
    for (int _i = 0; _i < NBCNum; ++_i) {
      if (NeumannBCs_[_i].inNBC(p)) {
        load_.segment<TEDIM_>(nI * TEDIM_) += NeumannBCs_[_i].force;
        NBC_nI(nNBC) = nI;
        ++nNBC;

        break;
      }
    }
  }
  NBC_nI.conservativeResize(nNBC);
  spdlog::debug("result path:{}", (WorkingResultDirectoryPath() / "NBCV.obj").string());
  WriteOBJ((WorkingResultDirectoryPath() / "thermo-elastic/NBCV.obj").string(),
           TV_(NBC_nI, Eigen::all), Eigen::VectorXi::LinSpaced(nNBC, 0, nNBC - 1));

  spdlog::debug("#DBC nodes: {}, #NBC particles: {}", nDBC, nNBC);

  // ensure (DBC intersect NBC) = (empty)
  // for (const int &nI : DBC_nI_) {
  //   load_.segment<TEDIM_>(nI * TEDIM_).setZero();
  // }
}

void ThermoelasticSim::preCompute(
    Eigen::VectorXd p_T, double p_T_ref,
    std::vector<Eigen::Vector<int, 4 * HEATDIM_>> p_elementId2ThermDofs) {
  computeFth(p_T, p_T_ref, p_elementId2ThermDofs);
}

void ThermoelasticSim::computeFth(
    Eigen::VectorXd p_T, double p_T_ref,
    std::vector<Eigen::Vector<int, 4 * HEATDIM_>> p_elementId2ThermDofs) {
  Fth_.resize(0);
  Fth_.resize(nDof_);
  Fth_.setZero();
  tbb::parallel_for(0, nEle_, 1, [&](int eI) {
    // F^e_th=beta*Sum_e(B^TD0[1 1 1 0 0 0]^T)*(Te-T0)
    Eigen::Matrix<double, 4, 3> X = TV_(TT_.row(eI), Eigen::all);
    Eigen::Matrix<double, 6, 12> B;
    B.setZero();
    ComputeBForTet(X, B);
    Eigen::VectorXi dofs_th = p_elementId2ThermDofs[eI];
    Eigen::VectorXi dofs_m  = eDof_[eI];
    double Te               = p_T(dofs_th).mean();
    double beta_rho         = Beta_(eI);
    Eigen::MatrixXd D0      = D_.array() / E_;
    Fth_(dofs_m) += beta_rho * (Te - p_T_ref) * B.transpose() * D0 *
                    (Eigen::VectorXd(6) << 1, 1, 1, 0, 0, 0).finished();
  });
}
}  // namespace da::sha