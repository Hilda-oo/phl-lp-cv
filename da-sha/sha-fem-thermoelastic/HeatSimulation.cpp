#include "HeatSimulation.h"
#include "sha-base-framework/frame.h"
#include "sha-simulation-utils/io_utils.h"
#include "sha-simulation-utils/other_utils.h"
#include "sha-simulation-utils/shape_function_utils.h"
#include <oneapi/tbb/parallel_for.h>

namespace da::sha {

void HeatSimulation::simulation() {
  computeK();
  addBCForLoad();
  solve();
}

void HeatSimulation::computeFeatures() {
  // eDof
  eDof_.resize(nEle_);
  tbb::parallel_for(0, nEle_, 1, [&](int eI) {
    Eigen::VectorXi TT_I = TT_.row(
        eI); //每个四面体的点索引，计算自由度的索引，热的结点只有一个自由度
    eDof_[eI](Eigen::seq(0, Eigen::last)) = TT_I;
  });

  // vNeighbor
  vNeighbor_.resize(0);
  vNeighbor_.resize(nN_);
  for (int eI = 0; eI < nEle_; ++eI) {
    const Eigen::Matrix<int, 1, 4> &eleVInd = TT_.row(eI);
    for (const auto &nI : eleVInd) { //遍历每个tet的每个vertex
      vNeighbor_[nI].insert(eleVInd.begin(), eleVInd.end());
    }
  }
  for (int nI = 0; nI < nN_; ++nI) { // remove itself
    vNeighbor_[nI].erase(nI);
  }

  // vFLoc
  vFLoc_.resize(0);
  vFLoc_.resize(nN_);
  for (int eI = 0; eI < nEle_; eI++) {
    for (int _nI = 0; _nI < eleNodeNum_; ++_nI) {
      const int &nI = TT_(eI, _nI); //遍历每个四面体的四个顶点索引
      vFLoc_[nI].insert(std::make_pair(eI, _nI));
    }
  }
}

void HeatSimulation::ComputeGlobalK() {
  K_ = Eigen::MatrixXd::Zero(nDof_, nDof_);
  for (int eI = 0; eI < nEle_; ++eI) {
    auto ele_id_dof = eDof_[eI];
    K_(ele_id_dof, ele_id_dof) += eleKe_[eI];
  }
  std::vector<Eigen::Triplet<double>> triplet;
  triplet.resize(0);
  for (int i = 0; i < nDof_; i++) {
    for (int j = i; j < nDof_; j++) {
      triplet.push_back({i, j, K_(i, j)});
    }
  }
  K_spMat_.resize(nDof_, nDof_);
  K_spMat_.setFromTriplets(triplet.begin(), triplet.end());
}

void HeatSimulation::computeK() { // assembly stiffness matrix
  spdlog::info("assembly stiffness matrix");

  eleKe_.resize(nEle_);
  //每个元素第 0 1 2 3 个顶点的坐标索引,如果是Dir就进行标记
  std::vector<Eigen::VectorXi> vInds(nEle_);
  tbb::parallel_for(0, nEle_, 1, [&](int eI) {
    // eleKe
    Eigen::Matrix<double, 4, 3> X = TV_(TT_.row(eI), Eigen::all);
    Eigen::Matrix<double, 4, 4> eleKe_I;
    double vol;
    ComputeHeatKeForTet(X, eleKe_I, vol);
    // eleKe_[eI] = eleKe_I * thermal_conductivity_;
    eleKe_[eI] = eleKe_I * material_int_TC_(eI);

    // vInds
    vInds[eI].resize(eleNodeNum_);
    for (int _nI = 0; _nI < eleNodeNum_; ++_nI) {
      int nI = TT_(eI, _nI);
      vInds[eI](_nI) = isDBC_(nI) ? (-nI - 1) : nI;
    }
  });
  ComputeGlobalK();
}

void HeatSimulation::solve() {
  spdlog::info("solve T");

  solver_.compute(K_spMat_);
  U_ = solver_.solve(F_);

  // Eigen::MatrixXd F = Eigen::MatrixXd::Zero(nDof_, 1);
  // for (int i = 0; i < nDof_; ++i) {
  //   F(i, 0) = F_.coeffRef(i, 0);
  // }
  // WriteMatrix(("/home/oo/project/vscode/Thermo-elasticSimulationForTet/exe/Workplace/results/Kth.txt"),
  // K_);
  // WriteMatrix("/home/oo/project/vscode/Thermo-elasticSimulationForTet/exe/Workplace/results/Fth.txt",
  // F);
  // WriteVector("/home/oo/project/vscode/Thermo-elasticSimulationForTet/exe/Workplace/results/Uth.txt",
  // U_);

  spdlog::info("compliance Cth = {:e}", F_.col(0).dot(U_));
  spdlog::info("T.mean()={:e},T.max={:e},T.min={:e}", U_.mean(), U_.maxCoeff(),
               U_.minCoeff());
}

void HeatSimulation::setBC() {
  spdlog::info("set Heat Boundary Conditions");

  // DBC
  int nDBC = 0;
  DBC_nI_.resize(nN_);
  isDBC_.setZero(nN_);
  int DBCNum = (int)DirichletBCs_.size();
  for (
      int nI = 0; nI < nN_;
      ++nI) { //遍历每个点，判断是否在Dir边界上，记录下DBC_nI_每个dir条件对应的点索引(也就可以找到对应的自由度)、isDBC_每个点是否在Dir边界上
    Eigen::Vector3d p = TV_.row(nI);
    for (int _i = 0; _i < DBCNum; ++_i) {
      int value = DirichletBCs_[_i].temperature;
      if (DirichletBCs_[_i].inDBC(p)) {
        DBC_nI_(nDBC) = nI;
        isDBC_(nI) = 1;
        ++nDBC;
        v_dofs_to_set_.emplace_back(std::make_pair(nI, value));
        break;
      }
    }
  }
  DBC_nI_.conservativeResize(nDBC);
  WriteOBJ(("/home/oo/project/vscode/Thermo-elasticSimulationForTet/exe/"
            "Workplace/results/HeatDBCV.obj"),
           TV_(DBC_nI_, Eigen::all),
           Eigen::VectorXi::LinSpaced(nDBC, 0, nDBC - 1));

  // NBC
  load_.resize(0);
  load_.setZero(nDof_);
  F_.resize(nDof_, 1);
  F_.setZero();
  int nNBC = 0;
  Eigen::VectorXi NBC_nI(nN_);
  int NBCNum = (int)NeumannBCs_.size();
  for (int nI = 0; nI < nN_; ++nI) {
    Eigen::Vector3d p = TV_.row(nI);
    for (int _i = 0; _i < NBCNum; ++_i) {
      if (NeumannBCs_[_i].inNBC(p)) {
        // load_(nI) += NeumannBCs_[_i].flux; //热里面结点索引就是结点自由度索引
        F_.coeffRef(nI, 0) += NeumannBCs_[_i].flux;
        NBC_nI(nNBC) = nI;
        ++nNBC;
        set_dofs_to_load_.insert(nI);
        break;
      }
    }
  }
  load_ /= nNBC == 0 ? 1.0 : (1.0 / nNBC);
  F_ /= nNBC == 0 ? 1.0 : (1.0 / nNBC);
  NBC_nI.conservativeResize(nNBC);
  WriteOBJ(("/home/oo/project/vscode/Thermo-elasticSimulationForTet/exe/"
            "Workplace/results/HeatNBCV.obj"),
           TV_(NBC_nI, Eigen::all),
           Eigen::VectorXi::LinSpaced(nNBC, 0, nNBC - 1));

  spdlog::debug("Heat #DBC nodes: {}, #NBC particles: {}", nDBC, nNBC);

  // // ensure (DBC intersect NBC) = (empty)
  // for (const int &nI : DBC_nI_) {
  //   load_(nI) = 0;  //热里面结点索引就是结点自由度索引
  // }
}

void HeatSimulation::addBCForLoad() {
  int DBCNum = (int)DirichletBCs_.size();
  //热的迪利克雷条件设置了固定温度
  for (
      int nI = 0; nI < nN_;
      ++nI) { //遍历每个点，判断是否在Dir边界上，记录下DBC_nI_每个dir条件对应的点索引、isDBC_每个点是否在Dir边界上
    Eigen::Vector3d p = TV_.row(nI);
    for (int _i = 0; _i < DBCNum; ++_i) {
      if (DirichletBCs_[_i].inDBC(p)) {
        K_spMat_.coeffRef(nI, nI) *= 1e7;
        K_.coeffRef(nI, nI) *= 1e7;
        // load_(nI) = K_(nI, nI) * DirichletBCs_[_i].temperature;
        F_.coeffRef(nI, 0) = K_(nI, nI) * DirichletBCs_[_i].temperature;
        break;
      }
    }
  }
}
} // namespace da::sha