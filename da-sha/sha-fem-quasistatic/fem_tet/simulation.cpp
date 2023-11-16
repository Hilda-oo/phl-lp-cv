#include <igl/AABB.h>
#include <igl/in_element.h>
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <oneapi/tbb.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "cpt-linear-solver/cholmod_solver.h"
#include "cpt-linear-solver/eigen_solver.h"
#include "sha-base-framework/frame.h"
#include "sha-simulation-utils/io_utils.h"
#include "sha-simulation-utils/material_utils.h"
#include "sha-simulation-utils/other_utils.h"
#include "sha-simulation-utils/shape_function_utils.h"
#include "simulation.h"

namespace da::sha {

FEMTetQuasiSimulator::FEMTetQuasiSimulator(Eigen::MatrixXd p_TV, Eigen::MatrixXi p_TT,
                                           Eigen::MatrixXi p_SF, double p_YM, double p_PR,
                                           std::vector<DirichletBC> p_DirichletBCs,
                                           std::vector<NeumannBC> p_NeumannBCs)
    : TV_(std::move(p_TV)),
      TT_(std::move(p_TT)),
      SF_(std::move(p_SF)),
      DirichletBCs_(std::move(p_DirichletBCs)),
      NeumannBCs_(std::move(p_NeumannBCs)) {
  ComputeElasticMatrix(p_YM, p_PR, D_);
  nN_         = (int)TV_.rows();
  nEle_       = (int)TT_.rows();
  nDof_       = nN_ * 3;
  eleNodeNum_ = 4;
  eleDofNum_  = 12;

  // update absBBox of Boundary Conditions
  Eigen::Vector3d modelMinBBox = TV_.colwise().minCoeff();
  Eigen::Vector3d modelMaxBBox = TV_.colwise().maxCoeff();
  for (auto &DBC : DirichletBCs_) {
    DBC.calcAbsBBox(modelMinBBox, modelMaxBBox);
  }
  for (auto &NBC : NeumannBCs_) {
    NBC.calcAbsBBox(modelMinBBox, modelMaxBBox);
  }
  setBC();

  computeFeatures();
  linSysSolver_ = std::make_shared<cpt::CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd, 3>>();
  linSysSolver_->SetPattern(vNeighbor_);
  linSysSolver_->AnalyzePattern();

  spdlog::info("mesh constructed");
  spdlog::info("nodes number: {}, dofs number: {}, tets element number: {}", nN_, nDof_, nEle_);
}

void FEMTetQuasiSimulator::simulation() {
  computeK();
  solve();
}

void FEMTetQuasiSimulator::output_surf_result() {
  Eigen::MatrixXd V0_surf(SVI_.size(), 3);
  Eigen::MatrixXd V_surf(SVI_.size(), 3);
  Eigen::VectorXd U_surf(SVI_.size());
  Eigen::VectorXd stress_surf(SVI_.size());
  Eigen::VectorXd stress_surf_x(SVI_.size());
  for (int svI = 0; svI < SVI_.size(); ++svI) {
    int vI            = SVI_[svI];
    Eigen::Vector3d u = U_.segment<3>(vI * 3);
    V0_surf.row(svI)  = TV_.row(vI);
    V_surf.row(svI)   = TV_.row(vI) + u.transpose();
    U_surf(svI)       = u.norm();

    // stress
    double stress_vI = 0;
    double stress_vI_x = 0;
    int cnt          = 0;
    for (const auto &item : vFLoc_[vI]) {
      int eleI                      = item.first;
      const auto &U_eleI            = U_(eDof_[eleI]);
      Eigen::Matrix<double, 4, 3> X = TV_(TT_.row(eleI), Eigen::all);

      Eigen::Matrix<double, 6, 12> B;
      ComputeBForTet(X, B);

      stress_vI += ComputeVonStress(D_ * B * U_eleI);
      stress_vI_x += (D_ * B * U_eleI)(0);
      ++cnt;
    }
    stress_surf(svI) = stress_vI / cnt;
    stress_surf_x(svI) = stress_vI_x / cnt;
  }

  igl::write_triangle_mesh((WorkingResultDirectoryPath() / "deformed-surf.obj").string(), V_surf,
                           F_surf_);
  // Utils::writeMatrix(outputPath + "surf_U.txt", Eigen::MatrixXd(U_surf));
  // Utils::writeMatrix(outputPath + "surf_V.txt", V_surf);
  // Eigen::MatrixXi F_tmp = F_surf.array() + 1;
  // Utils::writeMatrix(outputPath + "surf_F.txt", F_tmp);
  std::vector U_surf_vec(U_surf.data(), U_surf.data() + U_surf.size());
  WriteTriVTK((WorkingResultDirectoryPath() / "dis-color-surf.vtk").string(), V0_surf, F_surf_, {},
              U_surf_vec);
  std::vector stress_surf_vec(stress_surf.data(), stress_surf.data() + stress_surf.size());
  WriteTriVTK((WorkingResultDirectoryPath() / "stress-color-surf.vtk").string(), V0_surf, F_surf_,
              {}, stress_surf_vec);
  std::vector stress_surf_x_vec(stress_surf_x.data(), stress_surf_x.data() + stress_surf_x.size());
  WriteTriVTK((WorkingResultDirectoryPath() / "stress-x-color-surf.vtk").string(), V0_surf, F_surf_,
              {}, stress_surf_x_vec);
              
}

MatMesh3 FEMTetQuasiSimulator::GetSimulatedSurfaceMesh(Eigen::MatrixXd &mat_deformed_coordinates,
                                                       Eigen::VectorXd &vtx_displacement,
                                                       Eigen::VectorXd &vtx_stress) {
  Eigen::MatrixXd V0_surf(SVI_.size(), 3);
  mat_deformed_coordinates.resize(SVI_.size(), 3);
  vtx_displacement.resize(SVI_.size());
  vtx_stress.resize(SVI_.size());
  for (int svI = 0; svI < SVI_.size(); ++svI) {
    int vI                            = SVI_[svI];
    Eigen::Vector3d u                 = U_.segment<3>(vI * 3);
    V0_surf.row(svI)                  = TV_.row(vI);
    mat_deformed_coordinates.row(svI) = TV_.row(vI) + u.transpose();
    vtx_displacement(svI)             = u.norm();

    // stress
    double stress_vI = 0;
    int cnt          = 0;
    for (const auto &item : vFLoc_[vI]) {
      int eleI                      = item.first;
      const auto &U_eleI            = U_(eDof_[eleI]);
      Eigen::Matrix<double, 4, 3> X = TV_(TT_.row(eleI), Eigen::all);

      Eigen::Matrix<double, 6, 12> B;
      ComputeBForTet(X, B);

      stress_vI += ComputeVonStress(D_ * B * U_eleI);
      ++cnt;
    }
    vtx_stress(svI) = stress_vI / cnt;
  }
  MatMesh3 surface_mesh{.mat_coordinates = V0_surf, .mat_faces = F_surf_};
  return surface_mesh;
}

void FEMTetQuasiSimulator::postprocess(Eigen::MatrixXd &Q, Eigen::VectorXd &QU,
                                       Eigen::MatrixXd &Qstress) {
  igl::AABB<Eigen::MatrixXd, 3> tree;
  tree.init(TV_, TT_);
  Eigen::VectorXi tetI;
  igl::in_element(TV_, TT_, Q, tree, tetI);

  int nQ = static_cast<int>(Q.rows());
  QU.resize(nQ);
  Qstress.resize(nQ, 6);
  for (int qI = 0; qI < nQ; ++qI) {
    int tI                        = tetI(qI);
    if (tI == -1) {
      QU(qI) = 0.0;
      Qstress.row(qI) << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;
      continue;
    }
    const auto &U_tI              = U_(eDof_[tI]);
    Eigen::Matrix<double, 4, 3> X = TV_(TT_.row(tI), Eigen::all);

    Eigen::Matrix<double, 3, 12> N;
    Eigen::Matrix<double, 6, 12> B;
    ComputeNForTet(Q.row(qI), X, N);
    ComputeBForTet(X, B);

    Eigen::Vector3d u = N * U_tI;
    QU(qI)            = u.norm();
    Q.row(qI) += u;
    Qstress.row(qI) = D_ * B * U_tI;
  }
}

void FEMTetQuasiSimulator::computeFeatures() {
  // compute F_surf
  int cnt = 0;
  std::unordered_map<int, int> vI2SVI;
  for (int sfI = 0; sfI < SF_.rows(); ++sfI) {
    for (int j = 0; j < 3; ++j) {
      const int &vI = SF_(sfI, j);
      if (!vI2SVI.count(vI)) {
        vI2SVI[vI] = cnt++;
        SVI_.conservativeResize(cnt);
        SVI_(cnt - 1) = vI;
      }
    }
  }
  F_surf_.resize(SF_.rows(), 3);
  for (int sfI = 0; sfI < SF_.rows(); ++sfI) {
    F_surf_(sfI, 0) = vI2SVI[SF_(sfI, 0)];
    F_surf_(sfI, 1) = vI2SVI[SF_(sfI, 1)];
    F_surf_(sfI, 2) = vI2SVI[SF_(sfI, 2)];
  }

  // eDof
  eDof_.resize(nEle_);
  tbb::parallel_for(0, nEle_, 1, [&](int eI) {
    Eigen::VectorXi TT_I                     = TT_.row(eI);
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

void FEMTetQuasiSimulator::computeK() {  // assembly stiffness matrix
  spdlog::info("assembly stiffness matrix");

  std::vector<Eigen::MatrixXd> eleKe(nEle_);
  std::vector<Eigen::VectorXi> vInds(nEle_);
  tbb::parallel_for(0, nEle_, 1, [&](int eI) {
    // eleKe
    Eigen::Matrix<double, 4, 3> X = TV_(TT_.row(eI), Eigen::all);
    Eigen::Matrix<double, 12, 12> eleKe_I;
    double vol;
    ComputeKeForTet(X, D_, eleKe_I, vol);
    eleKe[eI] = eleKe_I;

    // vInds
    vInds[eI].resize(eleNodeNum_);
    for (int _nI = 0; _nI < eleNodeNum_; ++_nI) {
      int nI         = TT_(eI, _nI);
      vInds[eI](_nI) = isDBC_(nI) ? (-nI - 1) : nI;
    }
  });

  linSysSolver_->SetZero();
  tbb::parallel_for(0, nN_, 1, [&](int nI) {
    for (const auto &FLocI : vFLoc_[nI]) {
      AddBlockToMatrix<DIM_>(eleKe[FLocI.first].block(FLocI.second * DIM_, 0, DIM_, eleDofNum_),
                             vInds[FLocI.first], FLocI.second, linSysSolver_);
    }
  });
}

void FEMTetQuasiSimulator::solve() {
  spdlog::info("solve");
  linSysSolver_->Factorize();
  linSysSolver_->Solve(load_, U_);

  spdlog::info("compliance C = {:e}", load_.dot(U_));
  // TV1 = TV + U.reshaped(3, nN).transpose();
  // Utils::writeTetVTK(outputPath + "deformed.vtk", TV1, TT);
}

void FEMTetQuasiSimulator::setBC() {
  spdlog::info("set Boundary Conditions");

  // DBC
  int nDBC = 0;
  DBC_nI_.resize(nN_);
  isDBC_.setZero(nN_);
  int DBCNum = (int)DirichletBCs_.size();
  for (int nI = 0; nI < nN_; ++nI) {
    Eigen::Vector3d p = TV_.row(nI);
    for (int _i = 0; _i < DBCNum; ++_i) {
      if (DirichletBCs_[_i].inDBC(p)) {
        DBC_nI_(nDBC) = nI;
        isDBC_(nI)    = 1;
        ++nDBC;

        break;
      }
    }
  }
  DBC_nI_.conservativeResize(nDBC);
  WriteOBJ((WorkingResultDirectoryPath() / "DBCV.obj").string(), TV_(DBC_nI_, Eigen::all),
           Eigen::VectorXi::LinSpaced(nDBC, 0, nDBC - 1));

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
        load_.segment<DIM_>(nI * DIM_) = NeumannBCs_[_i].force;
        NBC_nI(nNBC)                   = nI;
        ++nNBC;

        break;
      }
    }
  }
  NBC_nI.conservativeResize(nNBC);
  WriteOBJ((WorkingResultDirectoryPath() / "NBCV.obj").string(), TV_(NBC_nI, Eigen::all),
           Eigen::VectorXi::LinSpaced(nNBC, 0, nNBC - 1));

  spdlog::debug("#DBC nodes: {}, #NBC particles: {}", nDBC, nNBC);

  // ensure (DBC intersect NBC) = (empty)
  for (const int &nI : DBC_nI_) {
    load_.segment<DIM_>(nI * DIM_).setZero();
  }
}
}  // namespace da::sha