#pragma once

#include <oneapi/tbb/parallel_for.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <map>
#include <utility>
#include <vector>
#include "sha-base-framework/declarations.h"
#include "sha-fem-thermoelastic/HeatSimulation.h"
#include "sha-fem-thermoelastic/ThermoelasticSim.h"
#include "sha-simulation-3d/CBN/background_mesh.h"
#include "unsupported/Eigen/src/KroneckerProduct/KroneckerTensorProduct.h"

namespace da::sha {
struct CtrlPara {
  double E        = 0.5;
  double volfrac  = 0.5;
  double penal    = 1.0;
  int max_loop    = 100;
  double r_min    = 2.0;
  double T_ref    = 295;
  double T_limit  = 325;
  double tol_x    = 0.00001;
  double E_factor = 1e-9;
  double R_E      = 8;
  double R_lambda = 0;
  double R_beta   = 0;
};

struct LimitedDof {
  int dof;
  int idx_of_load_dof;
  int idx_in_ele;

  LimitedDof(int dof, int idx_of_load_dof, int idx_in_ele)
      : dof(dof), idx_of_load_dof(idx_of_load_dof), idx_in_ele(idx_in_ele) {}
};

class ThermoelasticWrapper {
 private:
  std::shared_ptr<HeatSimulation> sp_thermal_sim_;
  std::shared_ptr<ThermoelasticSim> sp_therMech_sim_;
  std::shared_ptr<CtrlPara> sp_para_;
  Eigen::VectorXd Inted_;
  Eigen::VectorXi i_dFth_dT_, j_dFth_dT_;
  // i:i: j:每个元素自由度的索引
  Eigen::VectorXi i_dFth_drho_, j_dFth_drho_;
  Eigen::VectorXd rhos_;

  std::vector<Eigen::VectorXi> macId2tetsId_;
  Eigen::MatrixXd node_;
  int nEle_;

  Eigen::SparseMatrix<double> neighbors_;
public:
  std::shared_ptr<NestedBackgroundMesh> nested_background_;


 public:
  ThermoelasticWrapper(const std::shared_ptr<NestedBackgroundMesh> &p_nested_background,
                       double p_YM, double p_PR, double p_TC, double p_TEC,
                       const std::vector<sha::DirichletBC> &p_mechDBC,
                       const std::vector<sha::NeumannBC> &p_mechNBC,
                       const std::vector<sha::DirichletBC> &p_thermalDBC,
                       const std::vector<sha::NeumannBC> &p_thermalNBC,
                       std::shared_ptr<CtrlPara> p_para)
      : nested_background_(p_nested_background), sp_para_(p_para) {
    nEle_ = static_cast<int>(nested_background_->nested_cells_.size());
    Eigen::MatrixXd TV;
    Eigen::MatrixXi TT;
    // get TV, TT and the relationship of tets with macro cells from nested_background_mesh_
    spdlog::debug("process NestedBackgound Mesh");
    processNestedBackgoundMesh(TV, TT);
    // spdlog::debug("process Neighbors");
    // processNeighbors(TV, TT, p_para->r_min);
    node_            = TV;
    sp_therMech_sim_ = std::make_shared<sha::ThermoelasticSim>(TV, TT, p_YM, p_PR, p_TC, p_TEC,
                                                               p_mechDBC, p_mechNBC);

    sp_thermal_sim_ =
        std::make_shared<sha::HeatSimulation>(TV, TT, p_TC, p_thermalDBC, p_thermalNBC);
    preCompute();
  }

  // Compliance
  double EvaluateEnergyC(const Eigen::VectorXd &p_rhos, double E_min, double E0_m, double alpha0,
                         double lambda_min, double lambda0, Eigen::VectorXd &dC) const;

  auto EvaluateTemperatureConstrain(const Eigen::VectorXd &p_rhos, double lambda_min,
                                    double lambda0, const std::vector<int> &v_dof)
      -> Eigen::MatrixXd;

  void simulate(Eigen::VectorXd p_rhos, Eigen::VectorXd &dc_drho, Eigen::MatrixXd &dT_drho,
                double &C);

 public:
  double ComputeAverageMicroTetEdgeLength() {
    return nested_background_->ComputeAverageMicroTetEdgeLength();
  }

  Eigen::VectorXd getSetDofThermalU() {
    std::vector<int> v_dof(sp_thermal_sim_->set_dofs_to_load_.begin(),
                           sp_thermal_sim_->set_dofs_to_load_.end());
    return sp_thermal_sim_->U_(v_dof);
  }

  auto getTemperature() -> std::vector<double> {
    Eigen::VectorXd temp = sp_thermal_sim_->U_;
    int size             = temp.size();
    std::vector<double> temp_vec(size);
    std::copy_n(temp.data(), size, temp_vec.data());
    return temp_vec;
  }

  auto getRhos() -> std::vector<double> {
    int size = rhos_.size();
    std::vector<double> rhos_vec(size);
    std::copy_n(rhos_.data(), size, rhos_vec.data());
    return rhos_vec;
  }

  auto getCellStress() -> std::vector<std::vector<double>>;

  auto queryStress(Eigen::MatrixXd &query_points) -> std::vector<Eigen::VectorXd>;

  void extractResult(fs_path out_path);

 public:
  auto getContrlPara() { return sp_para_; }

  int getMacroCellNum() { return nEle_; }

  auto getNode() { return node_; }

  auto getNestedCells() { return nested_background_->nested_cells_; }

  auto getMacIdMaptetsId() { return macId2tetsId_; }

  auto getAllTetVol() { return sp_therMech_sim_->vol_; }

 private:
  void preCompute() {
    i_dFth_dT_ =
        Eigen::KroneckerProduct(sp_thermal_sim_->GetMapEleId2DofsMat(),
                                Eigen::VectorXi::Ones(sp_therMech_sim_->Get_DOFS_EACH_ELE()))
            .transpose()
            .reshaped();
    j_dFth_dT_ =
        Eigen::KroneckerProduct(sp_therMech_sim_->GetMapEleId2DofsMat(),
                                Eigen::RowVectorXi::Ones(sp_thermal_sim_->Get_DOFS_EACH_ELE()))
            .transpose()
            .reshaped();

    i_dFth_drho_ = (Eigen::VectorXi::LinSpaced(sp_therMech_sim_->GetNumEles(), 0,
                                               sp_therMech_sim_->GetNumEles()) *
                    Eigen::RowVectorXi::Ones(sp_therMech_sim_->Get_DOFS_EACH_ELE()))
                       .transpose()
                       .reshaped();
    j_dFth_drho_ = sp_therMech_sim_->GetMapEleId2DofsMat().transpose().reshaped();

    Inted_ = sp_therMech_sim_->GetD0() / sp_therMech_sim_->E_ *
             (Eigen::VectorXd(6) << 1, 1, 1, 0, 0, 0).finished();
  }

  void processNestedBackgoundMesh(Eigen::MatrixXd &p_TV, Eigen::MatrixXi &p_TT) {
    macId2tetsId_.clear();
    auto nested_cells         = nested_background_->nested_cells_;
    const int num_macro_cells = nested_cells.size();
    std::set<std::string> vset;
    std::map<std::string, int> vertice2Id;
    std::map<int, Eigen::VectorXd> id2Vertice;
    std::vector<Eigen::VectorXi> tetVec(0);
    int vN = 0;
    int tN = 0;
    for (index_t macro_cell_idx = 0; macro_cell_idx < num_macro_cells; ++macro_cell_idx) {
      const auto &macro_cell       = nested_cells.at(macro_cell_idx);
      const Eigen::MatrixXd &micV_ = macro_cell.tetrahedrons.mat_coordinates;
      const Eigen::MatrixXi &micT_ = macro_cell.tetrahedrons.mat_tetrahedrons;
      const int num_tetrahedrons   = macro_cell.tetrahedrons.NumTetrahedrons();
      const int num_vertices       = macro_cell.tetrahedrons.NumVertices();
      std::vector<int> micId2verticeId(num_vertices);
      Eigen::VectorXi id_tets_in_mac;
      id_tets_in_mac.resize(num_tetrahedrons);
      for (index_t vI = 0; vI < num_vertices; ++vI) {
        Eigen::Vector3d vertice = micV_.row(vI);
        std::string v_str =
            std::to_string(vertice(0)) + std::to_string(vertice(1)) + std::to_string(vertice(2));
        if (!vset.count(v_str)) {
          vset.insert(v_str);
          vertice2Id.insert({v_str, vN});
          id2Vertice.insert({vN, vertice});
          ++vN;
        }
        micId2verticeId[vI] = vertice2Id.at(v_str);
      }
      for (index_t tI = 0; tI < num_tetrahedrons; ++tI) {
        Eigen::Vector4i tet = micT_.row(tI);
        Eigen::Vector4i tetIdx;
        tetIdx << micId2verticeId[tet(0)], micId2verticeId[tet(1)], micId2verticeId[tet(2)],
            micId2verticeId[tet(3)];
        tetVec.push_back(tetIdx);
        id_tets_in_mac(tI) = tN;
        ++tN;
      }
      macId2tetsId_.push_back(id_tets_in_mac);
    }
    p_TV.resize(vN, 3);
    for (index_t vI = 0; vI < vN; ++vI) {
      p_TV.row(vI) = id2Vertice.at(vI);
    }
    p_TT.resize(tN, 4);
    for (index_t tI = 0; tI < tN; ++tI) {
      p_TT.row(tI) = tetVec.at(tI);
    }
  }

  void processNeighbors(Eigen::MatrixXd p_TV, Eigen::MatrixXi p_TT, double radius) {
    int tetNum = p_TT.rows();
    neighbors_.resize(tetNum, tetNum);
    spdlog::info("tetNUm:{}", tetNum);
    
    auto ti_te_distance = [&](Eigen::VectorXi t1, Eigen::VectorXi t2) -> double {
      Eigen::VectorXd centroid1 = p_TV(t1, Eigen::all).colwise().mean();
      Eigen::VectorXd centroid2 = p_TV(t2, Eigen::all).colwise().mean();
      return (centroid1 - centroid2).cwiseAbs().norm();
    };
    oneapi::tbb::parallel_for(0, static_cast<int>(tetNum), 1, [&](int _tI) {
      
      // for (int _tI = 0; _tI < tetNum; _tI++) {
        spdlog::info("tet:{}", _tI);
      Eigen::VectorXi ti = p_TT.row(_tI);
      for (int _i = _tI + 1; _i < tetNum; _i++) {
        if (neighbors_.coeff(_tI, _i) > 0 || neighbors_.coeff(_i, _tI) > 0) {
          continue;
        }
        Eigen::VectorXi te = p_TT.row(_i);
        double distance = ti_te_distance(ti, te);
        if (distance < radius) {
          neighbors_.coeffRef(_tI, _i) = distance;
        }
      }
    });
      // }
    spdlog::debug("neighbor:{}", neighbors_.nonZeros());
  }
};
}  // namespace da::sha
