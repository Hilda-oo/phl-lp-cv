#pragma once

#include "sha-simulation-utils/material_utils.h"
#include "simulation.h"

#define TEDIM_ 3
#define HEATDIM_ 1

namespace da::sha {

class ThermoelasticSim : public Simulation<TEDIM_> {
 public:
  Eigen::Matrix<double, 6, 6> D_;  // constitutive matrix
  double thermal_conductivity_;
  double thermal_expansion_coefficient_;
  double E_;

 public:
  Eigen::VectorXd Fth_;
  std::vector<Eigen::MatrixXd> eleBe_;

 public:  // constructor
  ThermoelasticSim(Eigen::MatrixXd p_TV, Eigen::MatrixXi p_TT,
                   std::vector<DirichletBC> p_DirichletBCs, std::vector<NeumannBC> p_NeumannBCs)
      : Simulation(p_TV, p_TT, p_DirichletBCs, p_NeumannBCs) {}

  ThermoelasticSim(Eigen::MatrixXd p_TV, Eigen::MatrixXi p_TT, double p_YM, double p_PR,
                   double p_TC, double p_TEC, std::vector<DirichletBC> p_DirichletBCs,
                   std::vector<NeumannBC> p_NeumannBCs)
      : Simulation(p_TV, p_TT, p_DirichletBCs, p_NeumannBCs),
        E_(p_YM),
        thermal_conductivity_(p_TC),
        thermal_expansion_coefficient_(p_TEC) {
    ComputeElasticMatrix(p_YM, p_PR, D_);
    nDof_       = nN_ * 3;
    eleNodeNum_ = 4;
    eleDofNum_  = 12;

    setBC();

    computeFeatures();

    spdlog::info("mesh constructed");
    spdlog::info("nodes number: {}, dofs number: {}, tets element number: {}", nN_, nDof_, nEle_);
  }

  ~ThermoelasticSim() {}

  void simulation();
  void computeFeatures();
  void computeK();
  void solve();
  void setBC();

  inline auto GetElementB(int eI) -> Eigen::MatrixXd { return eleBe_[eI]; }
  inline auto GetD0() -> Eigen::Matrix<double, 6, 6> { return D_; }
  inline auto GetDI(int eI) -> Eigen::Matrix<double, 6, 6> { return D_ / E_ * material_int_E_(eI); }
  // must be run before simulation
  void preCompute(Eigen::VectorXd p_T, double p_T_ref,
                  std::vector<Eigen::Vector<int, 4 * HEATDIM_>> p_elementId2ThermDofs);

 public:
  Eigen::VectorXd material_int_E_;
  Eigen::VectorXd Beta_;

  inline void setMaterialParam(Eigen::VectorXd p_E_vec, Eigen::VectorXd p_Beta_vec) {
    material_int_E_.resize(0);
    material_int_E_.resize(p_E_vec.size());
    material_int_E_ = p_E_vec;

    Beta_.resize(0);
    Beta_.resize(p_Beta_vec.size());
    Beta_ = p_Beta_vec;
  }

 private:
  void computeFth(Eigen::VectorXd p_T, double p_T_ref,
                  std::vector<Eigen::Vector<int, 4 * HEATDIM_>> p_elementId2ThermDofs);
};
}  // namespace da::sha
