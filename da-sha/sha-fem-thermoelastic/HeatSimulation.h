#pragma once

#include "simulation.h"
#include <Eigen/src/SparseCore/SparseMatrix.h>

#define HEATDIM_ 1

namespace da::sha {

class HeatSimulation : public Simulation<HEATDIM_> {
public:
  double thermal_conductivity_;
  std::set<unsigned> set_dofs_to_load_;

public: // constructor
  HeatSimulation(Eigen::MatrixXd p_TV, Eigen::MatrixXi p_TT,
                 std::vector<DirichletBC> p_DirichletBCs,
                 std::vector<NeumannBC> p_NeumannBCs)
      : Simulation(p_TV, p_TT, p_DirichletBCs, p_NeumannBCs) {}

  HeatSimulation(Eigen::MatrixXd p_TV, Eigen::MatrixXi p_TT,
                 double p_thermal_conductivity,
                 std::vector<DirichletBC> p_DirichletBCs,
                 std::vector<NeumannBC> p_NeumannBCs)
      : Simulation(p_TV, p_TT, p_DirichletBCs, p_NeumannBCs),
        thermal_conductivity_(p_thermal_conductivity) {
    nDof_ = nN_;
    eleNodeNum_ = 4;
    eleDofNum_ = 4;

    setBC();

    computeFeatures();

    spdlog::info("mesh constructed");
    spdlog::info("nodes number: {}, dofs number: {}, tets element number: {}",
                 nN_, nDof_, nEle_);
  }

  ~HeatSimulation() {}

  void simulation();

  void computeFeatures();
  void computeK();
  void solve();
  void setBC();

public:
  Eigen::VectorXd material_int_TC_;

  void setMatrialTC(Eigen::VectorXd p_TC_vec) {
    int size = p_TC_vec.size();
    material_int_TC_.resize(0);
    material_int_TC_.resize(size);
    material_int_TC_ = p_TC_vec;
  }
};
} // namespace da::sha
