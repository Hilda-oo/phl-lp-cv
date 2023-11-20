#pragma once

#include <Eigen/Eigen>
#include <boost/filesystem/path.hpp>

namespace da::sha {
using fs_path = boost::filesystem::path;

#define INIT_SOLVER(solver_name, A)                                            \
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver_name;        \
  solver_name.compute(A)

template <typename Scalar>
inline std::vector<Eigen::Triplet<Scalar>>
Vec2Triplet(const Eigen::VectorXi &I, const Eigen::VectorXi &J,
            const Eigen::Matrix<Scalar, -1, 1> &V) {
  std::vector<Eigen::Triplet<Scalar>> v_tri;
  for (int i = 0; i < I.size(); ++i) {
    v_tri.push_back({I(i), J(i), V(i)});
  }
  return v_tri;
}

inline auto CalR = [](double rho, double R) {
  return rho / (1.0 + R * (1.0 - rho));
};

// sensitivity utils
inline auto CalR_Vec(const Eigen::VectorXd &vec_rho, double R)
    -> Eigen::VectorXd {
  return vec_rho.array() / (1.0 + R * (1.0 - vec_rho.array()));
}
inline auto CalDRDrho(double rho, double R) -> double {
  double down = 1 + R * (1 - rho);
  return (1 + R) / (down * down);
}
inline auto CalDRDrho_Vec(const Eigen::VectorXd &vec_rho, double R)
    -> Eigen::VectorXd {
  auto down = 1 + R * (1 - vec_rho.array());
  return (1 + R) / (down * down);
}
inline auto CalE_Vec(const Eigen::VectorXd &vec_rho, double E_min, double E0_m,
                     double R_E) -> Eigen::VectorXd {
  return E_min + CalR_Vec(vec_rho, R_E).array() * (E0_m - E_min);
}
inline auto CalDEDrho_Vec(const Eigen::VectorXd &vec_rho, double E_min,
                          double E0_m, double R_E) -> Eigen::VectorXd {
  return CalDRDrho_Vec(vec_rho, R_E) * (E0_m - E_min);
}
inline auto CalLambda_Vec(const Eigen::VectorXd &vec_rho, double lambda_min,
                          double lambda0, double R_lambda) -> Eigen::VectorXd {
  return lambda_min +
         CalR_Vec(vec_rho, R_lambda).array() * (lambda0 - lambda_min);
}
inline auto CalDlambdaDrho(double rho, double lambda_min, double lambda0,
                           double R_lambda) -> double {
  return CalDRDrho(rho, R_lambda) * (lambda0 - lambda_min);
}
inline auto CalDlambdaDrho_Vec(const Eigen::VectorXd &vec_rho,
                               double lambda_min, double lambda0,
                               double R_lambda) -> Eigen::VectorXd {
  return CalDRDrho_Vec(vec_rho, R_lambda) * (lambda0 - lambda_min);
}

inline auto CalBeta(double rho, double E0_m, double alpha0, double R_beta)
    -> double {
  return CalR(rho, R_beta) * E0_m * alpha0;
}

inline auto CalBeta_Vec(const Eigen::VectorXd &vec_rho, double E0_m,
                        double alpha0, double R_beta) -> Eigen::VectorXd {
  return CalR_Vec(vec_rho, R_beta).array() * E0_m * alpha0;
}
inline auto CalDBetaDrho(double rho, double E0_m, double alpha0, double R_beta)
    -> double {
  return CalDRDrho(rho, R_beta) * E0_m * alpha0;
}
} // namespace da::sha