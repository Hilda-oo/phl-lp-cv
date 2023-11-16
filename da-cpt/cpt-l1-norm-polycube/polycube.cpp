#include "polycube.h"

#include <ifopt/bounds.h>
#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>
#include <ifopt/test_vars_constr_cost.h>
#include <ifopt/variable_set.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <ostream>
#include <stack>
#include <string>
#include <vector>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

namespace da {
namespace cpt {
template <typename Scalar>
void ComputeFaceAreasAndNormals(const Eigen::MatrixX<Scalar> &mat_coordinates,
                                const Eigen::MatrixXi &mat_faces,
                                Eigen::VectorX<Scalar> &face_area_values,
                                Eigen::MatrixX<Scalar> &mat_normals) {
  using VectorX          = Eigen::VectorX<Scalar>;
  using Vector3          = Eigen::Vector3<Scalar>;
  const size_t num_faces = mat_faces.rows();
  face_area_values.resize(num_faces);
  mat_normals.resize(num_faces, 3);
  for (int face_idx = 0; face_idx < num_faces; ++face_idx) {
    Vector3 vector_0_to_1 =
        mat_coordinates.row(mat_faces(face_idx, 0)) - mat_coordinates.row(mat_faces(face_idx, 1));
    Vector3 vector_0_to_2 =
        mat_coordinates.row(mat_faces(face_idx, 0)) - mat_coordinates.row(mat_faces(face_idx, 2));
    Vector3 face_normal        = vector_0_to_1.cross(vector_0_to_2);
    face_area_values(face_idx) = 0.5 * face_normal.norm();
    mat_normals.row(face_idx)  = face_normal / face_normal.norm();
  }
}

using RealType = double;
using MatrixX  = Eigen::MatrixX<RealType>;
using VectorX  = Eigen::VectorX<RealType>;
using Matrix34 = Eigen::Matrix<RealType, 3, 4>;

class CoordinateVariableSet : public ifopt::VariableSet {
 public:
  explicit CoordinateVariableSet(const MatrixX &mat_coordinates)
      : ifopt::VariableSet(mat_coordinates.size(), "coordinate"),
        num_vertices_(mat_coordinates.rows()),
        variables_(Eigen::Map<const VectorX>(mat_coordinates.data(), num_vertices_ * 3)) {}

  void SetVariables(const VectorX &variables) override { this->variables_ = variables; }

  VectorX GetValues() const override { return variables_; }

  const RealType *GetValueArray() const { return variables_.data(); }

  VecBound GetBounds() const override {
    VecBound bounds(GetRows(), ifopt::NoBound);
    return bounds;
  }

 private:
  const size_t num_vertices_;
  VectorX variables_;
};

class ConstraintTermL1 : public ifopt::ConstraintSet {
 public:
  explicit ConstraintTermL1(const Eigen::MatrixXi &mat_tetrahedrons,
                            const Eigen::MatrixXi &mat_faces, double origin_total_area,
                            size_t num_vertices)
      : ifopt::ConstraintSet(1, "Constraint"),
        mat_tetrahedrons_(mat_tetrahedrons),
        mat_triangle_faces_(mat_faces),
        origin_total_area_(origin_total_area),
        num_vertices_(num_vertices) {}

  VectorXd GetValues() const override {
    VectorXd g(GetRows());
    autodiff::VectorXvar variables =
        GetVariables()->GetComponent("coordinate")->GetValues().cast<autodiff::var>();
    Eigen::Map<autodiff::MatrixXvar> mat_polycube_coordinates(variables.data(), num_vertices_, 3);
    auto C = ComputeConstraintTermC(mat_polycube_coordinates);
    g(0)   = C.expr->val;
    return g;
  };

  VecBound GetBounds() const override {
    VecBound b(GetRows());
    b.at(0) = ifopt::BoundZero;
    return b;
  }

  void FillJacobianBlock(std::string var_set, Jacobian &jac_block) const override {
    if (var_set == "coordinate") {
      autodiff::VectorXvar variables =
          GetVariables()->GetComponent("coordinate")->GetValues().cast<autodiff::var>();
      Eigen::Map<autodiff::MatrixXvar> mat_polycube_coordinates(variables.data(), num_vertices_, 3);
      auto C             = ComputeConstraintTermC(mat_polycube_coordinates);
      VectorX gradient_C = autodiff::gradient(C, variables);
      for (int idx = 0; idx < 3 * num_vertices_; ++idx) {
        jac_block.coeffRef(0, idx) = gradient_C(idx);
      }
    }
  }

  auto ComputeConstraintTermC(Eigen::Map<autodiff::MatrixXvar> &mat_polycube_coordinates) const
      -> autodiff::var {
    autodiff::VectorXvar face_area_values;
    autodiff::MatrixXvar mat_face_normals;
    ComputeFaceAreasAndNormals<autodiff::var>(mat_polycube_coordinates, mat_triangle_faces_,
                                              face_area_values, mat_face_normals);
    autodiff::var C = face_area_values.sum() - origin_total_area_;
    return C;
  }

 private:
  const Eigen::MatrixXi &mat_triangle_faces_;
  const Eigen::MatrixXi &mat_tetrahedrons_;
  size_t num_vertices_;
  double origin_total_area_;
};

class CostTermL1 : public ifopt::CostTerm {
  using Var = autodiff::var;

 public:
  explicit CostTermL1(const Eigen::MatrixXi &mat_tetrahedrons, const Eigen::MatrixXi &mat_faces,
                      const Eigen::MatrixXi &mat_adjacent_face_pairs,
                      const std::vector<std::vector<int>> &map_face_idx_to_neighbor_indices,
                      const std::vector<Matrix34> &matrices_gradient, double origin_total_face_area,
                      const VectorXd &tetrahedron_volumes, size_t num_vertices, double alpha,
                      double beta)
      : ifopt::CostTerm("Cost"),
        mat_tetrahedrons_(mat_tetrahedrons),
        mat_triangle_faces_(mat_faces),
        mat_adjacent_face_pairs_(mat_adjacent_face_pairs),
        map_face_idx_to_neighbor_indices_(map_face_idx_to_neighbor_indices),
        matrices_gradient_operator_(matrices_gradient),
        origin_total_face_area_(origin_total_face_area),
        tetrahedron_volumes_(tetrahedron_volumes),
        num_vertices_(num_vertices),
        num_tetrahedrons_(mat_tetrahedrons.rows()),
        num_triangle_faces_(mat_triangle_faces_.rows()),
        alpha_(alpha),
        beta_(beta) {}

  double GetCost() const override {
    autodiff::VectorXvar variables =
        GetVariables()->GetComponent("coordinate")->GetValues().cast<Var>();
    Eigen::Map<autodiff::MatrixXvar> mat_polycube_coordinates(variables.data(), num_vertices_, 3);
    Var cost = ComputeCost(mat_polycube_coordinates);
    return cost.expr->val;
  };

  void FillJacobianBlock(std::string variable_label, Jacobian &mat_jacobian) const override {
    if (variable_label == "coordinate") {
      autodiff::VectorXvar variables =
          GetVariables()->GetComponent("coordinate")->GetValues().cast<autodiff::var>();
      Eigen::Map<autodiff::MatrixXvar> mat_polycube_coordinates(variables.data(), num_vertices_, 3);
      Var cost      = ComputeCost(mat_polycube_coordinates);
      auto gradient = autodiff::gradient(cost, variables);
      for (int idx = 0; idx < 3 * num_vertices_; ++idx) {
        mat_jacobian.coeffRef(0, idx) = gradient(idx);
      }
    }
  }

  auto ComputeCost(Eigen::Map<autodiff::MatrixXvar> &mat_polycube_coordinates) const -> Var {
    autodiff::MatrixXvar mat_face_normals;
    autodiff::VectorXvar face_areas;
    ComputeFaceAreasAndNormals<autodiff::var>(mat_polycube_coordinates, mat_triangle_faces_,
                                              face_areas, mat_face_normals);

    Var E_1    = ComputeCostTermE1(mat_face_normals, face_areas);
    Var E_arap = ComputeCostTermARAP(mat_polycube_coordinates);
    Var E_complexity =
        ComputeCostTermComplexity(mat_polycube_coordinates, face_areas, mat_face_normals);
    return alpha_ * E_1 + E_arap + beta_ * E_complexity;
  }

  auto ComputeCostTermE1(autodiff::MatrixXvar &mat_face_normals,
                         autodiff::VectorXvar &face_areas) const -> Var {
    autodiff::MatrixXvar mat_l1_face_normals =
        (mat_face_normals.cwiseAbs2().array() + 0.001).cwiseSqrt();
    autodiff::var E1 = face_areas.cwiseProduct(mat_l1_face_normals.rowwise().sum()).sum();
    return E1;
  }

  auto ComputeCostTermARAP(Eigen::Map<autodiff::MatrixXvar> &mat_polycube_coordinates) const
      -> Var {
    autodiff::var E_arap = 0;
    for (int tetrahedron_idx = 0; tetrahedron_idx < num_tetrahedrons_; ++tetrahedron_idx) {
      const auto &vertices_of_tetrahedron = mat_tetrahedrons_.row(tetrahedron_idx);
      Eigen::Matrix<Var, 4, 3> mat_tetrahedron_coords;
      for (int idx = 0; idx < 4; ++idx) {
        mat_tetrahedron_coords.row(idx) =
            mat_polycube_coordinates.row(vertices_of_tetrahedron(idx));
      }
      Eigen::Matrix<Var, 3, 3> mat_gradient =
          matrices_gradient_operator_[tetrahedron_idx] * mat_tetrahedron_coords;
      Eigen::JacobiSVD<Eigen::Matrix3d> svd_gradient;
      svd_gradient.compute(mat_gradient.cast<double>(), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Var delta = 0.5 * (mat_gradient - svd_gradient.matrixU() * svd_gradient.matrixV().transpose())
                            .squaredNorm();
      E_arap += tetrahedron_volumes_(tetrahedron_idx) * delta;
    }
    return E_arap / tetrahedron_volumes_.sum();
  }

  auto ComputeCostTermComplexity(Eigen::Map<autodiff::MatrixXvar> &mat_polycube_coordinates,
                                 autodiff::VectorXvar &face_areas,
                                 autodiff::MatrixXvar &mat_face_normals) const -> Var {
    Var energy_complexity = 0;

    const size_t num_edges = mat_adjacent_face_pairs_.rows();
    const size_t num_faces = mat_triangle_faces_.rows();

    autodiff::VectorXvar face_one_ring_areas(num_faces);
    for (int face_idx = 0; face_idx < num_faces; ++face_idx) {
      Var one_ring_area = 0;
      for (int neighbor_idx : map_face_idx_to_neighbor_indices_[face_idx]) {
        one_ring_area += face_areas(neighbor_idx);
      }
      face_one_ring_areas(face_idx) = one_ring_area;
    }
    double total = 0;
    for (int edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
      int face_i          = mat_adjacent_face_pairs_(edge_idx, 0);
      int face_j          = mat_adjacent_face_pairs_(edge_idx, 1);
      Var edge_local_area = face_areas(face_j) * face_areas(face_i) / face_one_ring_areas(face_i) +
                            face_areas(face_j) * face_areas(face_i) / face_one_ring_areas(face_j);
      total += edge_local_area.expr->val;
      energy_complexity +=
          edge_local_area *
          (mat_face_normals.row(face_i) - mat_face_normals.row(face_j)).squaredNorm();
    }
    return energy_complexity / origin_total_face_area_;
  }

 private:
  const Eigen::MatrixXi &mat_triangle_faces_;
  const Eigen::MatrixXi &mat_adjacent_face_pairs_;
  const Eigen::MatrixXi &mat_tetrahedrons_;
  const std::vector<std::vector<int>> &map_face_idx_to_neighbor_indices_;
  const std::vector<Matrix34> &matrices_gradient_operator_;
  const VectorXd &tetrahedron_volumes_;
  double origin_total_face_area_;
  const size_t num_vertices_;
  const size_t num_tetrahedrons_;
  const size_t num_triangle_faces_;

 private:
  const double alpha_;
  const double beta_;
};

auto SolveL1NormBasedPolycubeProblemByIfopt(
    const Eigen::MatrixXd &mat_coordinates, const Eigen::MatrixXi &mat_tetrahedrons,
    const Eigen::MatrixXi &mat_triangle_faces, const Eigen::MatrixXi &mat_adjacent_face_pairs,
    const std::vector<std::vector<int>> &map_face_idx_to_neighbor_indices,
    const std::vector<Matrix34> &matrices_gradient_operator,
    const Eigen::VectorXd &tetrahedron_volumes, double alpha, double beta, size_t num_iterations)
    -> Eigen::MatrixXd {
  const size_t num_vertices  = mat_coordinates.rows();
  const size_t num_variables = num_vertices * 3;

  Eigen::VectorXd origin_face_areas;
  Eigen::MatrixXd mat_face_normals;
  ComputeFaceAreasAndNormals<double>(mat_coordinates, mat_triangle_faces, origin_face_areas,
                                     mat_face_normals);
  double origin_total_face_area = origin_face_areas.sum();

  MatrixX mat_polycube_coordinates = mat_coordinates;
  Eigen::Map<VectorX> variable(mat_polycube_coordinates.data(), num_variables, 1);

  ifopt::Problem problem;
  problem.AddVariableSet(std::make_shared<CoordinateVariableSet>(mat_coordinates));
  problem.AddConstraintSet(std::make_shared<ConstraintTermL1>(
      mat_tetrahedrons, mat_triangle_faces, origin_total_face_area, num_vertices));
  problem.AddCostSet(std::make_shared<CostTermL1>(
      mat_tetrahedrons, mat_triangle_faces, mat_adjacent_face_pairs,
      map_face_idx_to_neighbor_indices, matrices_gradient_operator, origin_total_face_area,
      tetrahedron_volumes, num_vertices, alpha, beta));

  ifopt::IpoptSolver ipopt_solver;
  ipopt_solver.SetOption("linear_solver", "mumps");
  ipopt_solver.SetOption("jacobian_approximation", "exact");
  ipopt_solver.SetOption("print_level", 5);
  ipopt_solver.SetOption("tol", 1e-12);
  ipopt_solver.SetOption("max_iter", static_cast<int>(num_iterations));
  ipopt_solver.Solve(problem);
  Eigen::VectorXd optimal_variables = problem.GetOptVariables()->GetValues();
  return Eigen::Map<MatrixX>(optimal_variables.data(), num_vertices, 3);
}
}  // namespace cpt
}  // namespace da
