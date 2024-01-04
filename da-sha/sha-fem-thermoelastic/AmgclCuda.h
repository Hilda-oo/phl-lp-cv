//
// Created by cflin on 7/10/23.
//

#ifndef TOP3D_AMGCLCUDA_H
#define TOP3D_AMGCLCUDA_H

#include <iostream>
#include <amgcl/backend/cuda.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <Eigen/Eigen>
#include <amgcl/backend/cuda.hpp>
#include <amgcl/profiler.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include "LinearSolver.h"
#include "Amgcl.h"

namespace da {
    namespace sha {

            template<typename Scalar>
            class AmgclCuda : public LinearSolver<Scalar> {
                typedef amgcl::backend::cuda<Scalar> Backend;
                typedef amgcl::make_solver<
                        // Use AMG as preconditioner:
                        amgcl::amg<
                                Backend,
                                amgcl::coarsening::aggregation,
                                amgcl::relaxation::spai0
                        >,
                        // iterative solver:
                        amgcl::solver::cg<Backend>
                > Solver;

                static typename Solver::params &GetPrm() {
                    static typename Solver::params prm;
                    prm.solver.tol = 1e-10;
                    prm.solver.maxiter = 1500;
                    return prm;
                }

                static typename Backend::params &GetBprm() {
                    static typename Backend::params bprm;
                    cusparseCreate(&bprm.cusparse_handle);
                    return bprm;
                }

            public:
                AmgclCuda(const Eigen::SparseMatrix<Scalar, Eigen::ColMajor> &A,
                          typename Solver::params
                          &prm = GetPrm(),
                          typename Backend::params &bprm = GetBprm())
                        :
                        LinearSolver<Scalar>(),
                        n_(A.rows()),
                        solver_(Amgcl<Scalar>::SparseMatrix2CRS(A), prm, bprm) {
                    assert(A.rows() == A.cols());
                }

                Eigen::VectorX<Scalar> solve(const std::vector<Scalar> &rhs, bool
                verbose = true) {
                    thrust::device_vector<Scalar> d_rhs(rhs);
                    thrust::device_vector<Scalar> d_x(n_, 0.0);
                    auto [iter, error] = solver_(d_rhs, d_x);
                    if ( verbose && (error > 1e-4 || iter > 500)) {
                        std::cout << "tol: " << solver_.solver().prm.tol << '\t' << "solve_iter: "
                                  <<
                                  iter <<
                                  '\t'
                                  <<
                                  "error:"
                                  << error << std::endl;
                    }
                    std::vector<Scalar> v_x(d_x.begin(),d_x.end());
                    return Eigen::Map<Eigen::VectorX<Scalar>>(v_x.data(), v_x.size());
                }

                Eigen::VectorX<Scalar> solve(const Eigen::VectorX<Scalar> &rhs, bool
                verbose = true) {
                    std::vector<Scalar> v_rhs(rhs.data(), rhs.data() + rhs.size());
                    return solve(v_rhs, verbose);
                }


            private:
                int n_;
                Solver solver_;
            };

        } // da
    } // sha

#endif //TOP3D_AMGCLCUDA_H
