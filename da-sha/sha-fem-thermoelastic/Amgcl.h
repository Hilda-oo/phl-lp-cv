//
// Created by cflin on 7/7/23.
//

#ifndef TOP3D_AMGCL_H
#define TOP3D_AMGCL_H

#include <iostream>
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <Eigen/Eigen>
#include <amgcl/solver/lgmres.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include "LinearSolver.h"

namespace da::sha {

    template<typename Scalar>
    class Amgcl : public LinearSolver<Scalar> {
        typedef amgcl::backend::builtin<Scalar> Backend;
        typedef amgcl::make_solver<
                // Use AMG as preconditioner:
                amgcl::amg<
                        Backend,
                        amgcl::coarsening::smoothed_aggregation,
                        amgcl::relaxation::gauss_seidel
                >,
                // iterative solver:
                amgcl::solver::lgmres<Backend>
        > Solver;
    public:
        Amgcl(const Eigen::SparseMatrix<Scalar, Eigen::ColMajor> &A, typename Solver::params
        &prm = GetPrm())
                :
                n_(A.rows()),
                solver_(SparseMatrix2CRS(A), prm) {
            assert(A.rows() == A.cols());
        }


        Eigen::VectorX<Scalar> solve(const std::vector<Scalar> &rhs, bool
        verbose = true) {
            std::vector<Scalar> x(n_, 0);
            auto [iter, error] = solver_(rhs, x);
            if (verbose && (error > 1e-4 || iter > 500)) {
                std::cout << "tol: " << solver_.solver().prm.tol << '\t' << "solve_iter: " <<
                          iter <<
                          '\t'
                          <<
                          "error:"
                          << error << std::endl;
            }
            return Eigen::Map<Eigen::VectorX<Scalar>>(x.data(), x.size());
        }

        Eigen::VectorX<Scalar> solve(const Eigen::VectorX<Scalar> &rhs, bool
        verbose = true) {
            std::vector<Scalar> v_rhs(rhs.data(), rhs.data() + rhs.size());
            return solve(v_rhs, verbose);
        }

    public:
        static typename Solver::params &GetPrm() {
            static typename Solver::params prm;
            prm.solver.tol = 1e-10;
            prm.solver.maxiter = 1500;
//            prm.precond.coarsening.aggr.eps_strong=0.0;
//            prm.solver.M=100;
            return prm;
        }

        static std::tuple<ptrdiff_t, std::vector<Eigen::Index>, std::vector<Eigen::Index>,
                std::vector<Scalar>>
        SparseMatrix2CRS(Eigen::SparseMatrix<Scalar, Eigen::RowMajor> spM) {
            spM.makeCompressed();
            std::vector<Scalar> val(spM.valuePtr(), spM.valuePtr() + spM.nonZeros());
            std::vector<Eigen::Index> col(spM.innerIndexPtr(),
                                          spM.innerIndexPtr() + spM.nonZeros());
            std::vector<Eigen::Index> ptr(spM.outerIndexPtr(),
                                          spM.outerIndexPtr() + spM.rows() + 1);
            ptrdiff_t rows = spM.rows();
            return {rows, ptr, col, val};
        }

    private:
        int n_;
        Solver solver_;
    };

} // top

#endif //TOP3D_AMGCL_H
