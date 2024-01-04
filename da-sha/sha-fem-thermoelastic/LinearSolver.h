//
// Created by cflin on 7/10/23.
//

#ifndef TOP3D_LINEARSOLVER_H
#define TOP3D_LINEARSOLVER_H
#include <Eigen/Eigen>

namespace da::sha{
    template<typename Scalar>
    class LinearSolver{
    public:
        LinearSolver()=default;
        virtual Eigen::VectorX<Scalar> solve(const Eigen::VectorX<Scalar>&rhs,bool verbose=true)=0;
    };
}
#endif //TOP3D_LINEARSOLVER_H
