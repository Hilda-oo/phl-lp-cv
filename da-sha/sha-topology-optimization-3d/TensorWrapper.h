//
// Created by cflin on 4/20/23.
//

#ifndef TOP3D_TENSORWRAPPER_H
#define TOP3D_TENSORWRAPPER_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace da::sha {
    namespace top {
        template<typename Scalar, int NumIndices>
        class TensorWrapper : public Eigen::Tensor<Scalar, NumIndices> {
        public:
            using Base = Eigen::Tensor<Scalar, NumIndices>;
            using Eigen::Tensor<Scalar, NumIndices>::Tensor;

            template<typename... IndexTypes>
            Scalar &
            operator()(typename Base::Index firstIndex, typename Base::Index secondIndex, IndexTypes... otherIndices) {
                return Base::operator()(
                        Eigen::array<typename Base::Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
            }

            template<typename... IndexTypes>
            const Scalar &operator()(typename Base::Index firstIndex, typename Base::Index secondIndex,
                                     IndexTypes... otherIndices) const {
                return Base::operator()(
                        Eigen::array<typename Base::Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
            }

            const Eigen::Matrix<Scalar, -1, 1>
            operator()(const Eigen::MatrixXi &indices) const {
                Eigen::Matrix<Scalar, -1, 1> result(indices.rows());
                for (int i = 0; i < indices.rows(); ++i) {
                    result(i) = Base::operator()(indices(i, 0), indices(i, 1), indices(i, 2));
                }
                return result;
            }
        };


    } // top
} // da::sha
#endif //TOP3D_TENSORWRAPPER_H
