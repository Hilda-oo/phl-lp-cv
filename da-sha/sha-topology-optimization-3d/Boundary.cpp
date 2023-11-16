//
// Created by cflin on 4/21/23.
//

#include "Boundary.h"

namespace da::sha {
    namespace top {
        Eigen::MatrixXi Boundary::GetChosenCoords(const Eigen::VectorXi &x_sequence, const Eigen::VectorXi &y_sequence,
                                                  const Eigen::VectorXi &z_sequence) {
            Eigen::MatrixXi chosen_coords(x_sequence.size() * y_sequence.size() * z_sequence.size(), 3);
            chosen_coords.col(0) = x_sequence.replicate(y_sequence.size() * z_sequence.size(), 1);
            chosen_coords.col(1) = y_sequence.transpose().replicate(x_sequence.size(), 1).reshaped().replicate(
                    z_sequence.size(), 1);
            chosen_coords.col(2) = z_sequence.transpose().replicate(x_sequence.size() * y_sequence.size(),
                                                                    1).reshaped();
            return ReduceInvalidCoords(chosen_coords);
        }
    } // top
} // da::sha