#include "material_utils.h"

#include "sha-base-framework/frame.h"

namespace da::sha {

void ComputeElasticMatrix(double YM, double PR, Eigen::Matrix<double, 6, 6> &D) {
  D << 1.0 - PR, PR, PR, 0.0, 0.0, 0.0, PR, 1.0 - PR, PR, 0.0, 0.0, 0.0, PR, PR, 1.0 - PR, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * PR) / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      (1.0 - 2.0 * PR) / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * PR) / 2.0;
  D *= YM / (1.0 + PR) / (1.0 - 2.0 * PR);
}

double ComputeVonStress(const Eigen::Vector<double, 6> &stress) {
  return sqrt(0.5 * (pow(stress(0) - stress(1), 2.0) + pow(stress(1) - stress(2), 2.0) +
                     pow(stress(2) - stress(0), 2.0)) +
              3.0 * stress({3, 4, 5}).squaredNorm());
}

}  // namespace da::sha
