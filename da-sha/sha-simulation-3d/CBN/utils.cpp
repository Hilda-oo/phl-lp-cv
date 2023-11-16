#include "utils.h"

namespace {
inline double cotangent(const Eigen::RowVector3d &a, const Eigen::RowVector3d &b,
                        const Eigen::RowVector3d &c) {
  Eigen::RowVector3d ba = a - b;
  Eigen::RowVector3d bc = c - b;
  return (bc.dot(ba)) / ((bc.cross(ba)).norm() + 1.0e-9);
}
}  // namespace

namespace da::sha {

void ComputeBarycentric(const Eigen::MatrixXd &polygon, const Eigen::RowVector3d &p, int rowI,
                        Eigen::MatrixXd &weight) {
  int n = polygon.rows();
  assert(weight.cols() == n);
  double w_sum = 0.0;

  for (int i = 0; i < n; ++i) {
    int prev  = (i - 1 + n) % n;
    int next  = (i + 1) % n;
    double ct = cotangent(p, polygon.row(i), polygon.row(prev)) +
                cotangent(p, polygon.row(i), polygon.row(next));
    weight(rowI, i) = ct / ((p - polygon.row(i)).squaredNorm() + 1.0e-9);
    w_sum += weight(rowI, i);
  }
  weight.row(rowI) /= w_sum;
}

}  // namespace da::sha