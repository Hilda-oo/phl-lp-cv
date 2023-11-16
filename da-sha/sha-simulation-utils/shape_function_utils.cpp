#include "shape_function_utils.h"

#include "sha-base-framework/frame.h"

namespace da::sha {

void ComputeKe(double a, double b, double c, const Eigen::Matrix<double, 6, 6> &D,
               Eigen::Matrix<double, 24, 24> &Ke) {
  Eigen::Vector2d GP(-1.0 / sqrt(3.0), 1.0 / sqrt(3.0));
  Eigen::Vector2d GW(1.0, 1.0);
  Ke.setZero();
  Eigen::Matrix<double, 6, 9> L = Eigen::Matrix<double, 6, 9>::Zero();
  L(0, 0)                       = 1.0;
  L(1, 4)                       = 1.0;
  L(2, 8)                       = 1.0;
  L(3, 1)                       = 1.0;
  L(3, 3)                       = 1.0;
  L(4, 5)                       = 1.0;
  L(4, 7)                       = 1.0;
  L(5, 2)                       = 1.0;
  L(5, 6)                       = 1.0;

  Eigen::Matrix<double, 8, 3> tmp{{-a, -b, -c}, {a, -b, -c}, {a, b, -c}, {-a, b, -c},
                                  {-a, -b, c},  {a, -b, c},  {a, b, c},  {-a, b, c}};

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        double x = GP(i), y = GP(j), z = GP(k);
        Eigen::RowVector<double, 8> dNx{-(1.0 - y) * (1.0 - z) / 8.0, (1.0 - y) * (1.0 - z) / 8.0,
                                        (1.0 + y) * (1.0 - z) / 8.0,  -(1.0 + y) * (1.0 - z) / 8.0,
                                        -(1.0 - y) * (1.0 + z) / 8.0, (1.0 - y) * (1.0 + z) / 8.0,
                                        (1.0 + y) * (1.0 + z) / 8.0,  -(1.0 + y) * (1.0 + z) / 8.0};
        Eigen::RowVector<double, 8> dNy{-(1.0 - x) * (1.0 - z) / 8.0, -(1.0 + x) * (1.0 - z) / 8.0,
                                        (1.0 + x) * (1.0 - z) / 8.0,  (1.0 - x) * (1.0 - z) / 8.0,
                                        -(1.0 - x) * (1.0 + z) / 8.0, -(1.0 + x) * (1.0 + z) / 8.0,
                                        (1.0 + x) * (1.0 + z) / 8.0,  (1.0 - x) * (1.0 + z) / 8.0};
        Eigen::RowVector<double, 8> dNz{-(1.0 - x) * (1.0 - y) / 8.0, -(1.0 + x) * (1.0 - y) / 8.0,
                                        -(1.0 + x) * (1.0 + y) / 8.0, -(1.0 - x) * (1.0 + y) / 8.0,
                                        (1.0 - x) * (1.0 - y) / 8.0,  (1.0 + x) * (1.0 - y) / 8.0,
                                        (1.0 + x) * (1.0 + y) / 8.0,  (1.0 - x) * (1.0 + y) / 8.0};

        Eigen::Matrix<double, 3, 8> tmp1;
        tmp1(0, Eigen::all)  = dNx;
        tmp1(1, Eigen::all)  = dNy;
        tmp1(2, Eigen::all)  = dNz;
        Eigen::Matrix3d J    = tmp1 * tmp;
        Eigen::Matrix3d JInv = J.inverse();
        double JDet          = J.determinant();

        Eigen::Matrix<double, 9, 9> G         = Eigen::Matrix<double, 9, 9>::Zero();
        G(Eigen::seq(0, 2), Eigen::seq(0, 2)) = JInv;
        G(Eigen::seq(3, 5), Eigen::seq(3, 5)) = JInv;
        G(Eigen::seq(6, 8), Eigen::seq(6, 8)) = JInv;

        Eigen::Matrix<double, 9, 24> dN = Eigen::Matrix<double, 9, 24>::Zero();
        dN(0, Eigen::seq(0, 23, 3))     = dNx;
        dN(1, Eigen::seq(0, 23, 3))     = dNy;
        dN(2, Eigen::seq(0, 23, 3))     = dNz;
        dN(3, Eigen::seq(1, 23, 3))     = dNx;
        dN(4, Eigen::seq(1, 23, 3))     = dNy;
        dN(5, Eigen::seq(1, 23, 3))     = dNz;
        dN(6, Eigen::seq(2, 23, 3))     = dNx;
        dN(7, Eigen::seq(2, 23, 3))     = dNy;
        dN(8, Eigen::seq(2, 23, 3))     = dNz;

        Eigen::Matrix<double, 6, 24> Be = L * G * dN;
        Ke = Ke + GW(i) * GW(j) * GW(k) * JDet * (Be.transpose() * D * Be);
      }
    }
  }
}

void ComputeBe(double a, double b, double c, std::vector<Eigen::Matrix<double, 6, 24>> &Be) {
  assert(Be.size() == 8);  // 8 Gauss Points

  int gcnt = 0;  // count for Gauss Points
  Eigen::Vector2d GP(-1.0 / sqrt(3.0), 1.0 / sqrt(3.0));
  Eigen::Matrix<double, 6, 9> L = Eigen::Matrix<double, 6, 9>::Zero();
  L(0, 0)                       = 1.0;
  L(1, 4)                       = 1.0;
  L(2, 8)                       = 1.0;
  L(3, 1)                       = 1.0;
  L(3, 3)                       = 1.0;
  L(4, 5)                       = 1.0;
  L(4, 7)                       = 1.0;
  L(5, 2)                       = 1.0;
  L(5, 6)                       = 1.0;

  Eigen::Matrix<double, 8, 3> tmp{{-a, -b, -c}, {a, -b, -c}, {a, b, -c}, {-a, b, -c},
                                  {-a, -b, c},  {a, -b, c},  {a, b, c},  {-a, b, c}};

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        double x = GP(i), y = GP(j), z = GP(k);
        Eigen::RowVector<double, 8> dNx{-(1.0 - y) * (1.0 - z) / 8.0, (1.0 - y) * (1.0 - z) / 8.0,
                                        (1.0 + y) * (1.0 - z) / 8.0,  -(1.0 + y) * (1.0 - z) / 8.0,
                                        -(1.0 - y) * (1.0 + z) / 8.0, (1.0 - y) * (1.0 + z) / 8.0,
                                        (1.0 + y) * (1.0 + z) / 8.0,  -(1.0 + y) * (1.0 + z) / 8.0};
        Eigen::RowVector<double, 8> dNy{-(1.0 - x) * (1.0 - z) / 8.0, -(1.0 + x) * (1.0 - z) / 8.0,
                                        (1.0 + x) * (1.0 - z) / 8.0,  (1.0 - x) * (1.0 - z) / 8.0,
                                        -(1.0 - x) * (1.0 + z) / 8.0, -(1.0 + x) * (1.0 + z) / 8.0,
                                        (1.0 + x) * (1.0 + z) / 8.0,  (1.0 - x) * (1.0 + z) / 8.0};
        Eigen::RowVector<double, 8> dNz{-(1.0 - x) * (1.0 - y) / 8.0, -(1.0 + x) * (1.0 - y) / 8.0,
                                        -(1.0 + x) * (1.0 + y) / 8.0, -(1.0 - x) * (1.0 + y) / 8.0,
                                        (1.0 - x) * (1.0 - y) / 8.0,  (1.0 + x) * (1.0 - y) / 8.0,
                                        (1.0 + x) * (1.0 + y) / 8.0,  (1.0 - x) * (1.0 + y) / 8.0};

        Eigen::Matrix<double, 3, 8> tmp1;
        tmp1(0, Eigen::all)  = dNx;
        tmp1(1, Eigen::all)  = dNy;
        tmp1(2, Eigen::all)  = dNz;
        Eigen::Matrix3d J    = tmp1 * tmp;
        Eigen::Matrix3d JInv = J.inverse();

        Eigen::Matrix<double, 9, 9> G         = Eigen::Matrix<double, 9, 9>::Zero();
        G(Eigen::seq(0, 2), Eigen::seq(0, 2)) = JInv;
        G(Eigen::seq(3, 5), Eigen::seq(3, 5)) = JInv;
        G(Eigen::seq(6, 8), Eigen::seq(6, 8)) = JInv;

        Eigen::Matrix<double, 9, 24> dN = Eigen::Matrix<double, 9, 24>::Zero();
        dN(0, Eigen::seq(0, 23, 3))     = dNx;
        dN(1, Eigen::seq(0, 23, 3))     = dNy;
        dN(2, Eigen::seq(0, 23, 3))     = dNz;
        dN(3, Eigen::seq(1, 23, 3))     = dNx;
        dN(4, Eigen::seq(1, 23, 3))     = dNy;
        dN(5, Eigen::seq(1, 23, 3))     = dNz;
        dN(6, Eigen::seq(2, 23, 3))     = dNx;
        dN(7, Eigen::seq(2, 23, 3))     = dNy;
        dN(8, Eigen::seq(2, 23, 3))     = dNz;

        Be[gcnt].setZero(6, 24);
        Be[gcnt] = L * G * dN;
        ++gcnt;
      }
    }
  }
}

void ComputeN(const Eigen::RowVector3d &P, Eigen::Matrix<double, 3, 24> &N) {
  N.setZero();

  double x = P(0), y = P(1), z = P(2);
  // TODO: eps??
  assert(x >= -1.0 && x <= 1.0 && y >= -1.0 && y <= 1.0 && z >= -1.0 && z <= 1.0);
  Eigen::RowVector<double, 8> tmp;
  tmp(0)                     = 0.125 * (1.0 - x) * (1.0 - y) * (1.0 - z);
  tmp(1)                     = 0.125 * (1.0 + x) * (1.0 - y) * (1.0 - z);
  tmp(2)                     = 0.125 * (1.0 + x) * (1.0 + y) * (1.0 - z);
  tmp(3)                     = 0.125 * (1.0 - x) * (1.0 + y) * (1.0 - z);
  tmp(4)                     = 0.125 * (1.0 - x) * (1.0 - y) * (1.0 + z);
  tmp(5)                     = 0.125 * (1.0 + x) * (1.0 - y) * (1.0 + z);
  tmp(6)                     = 0.125 * (1.0 + x) * (1.0 + y) * (1.0 + z);
  tmp(7)                     = 0.125 * (1.0 - x) * (1.0 + y) * (1.0 + z);
  N(0, Eigen::seq(0, 23, 3)) = tmp;
  N(1, Eigen::seq(1, 23, 3)) = tmp;
  N(2, Eigen::seq(2, 23, 3)) = tmp;
}

void ComputeB(double a, double b, double c, const Eigen::RowVector3d &P,
              Eigen::Matrix<double, 6, 24> &B) {
  B.setZero();

  double x = P(0), y = P(1), z = P(2);
  // TODO: eps??
  assert(x >= -1.0 && x <= 1.0 && y >= -1.0 && y <= 1.0 && z >= -1.0 && z <= 1.0);
  Eigen::Matrix<double, 3, 8> tmp;
  tmp(0, 0) = -0.125 * (1.0 - y) * (1.0 - z);
  tmp(0, 1) = 0.125 * (1.0 - y) * (1.0 - z);
  tmp(0, 2) = 0.125 * (1.0 + y) * (1.0 - z);
  tmp(0, 3) = -0.125 * (1.0 + y) * (1.0 - z);
  tmp(0, 4) = -0.125 * (1.0 - y) * (1.0 + z);
  tmp(0, 5) = 0.125 * (1.0 - y) * (1.0 + z);
  tmp(0, 6) = 0.125 * (1.0 + y) * (1.0 + z);
  tmp(0, 7) = -0.125 * (1.0 + y) * (1.0 + z);

  tmp(1, 0) = -0.125 * (1.0 - x) * (1.0 - z);
  tmp(1, 1) = -0.125 * (1.0 + x) * (1.0 - z);
  tmp(1, 2) = 0.125 * (1.0 + x) * (1.0 - z);
  tmp(1, 3) = 0.125 * (1.0 - x) * (1.0 - z);
  tmp(1, 4) = -0.125 * (1.0 - x) * (1.0 + z);
  tmp(1, 5) = -0.125 * (1.0 + x) * (1.0 + z);
  tmp(1, 6) = 0.125 * (1.0 + x) * (1.0 + z);
  tmp(1, 7) = 0.125 * (1.0 - x) * (1.0 + z);

  tmp(2, 0) = -0.125 * (1.0 - x) * (1.0 - y);
  tmp(2, 1) = -0.125 * (1.0 + x) * (1.0 - y);
  tmp(2, 2) = -0.125 * (1.0 + x) * (1.0 + y);
  tmp(2, 3) = -0.125 * (1.0 - x) * (1.0 + y);
  tmp(2, 4) = 0.125 * (1.0 - x) * (1.0 - y);
  tmp(2, 5) = 0.125 * (1.0 + x) * (1.0 - y);
  tmp(2, 6) = 0.125 * (1.0 + x) * (1.0 + y);
  tmp(2, 7) = 0.125 * (1.0 - x) * (1.0 + y);

  tmp.row(0) /= a;
  tmp.row(1) /= b;
  tmp.row(2) /= c;

  B(0, Eigen::seq(0, 23, 3)) = tmp.row(0);
  B(1, Eigen::seq(1, 23, 3)) = tmp.row(1);
  B(2, Eigen::seq(2, 23, 3)) = tmp.row(2);

  B(3, Eigen::seq(0, 23, 3)) = tmp.row(1);
  B(3, Eigen::seq(1, 23, 3)) = tmp.row(0);

  B(4, Eigen::seq(1, 23, 3)) = tmp.row(2);
  B(4, Eigen::seq(2, 23, 3)) = tmp.row(1);

  B(5, Eigen::seq(0, 23, 3)) = tmp.row(2);
  B(5, Eigen::seq(2, 23, 3)) = tmp.row(0);
}

void ComputeNForTet(const Eigen::RowVector3d &P, const Eigen::Matrix<double, 4, 3> &X,
                    Eigen::Matrix<double, 3, 12> &N) {
  Eigen::Matrix<double, 4, 4> H;
  H.col(0).setOnes();
  H(Eigen::all, Eigen::seq(1, Eigen::last)) = X;
  double V6                                 = H.determinant();

  Eigen::Matrix<int, 4, 3> index;
  index << 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2;

  Eigen::RowVector4d a;
  Eigen::RowVector4d b;
  Eigen::RowVector4d c;
  Eigen::RowVector4d d;

  a(0) = H(index.row(0), index.row(0)).determinant();
  a(1) = -H(index.row(1), index.row(0)).determinant();
  a(2) = H(index.row(2), index.row(0)).determinant();
  a(3) = -H(index.row(3), index.row(0)).determinant();

  b(0) = -H(index.row(0), index.row(1)).determinant();
  b(1) = H(index.row(1), index.row(1)).determinant();
  b(2) = -H(index.row(2), index.row(1)).determinant();
  b(3) = H(index.row(3), index.row(1)).determinant();

  c(0) = H(index.row(0), index.row(2)).determinant();
  c(1) = -H(index.row(1), index.row(2)).determinant();
  c(2) = H(index.row(2), index.row(2)).determinant();
  c(3) = -H(index.row(3), index.row(2)).determinant();

  d(0) = -H(index.row(0), index.row(3)).determinant();
  d(1) = H(index.row(1), index.row(3)).determinant();
  d(2) = -H(index.row(2), index.row(3)).determinant();
  d(3) = H(index.row(3), index.row(3)).determinant();

  Eigen::RowVector4d NN;
  NN(0) = (a(0) + b(0) * P(0) + c(0) * P(1) + d(0) * P(2)) / V6;
  NN(1) = (a(1) + b(1) * P(0) + c(1) * P(1) + d(1) * P(2)) / V6;
  NN(2) = (a(2) + b(2) * P(0) + c(2) * P(1) + d(2) * P(2)) / V6;
  NN(3) = (a(3) + b(3) * P(0) + c(3) * P(1) + d(3) * P(2)) / V6;

  N.setZero();
  N(0, Eigen::seq(0, Eigen::last, 3)) = NN;
  N(1, Eigen::seq(1, Eigen::last, 3)) = NN;
  N(2, Eigen::seq(2, Eigen::last, 3)) = NN;
}

void ComputeBForTet(const Eigen::Matrix<double, 4, 3> &X, Eigen::Matrix<double, 6, 12> &B) {
  Eigen::Matrix<double, 4, 4> H;
  H.col(0).setOnes();
  H(Eigen::all, Eigen::seq(1, Eigen::last)) = X;
  double V6                                 = H.determinant();

  Eigen::Matrix<int, 4, 3> index;
  index << 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2;

  Eigen::RowVector4d b;
  Eigen::RowVector4d c;
  Eigen::RowVector4d d;

  b(0) = -H(index.row(0), index.row(1)).determinant();
  b(1) = H(index.row(1), index.row(1)).determinant();
  b(2) = -H(index.row(2), index.row(1)).determinant();
  b(3) = H(index.row(3), index.row(1)).determinant();

  c(0) = H(index.row(0), index.row(2)).determinant();
  c(1) = -H(index.row(1), index.row(2)).determinant();
  c(2) = H(index.row(2), index.row(2)).determinant();
  c(3) = -H(index.row(3), index.row(2)).determinant();

  d(0) = -H(index.row(0), index.row(3)).determinant();
  d(1) = H(index.row(1), index.row(3)).determinant();
  d(2) = -H(index.row(2), index.row(3)).determinant();
  d(3) = H(index.row(3), index.row(3)).determinant();

  B.setZero();
  B(0, Eigen::seq(0, Eigen::last, 3)) = b;
  B(1, Eigen::seq(1, Eigen::last, 3)) = c;
  B(2, Eigen::seq(2, Eigen::last, 3)) = d;
  B(3, Eigen::seq(0, Eigen::last, 3)) = c;
  B(3, Eigen::seq(1, Eigen::last, 3)) = b;
  B(4, Eigen::seq(1, Eigen::last, 3)) = d;
  B(4, Eigen::seq(2, Eigen::last, 3)) = c;
  B(5, Eigen::seq(0, Eigen::last, 3)) = d;
  B(5, Eigen::seq(2, Eigen::last, 3)) = b;
  B /= V6;
}

void ComputeKeForTet(const Eigen::Matrix<double, 4, 3> &X, const Eigen::Matrix<double, 6, 6> &D,
                     Eigen::Matrix<double, 12, 12> &Ke, double &Vol) {
  Eigen::Matrix<double, 4, 4> H;
  H.col(0).setOnes();
  H(Eigen::all, Eigen::seq(1, Eigen::last)) = X;
  double V6                                 = H.determinant();
  Vol                                       = V6 / 6.0;

  Eigen::Matrix<int, 4, 3> index;
  index << 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2;

  Eigen::RowVector4d b;
  Eigen::RowVector4d c;
  Eigen::RowVector4d d;

  b(0) = -H(index.row(0), index.row(1)).determinant();
  b(1) = H(index.row(1), index.row(1)).determinant();
  b(2) = -H(index.row(2), index.row(1)).determinant();
  b(3) = H(index.row(3), index.row(1)).determinant();

  c(0) = H(index.row(0), index.row(2)).determinant();
  c(1) = -H(index.row(1), index.row(2)).determinant();
  c(2) = H(index.row(2), index.row(2)).determinant();
  c(3) = -H(index.row(3), index.row(2)).determinant();

  d(0) = -H(index.row(0), index.row(3)).determinant();
  d(1) = H(index.row(1), index.row(3)).determinant();
  d(2) = -H(index.row(2), index.row(3)).determinant();
  d(3) = H(index.row(3), index.row(3)).determinant();

  Eigen::Matrix<double, 6, 12> B;
  B.setZero();
  B(0, Eigen::seq(0, Eigen::last, 3)) = b;
  B(1, Eigen::seq(1, Eigen::last, 3)) = c;
  B(2, Eigen::seq(2, Eigen::last, 3)) = d;
  B(3, Eigen::seq(0, Eigen::last, 3)) = c;
  B(3, Eigen::seq(1, Eigen::last, 3)) = b;
  B(4, Eigen::seq(1, Eigen::last, 3)) = d;
  B(4, Eigen::seq(2, Eigen::last, 3)) = c;
  B(5, Eigen::seq(0, Eigen::last, 3)) = d;
  B(5, Eigen::seq(2, Eigen::last, 3)) = b;
  B /= V6;

  Ke = Vol * (B.transpose() * D * B);
}
}  // namespace da::sha