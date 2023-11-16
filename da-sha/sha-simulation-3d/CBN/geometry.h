#pragma once

#include <Eigen/Eigen>

namespace da::sha {
class Triangle {
 public:
  // use standard rectangle to fit triangle
  Eigen::Matrix<double, 4, 3> Coords;

 public:
  explicit Triangle(const Eigen::Matrix3d &p_Coords);

 public:
  // update Coords
  void Update(const Eigen::Matrix3d &p_Coords);

  // get two base vector
  Eigen::RowVector3d GetBase1(const Eigen::RowVector3d &local) const;
  Eigen::RowVector3d GetBase2(const Eigen::RowVector3d &local) const;
  // compute determinant of Jacobian matrix
  double GetDetJac(const Eigen::RowVector3d &local) const;
  // get normal vector
  Eigen::RowVector3d GetNormal(const Eigen::RowVector3d &local) const;
  // map local coordinates (standard rectangle) to global coordinates
  void MapLocalToGlobal(const Eigen::RowVector3d &local, Eigen::RowVector3d &global) const;
};

inline Triangle::Triangle(const Eigen::Matrix3d &p_Coords) {
  Coords({0, 1, 2}, Eigen::all) = p_Coords;
  Coords.row(3)                 = Coords.row(2);
}

inline void Triangle::Update(const Eigen::Matrix3d &p_Coords) {
  Coords({0, 1, 2}, Eigen::all) = p_Coords;
  Coords.row(3)                 = Coords.row(2);
}

inline Eigen::RowVector3d Triangle::GetBase1(const Eigen::RowVector3d &local) const {
  return 0.25 *
         ((-1.0) * (1.0 - local(1)) * Coords.row(0) + (1.0) * (1.0 - local(1)) * Coords.row(1) +
          (1.0) * (1.0 + local(1)) * Coords.row(2) + (-1.0) * (1.0 + local(1)) * Coords.row(3));
}

inline Eigen::RowVector3d Triangle::GetBase2(const Eigen::RowVector3d &local) const {
  return 0.25 *
         ((1.0 - local(0)) * (-1.0) * Coords.row(0) + (1.0 + local(0)) * (-1.0) * Coords.row(1) +
          (1.0 + local(0)) * (1.0) * Coords.row(2) + (1.0 - local(0)) * (1.0) * Coords.row(3));
}

inline double Triangle::GetDetJac(const Eigen::RowVector3d &local) const {
  // calculate base vector
  Eigen::RowVector3d base1 = GetBase1(local);
  Eigen::RowVector3d base2 = GetBase2(local);
  assert(base1.norm() > 1.0e-6 && base2.norm() > 1.0e-6);
  // return determinant of Jacobian matrix
  return base1.cross(base2).norm();
}

inline Eigen::RowVector3d Triangle::GetNormal(const Eigen::RowVector3d &local) const {
  // calculate base vector
  Eigen::RowVector3d base1 = GetBase1(local);
  Eigen::RowVector3d base2 = GetBase2(local);
  assert(base1.norm() > 1.0e-6 && base2.norm() > 1.0e-6);
  // return normal vector
  return (base1.cross(base2)).normalized();
}

inline void Triangle::MapLocalToGlobal(const Eigen::RowVector3d &local,
                                       Eigen::RowVector3d &global) const {
  global = 0.25 * ((1.0 - local(0)) * (1.0 - local(1)) * Coords.row(0) +
                   (1.0 + local(0)) * (1.0 - local(1)) * Coords.row(1) +
                   (1.0 + local(0)) * (1.0 + local(1)) * Coords.row(2) +
                   (1.0 - local(0)) * (1.0 + local(1)) * Coords.row(3));
}

}  // namespace da