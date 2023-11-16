//
// Created by Wei Chen on 3/2/22
//

#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include <Eigen/Eigen>
namespace da::sha {
    namespace top {

        class Material {
        public:
            double YM=1.0, PR=0.3; // Young's modulus, Poisson's ratio
            double density=1000; // density
            double alpha=1; // scaling factor
            double k0=1; // thermal conductivity
            Eigen::Matrix<double, 6, 6> D; // constitutive matrix
            

        public:
            Material(double thermal_conductivity):Material(1.0,0.3){k0=thermal_conductivity;}
            Material(double p_YM, double p_PR) : Material(p_YM, p_PR, 1000, 1) {}

            Material(double p_YM, double p_PR, double p_density, double p_alpha);

            // compute element stiffness matrix of 3D linear elastic material
            // using 2-nd Gauss integration
            // @input
            //  a, b, c: half size of element
            //  D      : constitutive matrix, (6, 6)
            // @output
            //  Ke     : element matrix, (24, 24)
            static void computeKe(double a, double b, double c, const Eigen::Matrix<double, 6, 6> &D,
                                  Eigen::Matrix<double, 24, 24> &Ke);

            // compute element stiffness matrix of 3D linear elastic material in heat condition
            // using 2-nd Gauss integration
            // @input
            //  a, b, c: half size of element
            //  k0      : thermal_conductivity
            // @output
            //  heatKe     : element matrix, (8, 8)
            static Eigen::Matrix<double, 8, 8>  computeHeatKe(double a,double b,double c,double k0);

            // compute element strain-displacement matrix B in 8 Gauss Points
            // @input
            //  a, b, c: half size of element
            // @output
            //  Be     : element strain-displacement matrix, (6, 24)
            static void computeBe(double a, double b, double c, std::vector<Eigen::Matrix<double, 6, 24>> &Be);

            // compute element shape function matrix in a given local point P
            // @input
            //  P: given local point, in [-1, 1] * [-1, 1] * [-1, 1]
            // @output
            //  N: shape function matrix N, (3, 24)
            static void computeN(const Eigen::RowVector3d &P, Eigen::Matrix<double, 3, 24> &N);

            // compute element strain-displacement matrix B in a given local point P
            // @input
            //  a, b, c: half size of element
            //  P: given local point, in [-1, 1] * [-1, 1] * [-1, 1]
            // @output
            //  B: element strain-displacement matrix, (6, 24)
            static void
            computeB(double a, double b, double c, const Eigen::RowVector3d &P, Eigen::Matrix<double, 6, 24> &B);

    private:
            static Eigen::RowVector<double,8> computeNi(const Eigen::RowVector3d &P){
                Eigen::RowVector<double,8>Ni;
                Ni.setZero();
                double x = P(0), y = P(1), z = P(2);
                // TODO: eps??
                assert(x >= -1.0 && x <= 1.0 && y >= -1.0 && y <= 1.0 && z >= -1.0 && z <= 1.0);
                Eigen::RowVector<double, 8> tmp;
                Ni(0) = 0.125 * (1.0 - x) * (1.0 - y) * (1.0 - z);
                Ni(1) = 0.125 * (1.0 + x) * (1.0 - y) * (1.0 - z);
                Ni(2) = 0.125 * (1.0 + x) * (1.0 + y) * (1.0 - z);
                Ni(3) = 0.125 * (1.0 - x) * (1.0 + y) * (1.0 - z);
                Ni(4) = 0.125 * (1.0 - x) * (1.0 - y) * (1.0 + z);
                Ni(5) = 0.125 * (1.0 + x) * (1.0 - y) * (1.0 + z);
                Ni(6) = 0.125 * (1.0 + x) * (1.0 + y) * (1.0 + z);
                Ni(7) = 0.125 * (1.0 - x) * (1.0 + y) * (1.0 + z);
                return Ni;
            }
            static Eigen::Matrix<double,3,8> computedNi(const Eigen::RowVector3d &P){
                double x = P(0), y = P(1), z = P(2);
                // TODO: eps??
                assert(x >= -1.0 && x <= 1.0 && y >= -1.0 && y <= 1.0 && z >= -1.0 && z <= 1.0);
                Eigen::Matrix<double, 3, 8> dNi;
                dNi(0,0) = 0.125 *  - 1 * (1.0 - y) * (1.0 - z);
                dNi(0,1) = 0.125 *    1 * (1.0 - y) * (1.0 - z);
                dNi(0,2) = 0.125 *    1 * (1.0 - y) * (1.0 - z);
                dNi(0,3) = 0.125 *  - 1 * (1.0 + y) * (1.0 - z);
                dNi(0,4) = 0.125 *  - 1 * (1.0 + y) * (1.0 - z);
                dNi(0,5) = 0.125 *    1 * (1.0 - y) * (1.0 + z);
                dNi(0,6) = 0.125 *    1 * (1.0 - y) * (1.0 + z);
                dNi(0,7) = 0.125 *  - 1 * (1.0 + y) * (1.0 + z);
                dNi(1,0) = 0.125 * (1.0 - x) *  - 1 * (1.0 - z);
                dNi(1,1) = 0.125 * (1.0 + x) *  - 1 * (1.0 - z);
                dNi(1,2) = 0.125 * (1.0 + x) *  - 1 * (1.0 - z);
                dNi(1,3) = 0.125 * (1.0 - x) *  + 1 * (1.0 - z);
                dNi(1,4) = 0.125 * (1.0 - x) *  + 1 * (1.0 - z);
                dNi(1,5) = 0.125 * (1.0 + x) *  - 1 * (1.0 + z);
                dNi(1,6) = 0.125 * (1.0 + x) *  - 1 * (1.0 + z);
                dNi(1,7) = 0.125 * (1.0 - x) *  + 1 * (1.0 + z);
                dNi(2,0) = 0.125 * (1.0 - x) * (1.0 - y) *  - 1;
                dNi(2,1) = 0.125 * (1.0 + x) * (1.0 - y) *  - 1;
                dNi(2,2) = 0.125 * (1.0 + x) * (1.0 + y) *  - 1;
                dNi(2,3) = 0.125 * (1.0 - x) * (1.0 + y) *  - 1;
                dNi(2,4) = 0.125 * (1.0 - x) * (1.0 - y) *  + 1;
                dNi(2,5) = 0.125 * (1.0 + x) * (1.0 - y) *  + 1;
                dNi(2,6) = 0.125 * (1.0 + x) * (1.0 + y) *  + 1;
                dNi(2,7) = 0.125 * (1.0 - x) * (1.0 + y) *  + 1;
                return dNi;
            }
        };

    } // top
} // da::sha

#endif // MATERIAL_HPP