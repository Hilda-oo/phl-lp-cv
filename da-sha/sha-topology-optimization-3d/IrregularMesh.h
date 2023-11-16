//
// Created by cflin on 4/24/23.
//

#ifndef TOP3D_IRREGULARMESH_H
#define TOP3D_IRREGULARMESH_H

#include "Mesh.h"

namespace da::sha {
    namespace top {
    class Boundary;
        struct ModelMesh {
            Eigen::MatrixXd points;
            Eigen::MatrixXi surfaces;
        };

        class IrregularMesh : public Mesh {
            friend class Boundary;
        public:
            IrregularMesh(const fs_path &arbitrary_stl_path,const fs_path &chosen_stl_path,double relative_length_of_voxel,int dofs_each_node=3);

            int GetNumDofs() const override {
                return num_node_pixel_ * DOFS_EACH_NODE;
            }

            int GetNumEles() const override {
                return num_pixel_;
            }

            int GetNumNodes() const override {
                return num_node_pixel_;
            }



        private:
        double EvaluateArbitrarySDF(const Eigen::Vector3d &point);
        double EvaluateChosenSDF(const Eigen::Vector3d &point);

         Eigen::VectorXd EvaluateArbitrarySDF (const Eigen::MatrixXd & points);

         Eigen::VectorXd  EvaluateChosenSDF(const Eigen::MatrixXd & points);
            double num_pixel_;
            double num_node_pixel_;
            Eigen::Vector3d min_point_box_;
            Eigen::Vector3d len_box_;
            ModelMesh arbitrary_mesh_;
            ModelMesh chosen_part_;


        };

    } // top
} // da::sha
#endif //TOP3D_IRREGULARMESH_H
