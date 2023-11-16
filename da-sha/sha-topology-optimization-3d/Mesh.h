//
// Created by cflin on 4/20/23.
//

#ifndef TOP3D_MESH_H
#define TOP3D_MESH_H

#include "Util.h"
#include "unsupported/Eigen/KroneckerProduct"

namespace da::sha {
    namespace top {


        class Mesh {
        public:
            Mesh():origin_(0,0,0),len_pixel_(1.0){};

            Mesh(int len_x, int len_y, int len_z,int dofs_each_node=3);

            int GetLx() const {
                return lx_;
            }

            int GetLy() const {
                return ly_;
            }

            int GetLz() const {
                return lz_;
            }

            Eigen::VectorXi GetPixelIdx() const {
                return valid_ele_idx_;
            }

            virtual int GetNumDofs() const {
                return num_node_ * DOFS_EACH_NODE;
            }

            virtual int GetNumEles() const {
                return num_ele_;
            }

            virtual int GetNumNodes() const {
                return num_node_;
            }

            Eigen::MatrixXi GetEleId2DofsMap() const {
                return mat_ele_id2dofs_;
            }

            Eigen::MatrixXi GetEleId2NodeIdsMap()const{
                return mat_ele_id2node_id_;
            }

            Eigen::MatrixXi GetNodeId2CoordMap()const{
                return mat_node_id2node_coord_;
            }

            Tensor3i GetEleCoord2IdTensor() const {
                return ten_ele_coord2ele_id_;
            }

            Eigen::VectorXi GetChosenEleIdx()const{
                return chosen_ele_id_;
            }
            Eigen::Vector3d GetOrigin()const {
                return origin_;
            }
            Eigen::Vector3d GetLenBox()const{
                return len_box_;
            }
            double GetPixelLen()const{
                return len_pixel_;
            }
            int Get_NUM_NODES_EACH_ELE()const{
                return NUM_NODES_EACH_ELE;
            }
            int Get_DOFS_EACH_NODE()const{
                return DOFS_EACH_NODE;
            }
            Eigen::VectorXd GetInitEleRho() const{
                return init_ele_rho_;
            }

            Eigen::VectorXi MapNodeCoord2Id(const Eigen::MatrixXi &node_coords) const {
                return ten_node_coord2node_id_(node_coords);
            }

            Eigen::VectorXi MapEleCoord2Id(const Eigen::MatrixXi &ele_coords) const {
                return ten_ele_coord2ele_id_(ele_coords);
            }

            Eigen::MatrixXi MapEleId2Dofs(const Eigen::VectorXi &ele_ids) const {
                return mat_ele_id2dofs_(ele_ids, all);
            }

            Eigen::VectorXi MapEleId2Dofs(const int ele_id) const {
                return mat_ele_id2dofs_(ele_id, all);
            }

            



        protected:
            static const int NUM_NODES_EACH_ELE = 8;
            int DOFS_EACH_NODE= 3;
            int lx_;
            int ly_;
            int lz_;
            int num_ele_;
            int num_node_;
            Eigen::VectorXi valid_ele_idx_;
            Tensor3i ten_node_coord2node_id_;
            Eigen::MatrixXi mat_node_id2node_coord_;// num_node x 3
            Tensor3i ten_ele_coord2ele_id_;
            Eigen::MatrixXi mat_ele_id2dofs_;// num_ele_x (8*3)
            Eigen::MatrixXi mat_ele_id2node_id_;// num_ele x 8
            
            Eigen::VectorXi chosen_ele_id_;

            Eigen::Vector3d origin_;
            Eigen::Vector3d len_box_;
            double len_pixel_;
            Eigen::VectorXd init_ele_rho_;

        };

    } // top
} // da::sha
#endif //TOP3D_MESH_H
