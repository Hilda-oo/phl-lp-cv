/*
 * @Author: lab pc yjxkwp@foxmail.com
 * @Date: 2023-04-28 22:24:13
 * @LastEditors: lab pc yjxkwp@foxmail.com
 * @LastEditTime: 2023-04-29 12:17:53
 * @FilePath: /designauto/da-sha/sha-topology-optimization-3d/Mesh.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
//
// Created by cflin on 4/20/23.
//

#include "Mesh.h"
#include "sha-topology-optimization-3d/Util.h"

namespace da::sha {
    namespace top {
        Mesh::Mesh(int len_x, int len_y, int len_z,int dofs_each_node) : DOFS_EACH_NODE(dofs_each_node), lx_(len_x), ly_(len_y), lz_(len_z),
                                                                        num_node_((lx_ + 1) * (ly_ + 1) * (lz_ + 1)),
                                                                        num_ele_(lx_ * ly_ * lz_), origin_(0,0,0), len_pixel_(1.0), len_box_(len_x,len_y,len_z){
            valid_ele_idx_ = Eigen::VectorXi::LinSpaced(num_ele_, 0, num_ele_ - 1);
            init_ele_rho_=Eigen::VectorXd::Ones(num_ele_);

            // ten_node_coord2node_id_ = Tensor3i(num_node_, 1, 1);
            // for (int i = 0; i < num_node_; ++i) {
            //     ten_node_coord2node_id_(i, 0, 0) = i;
            // }
            // ten_node_coord2node_id_ = ten_node_coord2node_id_.reshape(Eigen::array<Eigen::DenseIndex, 3>{lx_ + 1, ly_ + 1, lz_ + 1});

            // get node_coord <---> node_id
            int cnt_node_pixel = 0;
            std::vector<Eigen::Vector3i> v_node_coords;
            ten_node_coord2node_id_=Tensor3i(lx_ + 1, ly_ + 1, lz_ + 1);
            for (int k = 0; k < lz_ + 1; ++k) {
                for (int j = 0; j < ly_ + 1; ++j) {
                    for (int i = 0; i < lx_ + 1; ++i) {
                        ten_node_coord2node_id_(i, j, k) = cnt_node_pixel;
                        v_node_coords.push_back({i,j,k});
                        ++cnt_node_pixel;
                    }
                }
            }
            // fill mat_node_id 2 node_coord
            mat_node_id2node_coord_=Eigen::MatrixXi(v_node_coords.size(),3);
            for(int i=0;i<v_node_coords.size();++i){
                mat_node_id2node_coord_.row(i)=v_node_coords[i];
            }
            //

            ten_ele_coord2ele_id_ = Tensor3i(num_ele_, 1, 1);
            for (int i = 0; i < num_ele_; ++i) {
                ten_ele_coord2ele_id_(i, 0, 0) = i;
            }
            ten_ele_coord2ele_id_ = ten_ele_coord2ele_id_.reshape(Eigen::array<Eigen::DenseIndex, 3>{lx_, ly_, lz_});

            mat_ele_id2dofs_.resize(num_ele_, NUM_NODES_EACH_ELE * DOFS_EACH_NODE);
            mat_ele_id2node_id_.resize(num_ele_,NUM_NODES_EACH_ELE);
            static const Eigen::MatrixXi delta_coord = (Eigen::MatrixXi(8, 3) <<
                                                                              0, 0, 0,
                    1, 0, 0,
                    1, 1, 0,
                    0, 1, 0,
                    0, 0, 1,
                    1, 0, 1,
                    1, 1, 1,
                    0, 1, 1
            ).finished();
            for (int k = 0; k < lz_; ++k) {
                for (int j = 0; j < ly_; ++j) {
                    for (int i = 0; i < lx_; ++i) {
                        int cur_ele_id = ten_ele_coord2ele_id_(i, j, k);
                        Eigen::MatrixXi world_node_coords = delta_coord.rowwise() + Eigen::RowVector3i(i, j, k);
                        assert(world_node_coords.rows() == 8 && world_node_coords.cols() == 3);
                        Eigen::Vector<int, 8> node_ids = ten_node_coord2node_id_(world_node_coords);
                        mat_ele_id2node_id_.row(cur_ele_id)=node_ids;
                        for (int nodi = 0; nodi < NUM_NODES_EACH_ELE; ++nodi) {
                            for(int dofi=0; dofi < DOFS_EACH_NODE; ++dofi){
                                mat_ele_id2dofs_(cur_ele_id, DOFS_EACH_NODE * nodi + dofi)= DOFS_EACH_NODE * node_ids(nodi) + dofi;
                            }
                        }
                    }
                }
            }


        }
    } // top
} // da::sha