//
// Created by cflin on 4/24/23.
//

#include "IrregularMesh.h"
#include <igl/march_cube.h>
#include <igl/readSTL.h>
#include <igl/read_triangle_mesh.h>
#include <igl/signed_distance.h>
#include <spdlog/spdlog.h>
#include "Eigen/src/Core/Matrix.h"
namespace da::sha {
    namespace top {
        IrregularMesh::IrregularMesh(const fs_path &arbitrary_stl_path,const fs_path &chosen_stl_path,double relative_length_of_voxel,int dofs_each_node): Mesh() {
            // get num_pixels
            const double percentage_of_min_len = relative_length_of_voxel;
            igl::read_triangle_mesh(arbitrary_stl_path.string(),arbitrary_mesh_.points,arbitrary_mesh_.surfaces);

            ModelMesh chosen_part;
            igl::read_triangle_mesh(chosen_stl_path.string(),chosen_part_.points,chosen_part_.surfaces);

            origin_= min_point_box_ = arbitrary_mesh_.points.colwise().minCoeff();
            Eigen::Vector3d box_max_point = arbitrary_mesh_.points.colwise().maxCoeff();

            len_box_ = box_max_point - min_point_box_;
            len_pixel_ = len_box_.minCoeff() * percentage_of_min_len;
            Eigen::Vector3d d_num_pixels = len_box_ / len_pixel_;
            Eigen::Vector3i num_pixels(std::ceil(d_num_pixels(0)), std::ceil(d_num_pixels(1)),
                                       std::ceil(d_num_pixels(2)));

            DOFS_EACH_NODE=dofs_each_node;
            lx_ = num_pixels(0);
            ly_ = num_pixels(1);
            lz_ = num_pixels(2);
            num_node_ = (lx_ + 1) * (ly_ + 1) * (lz_ + 1);
            num_ele_ = lx_ * ly_ * lz_;

            ten_ele_coord2ele_id_ = Tensor3i(lx_, ly_, lz_);
            ten_ele_coord2ele_id_.setConstant(-1);
            ten_node_coord2node_id_ = Tensor3i(lx_ + 1, ly_ + 1, lz_ + 1);
            ten_node_coord2node_id_.setConstant(-1);



            auto LocalCoord2World = [&](const Eigen::MatrixXd &local_coord)->Eigen::MatrixXd {
                Eigen::MatrixXd world_coord = (local_coord* len_pixel_).rowwise() + min_point_box_.transpose();
                return world_coord;
            };

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
            // precompute coords needed to query
            const int SAMPLE_POINTS_EACH_DIM=3;
            double space=1.0/SAMPLE_POINTS_EACH_DIM;
            
            //      fill local_coords
            Eigen::MatrixXd local_coords(SAMPLE_POINTS_EACH_DIM*SAMPLE_POINTS_EACH_DIM*SAMPLE_POINTS_EACH_DIM+delta_coord.rows(),3);
            int sample_cnt=0;
            for(double spci=space/2.0;spci<1.0;spci+=space){
                for(double spcj=space/2.0;spcj<1.0;spcj+=space){
                    for(double spck=space/2.0;spck<1.0;spck+=space){
                        local_coords.row(sample_cnt++)=Eigen::Vector3d(spci,spcj,spck);
                    }
                }
            }
            //      append cornor coords
            for(int i=0;i<delta_coord.rows();++i){
                local_coords.row(sample_cnt++)=delta_coord.row(i).cast<double>();
            }

            Eigen::MatrixXd coords_to_query(num_ele_*sample_cnt,3);
            int coord_cnt=0;
            for (int k = 0; k < lz_; ++k) {
              for (int j = 0; j < ly_; ++j) {
                for (int i = 0; i < lx_; ++i) {
                    for(int si=0;si<sample_cnt;++si){
                        coords_to_query.row(coord_cnt++)=Eigen::Vector3d(i,j,k).transpose()+local_coords.row(si);
                    }
                }
              }
            }

            Eigen::VectorXd arbi_sdf_precomputed=EvaluateArbitrarySDF(LocalCoord2World(coords_to_query));
            Eigen::VectorXd chosen_sdf_precomputed=EvaluateChosenSDF(LocalCoord2World(coords_to_query));

            Eigen::VectorXd rho_ele_arbi=Eigen::VectorXd::Zero(num_ele_);
            for(int i=0;i<rho_ele_arbi.size();++i){
                for(int si=0;si<sample_cnt;++si){
                    rho_ele_arbi(i)+=arbi_sdf_precomputed(i*sample_cnt+si)>0.0 ? 1.0/sample_cnt:0.0;
                }
            }
            Eigen::VectorXd rho_ele_chosen=Eigen::VectorXd::Zero(num_ele_);
            for(int i=0;i<rho_ele_chosen.size();++i){
                for(int si=0;si<sample_cnt;++si){
                    rho_ele_chosen(i)+=chosen_sdf_precomputed(i*sample_cnt+si)>0.0 ? 1.0/sample_cnt:0.0;
                }
            }
            // std::cout<<rho_ele_arbi.topRows(50);
            // get num_pixel_ && fill pixel id
            std::vector<int> v_chosen_ele_id;
            std::vector<double> v_rho_non_empty;
            int cnt_pixel = 0;
            for (int k = 0; k < lz_; ++k) {
                for (int j = 0; j < ly_; ++j) {
                    for (int i = 0; i < lx_; ++i) {
                        int idx=k*lx_*ly_+j*lx_+i;
                        if (rho_ele_arbi(idx)>0.0) {
                            if(rho_ele_chosen(idx)>0.0){
                                v_chosen_ele_id.push_back(cnt_pixel);
                            }
                            v_rho_non_empty.push_back(true?1.0:rho_ele_arbi(idx));
                            ten_ele_coord2ele_id_(i, j, k) = cnt_pixel++;
                            for (int di = 0; di < delta_coord.rows(); ++di) {
                                Eigen::Vector3i cur_delta_coord = delta_coord.row(di);
                                ten_node_coord2node_id_(i + cur_delta_coord(0), j + cur_delta_coord(1),
                                                        k + cur_delta_coord(2)) = 1;
                            }
                        }
                    }
                }
            }
            chosen_ele_id_=Eigen::Map<decltype(chosen_ele_id_)>(v_chosen_ele_id.data(),v_chosen_ele_id.size());
            init_ele_rho_=Eigen::Map<Eigen::VectorXd>(v_rho_non_empty.data(),v_rho_non_empty.size());
            num_pixel_ = cnt_pixel;

            // get_num_node_pixel && fill node_id
            int cnt_node_pixel = 0;
            std::vector<Eigen::Vector3i> v_node_coords;
            for (int k = 0; k < lz_ + 1; ++k) {
                for (int j = 0; j < ly_ + 1; ++j) {
                    for (int i = 0; i < lx_ + 1; ++i) {
                        if (ten_node_coord2node_id_(i, j, k) == 1) {
                            ten_node_coord2node_id_(i, j, k) = cnt_node_pixel;
                            v_node_coords.push_back({i,j,k});
                            ++cnt_node_pixel;
                        }
                    }
                }
            }
            num_node_pixel_ = cnt_node_pixel;
            // fill mat_node_id 2 node_coord
            mat_node_id2node_coord_=Eigen::MatrixXi(v_node_coords.size(),3);
            for(int i=0;i<v_node_coords.size();++i){
                mat_node_id2node_coord_.row(i)=v_node_coords[i];
            }

            // fill mat_ele_id2dofs_
            mat_ele_id2dofs_.resize(num_pixel_, NUM_NODES_EACH_ELE * DOFS_EACH_NODE);
            mat_ele_id2node_id_.resize(num_pixel_,NUM_NODES_EACH_ELE);
            for (int k = 0; k < lz_; ++k) {
                for (int j = 0; j < ly_; ++j) {
                    for (int i = 0; i < lx_; ++i) {
                        int cur_ele_id = ten_ele_coord2ele_id_(i, j, k);
                        if (cur_ele_id == -1) {
                            continue;
                        }
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
            // fill valid_ele_idx_
            valid_ele_idx_.resize(num_pixel_);
            Tensor3i ten_tmp_col = ten_ele_coord2ele_id_.reshape(Eigen::array<Eigen::DenseIndex , 3>{num_ele_, 1, 1});
            cnt_pixel = 0;
            for (int i = 0; i < ten_tmp_col.size(); ++i) {
                if (ten_tmp_col(i, 0, 0) != -1) {
                    valid_ele_idx_(cnt_pixel) = i;
                    ++cnt_pixel;
                }
            }
            assert(cnt_pixel == num_pixel_);
        }
        double IrregularMesh::EvaluateArbitrarySDF(const Eigen::Vector3d &point) {
            return EvaluateArbitrarySDF((Eigen::MatrixXd(1,3)<<point.transpose()).finished())(0);
        }

        double IrregularMesh::EvaluateChosenSDF(const Eigen::Vector3d &point){
            return EvaluateChosenSDF((Eigen::MatrixXd(1,3)<<point.transpose()).finished())(0);
        }

        Eigen::VectorXd IrregularMesh::EvaluateArbitrarySDF(const Eigen::MatrixXd &points) {
          Eigen::VectorXd S,I;
          Eigen::MatrixXd C,N;
          igl::signed_distance(points,arbitrary_mesh_.points,arbitrary_mesh_.surfaces,igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL,S,I,C,N);
          return -S;
        }
        Eigen::VectorXd IrregularMesh::EvaluateChosenSDF(const Eigen::MatrixXd &points) {
          Eigen::VectorXd S,I;
          Eigen::MatrixXd C,N;
          igl::signed_distance(points,chosen_part_.points,chosen_part_.surfaces,igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL,S,I,C,N);
          return -S;
        }
        } // top
} // da::sha