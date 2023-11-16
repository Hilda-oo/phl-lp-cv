//
// Created by cflin on 4/21/23.
//

#ifndef TOP3D_BOUNDARY_H
#define TOP3D_BOUNDARY_H

#include "Util.h"
#include "Eigen/src/Geometry/AlignedBox.h"
#include "Mesh.h"


namespace da::sha {
    namespace top {
// index and coordinate system of a pixel
//                            4-------7
//                           /|      /|
//                          5-------6 |
//                          | |     | |
//                          | 0-----|-3
//                          |/      |/
//                          1-------2
//        z
//        |
//        |
//        |____ y
//       /
//      /
//     x
        class Boundary {
        public:
            Boundary(std::shared_ptr<Mesh> sp_mesh) : sp_mesh_(sp_mesh) {

            }

            Eigen::MatrixXi GetTopBoundary() {
                Eigen::VectorXi x_sequence = Eigen::VectorXi::LinSpaced(sp_mesh_->GetLx() + 1, 0, sp_mesh_->GetLx());
                Eigen::VectorXi y_sequence = Eigen::VectorXi::LinSpaced(sp_mesh_->GetLy() + 1, 0, sp_mesh_->GetLy());
                Eigen::VectorXi z_sequence = Eigen::Vector<int, 1>(sp_mesh_->GetLz());
                return GetChosenCoords(x_sequence, y_sequence, z_sequence);
            }

            Eigen::MatrixXi GetBottomBoundary() {
                Eigen::VectorXi x_sequence = Eigen::VectorXi::LinSpaced(sp_mesh_->GetLx() + 1, 0, sp_mesh_->GetLx());
                Eigen::VectorXi y_sequence = Eigen::VectorXi::LinSpaced(sp_mesh_->GetLy() + 1, 0, sp_mesh_->GetLy());
                Eigen::VectorXi z_sequence = Eigen::Vector<int, 1>(0);
                return GetChosenCoords(x_sequence, y_sequence, z_sequence);
            }

            Eigen::MatrixXi GetLeftBoundary() {
                Eigen::VectorXi x_sequence = Eigen::VectorXi::LinSpaced(sp_mesh_->GetLx() + 1, 0, sp_mesh_->GetLx());
                Eigen::VectorXi y_sequence = Eigen::Vector<int, 1>(0);
                Eigen::VectorXi z_sequence = Eigen::VectorXi::LinSpaced(sp_mesh_->GetLz() + 1, 0, sp_mesh_->GetLz());
                return GetChosenCoords(x_sequence, y_sequence, z_sequence);
            }

            Eigen::MatrixXi GetRightBoundary() {
                Eigen::VectorXi x_sequence = Eigen::VectorXi::LinSpaced(sp_mesh_->GetLx() + 1, 0, sp_mesh_->GetLx());
                Eigen::VectorXi y_sequence = Eigen::Vector<int, 1>(sp_mesh_->GetLy());
                Eigen::VectorXi z_sequence = Eigen::VectorXi::LinSpaced(sp_mesh_->GetLz() + 1, 0, sp_mesh_->GetLz());
                return GetChosenCoords(x_sequence, y_sequence, z_sequence);
            }

            Eigen::MatrixXi GetTopSurfaceCenter() {
                Eigen::MatrixXi coord(1, 3);
                coord << sp_mesh_->GetLx() / 2, sp_mesh_->GetLy() / 2, sp_mesh_->GetLz();
                return coord;
            }

            Eigen::MatrixXi GetCornerPoint(int i) {
                bool is_top = i / 4 == 1;
                i %= 4;
                Eigen::MatrixXi coord(1, 3);
                int lx = sp_mesh_->GetLx(), ly = sp_mesh_->GetLy(), lz = sp_mesh_->GetLz();
                switch (i) {
                    case 0:
                        coord << 0, 0, 0;
                        break;
                    case 1:
                        coord << lx, 0, 0;
                        break;
                    case 2:
                        coord << lx, ly, 0;
                        break;
                    case 3:
                        coord << 0, ly, 0;
                        break;
                    default:
                        spdlog::debug("never arrive!");
                }
                if (is_top) {
                    coord.col(2).setConstant(lz);
                }
                return coord;
            }

            Eigen::MatrixXi GetRightBottomMidPoint() {
                int lx = sp_mesh_->GetLx();
                int ly = sp_mesh_->GetLy();
                if (lx % 2 == 0) {
                    Eigen::MatrixXi bound_coords(1, 3);
                    bound_coords << lx / 2, ly, 0;
                    return bound_coords;
                }
                // else
                Eigen::MatrixXi bound_coords(2, 3);
                bound_coords << lx / 2, ly, 0,
                        lx / 2 + 1, ly, 0;
                return bound_coords;

            }

            Eigen::MatrixXi GetBoundaryCoordsInAbsoluteAlignedBox(const Eigen::AlignedBox3d &abs_box){
                Eigen::Vector3d min_point= MapWorldCoords2Local(abs_box.min());
                Eigen::Vector3d max_point= MapWorldCoords2Local(abs_box.max());
                Eigen::VectorXi i_min_point=(min_point.array()+0.0).cast<int>();
                Eigen::VectorXi i_max_point=(max_point.array()+1.0).cast<int>();
                i_min_point=i_min_point.cwiseMax(0).cwiseMin(Eigen::Vector3i(sp_mesh_->GetLx(),sp_mesh_->GetLy(),sp_mesh_->GetLz()));
                i_max_point=i_max_point.cwiseMax(0).cwiseMin(Eigen::Vector3i(sp_mesh_->GetLx(),sp_mesh_->GetLy(),sp_mesh_->GetLz()));
                Eigen::VectorXi x_seq     = Eigen::VectorXi::LinSpaced(i_max_point.x()-i_min_point.x()+1,i_min_point.x(),i_max_point.x());
                Eigen::VectorXi y_seq     = Eigen::VectorXi::LinSpaced(i_max_point.y()-i_min_point.y()+1,i_min_point.y(),i_max_point.y());
                Eigen::VectorXi z_seq     = Eigen::VectorXi::LinSpaced(i_max_point.z()-i_min_point.z()+1,i_min_point.z(),i_max_point.z());
//                                std::cout<<"x_seq:"<<x_seq<<std::endl<<std::endl;
//                                std::cout<<"y_seq:"<<y_seq<<std::endl<<std::endl;
//                                std::cout<<"z_seq:"<<z_seq<<std::endl<<std::endl;
                return ReduceNonBoundaryCoords(GetChosenCoords(x_seq,y_seq,z_seq));
            }
            Eigen::MatrixXi GetChosenCoordsByRelativeAlignedBox(const Eigen::AlignedBox3d & box){
                Eigen::Vector3d min_point=box.min();
                Eigen::Vector3d max_point=box.max();
                Eigen::Vector3d size_model(sp_mesh_->GetLx(),sp_mesh_->GetLy(),sp_mesh_->GetLz());
                min_point=min_point.array()*size_model.array();
                max_point=max_point.array()*size_model.array();
                Eigen::VectorXi i_min_point=(min_point.array()+0.5).cast<int>();
                Eigen::VectorXi i_max_point=(max_point.array()+0.5).cast<int>();
                i_min_point=i_min_point.cwiseMax(0).cwiseMin(Eigen::Vector3i(sp_mesh_->GetLx(),sp_mesh_->GetLy(),sp_mesh_->GetLz()));
                i_max_point=i_max_point.cwiseMax(0).cwiseMin(Eigen::Vector3i(sp_mesh_->GetLx(),sp_mesh_->GetLy(),sp_mesh_->GetLz()));
                Eigen::VectorXi x_seq     = Eigen::VectorXi::LinSpaced(i_max_point.x()-i_min_point.x()+1,i_min_point.x(),i_max_point.x());
                Eigen::VectorXi y_seq     = Eigen::VectorXi::LinSpaced(i_max_point.y()-i_min_point.y()+1,i_min_point.y(),i_max_point.y());
                Eigen::VectorXi z_seq     = Eigen::VectorXi::LinSpaced(i_max_point.z()-i_min_point.z()+1,i_min_point.z(),i_max_point.z());
//                std::cout<<"x_seq:"<<x_seq<<std::endl<<std::endl;
//                std::cout<<"y_seq:"<<y_seq<<std::endl<<std::endl;
//                std::cout<<"z_seq:"<<z_seq<<std::endl<<std::endl;
                return GetChosenCoords(x_seq,y_seq,z_seq);
            }

            Eigen::MatrixXi GetChosenCoords(const Eigen::VectorXi &x_sequence, const Eigen::VectorXi &y_sequence,
                                            const Eigen::VectorXi &z_sequence);
            
            Eigen::MatrixXd MapWorldCoords2TopologyCoords(const Eigen::MatrixXd& world_coord){
                Eigen::MatrixXd top_coords=MapWorldCoords2Local(world_coord);
                for(int i=0;i<top_coords.rows();++i){
                    top_coords(i,0)=std::max(std::min(top_coords(i,0),(double)sp_mesh_->GetLx()),0.0);
                    top_coords(i,1)=std::max(std::min(top_coords(i,1),(double)sp_mesh_->GetLy()),0.0);
                    top_coords(i,2)=std::max(std::min(top_coords(i,2),(double)sp_mesh_->GetLz()),0.0);
                }
                return top_coords;
            }
            Eigen::Vector3i ClampTopEleCoords(const Eigen::Vector3i & top_coord){
                return ClampTopEleCoords((Eigen::MatrixXi(1, 3) << top_coord).finished()).topRows(1);
            }
            Eigen::MatrixXi ClampTopEleCoords(const Eigen::MatrixXi & top_coords){
                Eigen::MatrixXi clamped_top_coords=top_coords;
                for(int i=0;i<top_coords.rows();++i){
                    clamped_top_coords(i,0)=std::max(std::min(top_coords(i,0),sp_mesh_->GetLx()-1),0);
                    clamped_top_coords(i,1)=std::max(std::min(top_coords(i,1),sp_mesh_->GetLy()-1),0);
                    clamped_top_coords(i,2)=std::max(std::min(top_coords(i,2),sp_mesh_->GetLz()-1),0);
                }
                return clamped_top_coords;
            }
            

        private:
              Eigen::Vector3d MapWorldCoords2Local(const Eigen::Vector3d & world_coord){
                  return MapWorldCoords2Local((Eigen::MatrixXd(1,3)<<world_coord.transpose()).finished()).row(0);
              }
              Eigen::MatrixXd MapWorldCoords2Local(const Eigen::MatrixXd & world_coords){
                Eigen::MatrixXd local_coords=world_coords.rowwise()-sp_mesh_->GetOrigin().transpose();
                local_coords*=1.0/sp_mesh_->GetPixelLen();
                return local_coords;
              }
              bool is_boundary_node(const Eigen::Vector3i & coord){
                if(coord.x()==0 || coord.x()==sp_mesh_->GetLx() || coord.y()==0 || coord.y()==sp_mesh_->GetLy() || coord.z()==0 || coord.z()==sp_mesh_->GetLz())
                    return true;
                static const Eigen::MatrixXi delta_neighbor_coord = (Eigen::MatrixXi(6, 3) <<
                                                            1, 0, 0,
                                                            -1, 0, 0,
                                                            0, 1, 0,
                                                            0, -1, 0,
                                                            0, 0, 1,
                                                            0, 0, -1
                                                            ).finished();
                for(int di=0;di<delta_neighbor_coord.rows();++di){
                    if(sp_mesh_->MapNodeCoord2Id(coord.transpose()+delta_neighbor_coord.row(di))(0)==-1){
                        return true;
                    }
                }
                return false;
              }
             Eigen::MatrixXi ReduceNonBoundaryCoords(const Eigen::MatrixXi &coords){
                    Eigen::VectorXi mapped_id=sp_mesh_->MapNodeCoord2Id(coords);
                    Eigen::MatrixXi masked_coord = coords;
                    int j = 0;
                    for (int i = 0; i < coords.rows(); ++i) {
                        if(is_boundary_node(coords.row(i))){
                            masked_coord.row(j++) = coords.row(i);
                        }
                    }
                    assert(j > 0);
                    return masked_coord.topRows(j);

             }
            Eigen::MatrixXi ReduceInvalidCoords(const Eigen::MatrixXi &coords) {
                Eigen::VectorXi node_ids = sp_mesh_->MapNodeCoord2Id(coords);
                Eigen::MatrixXi masked_coord = coords;
                int j = 0;
                for (int i = 0; i < node_ids.rows(); ++i) {
                    if (node_ids[i] != -1) {
                        masked_coord.row(j++) = coords.row(i);
                    }
                }
//                std::cout<<"coords:"<<coords<<std::endl;
//                std::cout<<"coords_masked:"<<coords<<std::endl;
                assert(j > 0);
                return masked_coord.topRows(j);

            }

            std::shared_ptr<Mesh> sp_mesh_;
        };

    } // top
} // da::sha
#endif //TOP3D_BOUNDARY_H
