//
// Created by cflin on 4/20/23.
//

#ifndef TOP3D_TOP3D_H
#define TOP3D_TOP3D_H

#include <set>
#include <memory>
#include <string>
#include <filesystem>
#include <Eigen/CholmodSupport>
#include <SuiteSparseQR.hpp>
#include <spdlog/spdlog.h>
#include <Eigen/UmfPackSupport>
#include <mma/MMASolver.h>
#include "Util.h"
#include "Material.hpp"
#include "Mesh.h"
#include "Boundary.h"
#include "IrregularMesh.h"
namespace da::sha {
    namespace top {
        struct CtrlPara {
            double volfrac = 0.5;
            double penal = 1.0;
            int max_loop = 100;
            double r_min = 2.0;

            double tol_x = 0.01;
            double E_factor = 1e-9;
        };

        class Top3d {
        public:
            Top3d(std::shared_ptr<CtrlPara> sp_para, std::shared_ptr<Material> sp_material,
                  std::shared_ptr<Mesh> sp_mesh) : sp_para_(sp_para
            ), sp_material_(sp_material), sp_mesh_(sp_mesh) {
                F_ = SpMat(sp_mesh_->GetNumDofs(), 1);
                K_ = SpMat(sp_mesh_->GetNumDofs(), sp_mesh_->GetNumDofs());
                spdlog::info("DOF: {}", K_.rows());
                spdlog::info("start to precompute...");
                Precompute();
            }
            /// add Dirichlet boundary condition
            /// \param DBC_coords nx3, mesh coordinates to fixed
            /// \param directions 1x[3|1], directions([0|1|2])!=0 means that fixing [x|y|z] direction; Only directions(0) will be use for head_condition
            void AddDBC(const Eigen::MatrixXi &DBC_coords, const Eigen::Vector3i &directions) {
                Eigen::VectorXi node_id_to_fix = sp_mesh_->MapNodeCoord2Id(DBC_coords);
                for (int i = 0; i < node_id_to_fix.size(); ++i) {
                    for (int dir = 0; dir < sp_mesh_->Get_DOFS_EACH_NODE(); ++dir)
                        if (directions[dir])
                            dofs_to_fixed_.insert(GetDof(node_id_to_fix[i],dir));
                }
            }
            void AddDBC(const Eigen::MatrixXd & DBC_coords,const Eigen::Vector3i &directions){
                Eigen::MatrixXi low_DBC_coords=DBC_coords.cast<int>();
                Eigen::MatrixXi around_low=Eigen::MatrixXi(low_DBC_coords.rows()*8,3);
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
                for(int i=0;i<low_DBC_coords.rows();++i){
                    for(int di=0;di<8;++di){
                        around_low.row(i*8+di)=low_DBC_coords.row(i)+delta_coord.row(di);
                    }
                }
                AddDBC(around_low,directions);
            }

            void AddNBC(const Eigen::MatrixXd &NBC_coords,const Eigen::Vector3d & forces){
                Eigen::MatrixXi low_DBC_coords=NBC_coords.cast<int>();
                Eigen::MatrixXd local_coords=(NBC_coords-low_DBC_coords.cast<double>()).array()*2.0-1.0;// map coords to [-1,1]
                Eigen::VectorXi ele_ids=sp_mesh_->MapEleCoord2Id(low_DBC_coords);
                for(int i=0;i<ele_ids.size();++i){
                    if(ele_ids(i)==-1){
                        spdlog::warn("force is on empty pixel!");
                        continue;
                    }
                    Eigen::VectorXi dof_of_the_ele=sp_mesh_->MapEleId2Dofs(ele_ids(i));
                    Eigen::Matrix<double, 3, 24>  N;
                    // sp_material_->computeN(Eigen::RowVector3d(local_coords(i,0),local_coords(i,1),local_coords(i,2)), N);//TODO: fixme
                    Eigen::VectorXd node_forces=N.transpose()*forces;
                    for(int dofi=0;dofi<dof_of_the_ele.size();++dofi)
                        F_.coeffRef(dof_of_the_ele(i),0)+=node_forces(i);
                }
            }

            /// add Neumann boundary condition
            /// \param NBC_coords nx3, mesh coordinates to load forces
            /// \param forces 1x[3|1], forces([0|1|2]) means that load corresponding force in [x|y|z] direction; Only forces(0) will be use for head_condition
            void AddNBC(const Eigen::MatrixXi &NBC_coords, const Eigen::Vector3d &forces) {
                Eigen::VectorXi node_id_to_load = sp_mesh_->MapNodeCoord2Id(NBC_coords);
                for (int i = 0; i < node_id_to_load.size(); ++i) {
                    for (int dir = 0; dir <sp_mesh_->Get_DOFS_EACH_NODE(); ++dir)
                        if (forces[dir])
                            F_.coeffRef(GetDof(node_id_to_load[i], dir), 0) += forces(dir);
                }
            }

            Tensor3d TopOptMainLoop();


        std::vector<Tensor3d> GetTensorOfStress(const Eigen::VectorXd &which_col_of_stress);
        Tensor3d GetRhoFieldOneFilled()const{
            return rho_field_one_filled_;
        }
        Tensor3d GetRhoFieldZeroFilled()const{
            return rho_field_zero_filled_;
        }
        private:
            Tensor3d GetTensorFromCol(const Eigen::VectorXd & proprty_col);


            void IntroduceFixedDofs(Eigen::SparseMatrix<double> &K, Eigen::SparseMatrix<double> &F) {
                for (auto dof: dofs_to_fixed_) {
                    K.coeffRef(dof, dof) *= 1e10;
                    F.coeffRef(dof, 0) = 0;
                }
            }

            void Precompute();

            int GetDof(int node_id, int dir) {
                return node_id * sp_mesh_->Get_DOFS_EACH_NODE() + dir;
            }

        private:
            std::shared_ptr<CtrlPara> sp_para_;
            std::shared_ptr<Material> sp_material_;
            std::shared_ptr<Mesh> sp_mesh_;
        private:
            SpMat F_;
            Eigen::VectorXi iK_, jK_;
            Eigen::VectorXd sKe_;

            Eigen::MatrixXd Ke_;
            SpMat K_;

            std::set<int> dofs_to_fixed_;
            SpMat H_;
            Eigen::VectorXd Hs_;
            // result
            Eigen::VectorXd U_;
            Eigen::VectorXd rho_;
            Tensor3d rho_field_one_filled_;
            Tensor3d rho_field_zero_filled_;


        };

    } // top
} // da::sha
#endif //TOP3D_TOP3D_H
