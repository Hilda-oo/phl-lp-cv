//
// Created by cflin on 4/20/23.
//

#include "Util.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <ratio>
#include "Boundary.h"
#include "Mesh.h"
#include "Top3d.h"
#include "igl/readOBJ.h"
#include "sha-surface-mesh/matmesh.h"
namespace da::sha {
namespace top {
void WriteTensorToVtk(const fs_path &file_path, const Tensor3d &t3, std::shared_ptr<Mesh> sp_mesh) {
  HexahedralMatMesh matmesh;
  Eigen::Vector3d origin = sp_mesh->GetOrigin();
  double pixel_len       = sp_mesh->GetPixelLen();
  matmesh.mat_coordinates =
      (sp_mesh->GetNodeId2CoordMap().cast<double>() * pixel_len).rowwise() + origin.transpose();
  matmesh.mat_hexahedrons = sp_mesh->GetEleId2NodeIdsMap();

  std::vector<double> v_ele_data;
  Eigen::VectorXi pixel_idx = sp_mesh->GetPixelIdx();
  for (int i = 0; i < pixel_idx.size(); ++i) {
    v_ele_data.push_back(*(t3.data() + pixel_idx[i]));
  }
  assert(matmesh.IsHexahedral());
  WriteHexahedralMatmeshToVtk(file_path, matmesh, std::vector<double>(), v_ele_data);
}

std::pair<MatMesh3, Eigen::VectorXd> GetMeshVertexPropty(const fs_path &fs_obj,
                                                    const Tensor3d &ten_rho_field,
                                                    Boundary &r_bdr,bool offset_flg) {
  MatMesh3 obj_mesh;
  igl::readOBJ(fs_obj.string(), obj_mesh.mat_coordinates, obj_mesh.mat_faces);
  if(offset_flg){
    // for box
    for(int i=0;i<obj_mesh.mat_coordinates.rows();++i){
      obj_mesh.mat_coordinates(i,0)+=ten_rho_field.dimension(0)/2.0;
      obj_mesh.mat_coordinates(i,1)+=ten_rho_field.dimension(1)/2.0;
      obj_mesh.mat_coordinates(i,2)+=ten_rho_field.dimension(2)/2.0;
    }
  }
  // trilinear interploation
  Eigen::MatrixXd top_space_coords = r_bdr.MapWorldCoords2TopologyCoords(obj_mesh.mat_coordinates);
  Eigen::MatrixXi center_coords    = (top_space_coords.array() + 0.5).cast<int>();
  static Eigen::MatrixXi delta_coords =
      (Eigen::MatrixXi(8, 3) << -1, -1, -1, 0, -1, -1, 0, 0, -1, -1, 0, -1,

       -1, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0)
          .finished();

  Eigen::VectorXd rho_each_vertex(top_space_coords.rows());
  for (int i = 0; i < center_coords.rows(); ++i) {
    // each coord
    Eigen::Vector3i i_center_coord = center_coords.row(i);
    Eigen::MatrixXi around_coords  = i_center_coord.transpose().replicate(8, 1) + delta_coords;
    around_coords                  = r_bdr.ClampTopEleCoords(around_coords);
    Eigen::VectorXd rho_around     = ten_rho_field(around_coords);  // 8x1
    double rho_max=rho_around.maxCoeff();
    for(int ri=0;ri<rho_around.size();++ri){
      if(rho_around(ri)<1e-3){
          rho_around(ri)=rho_max;
      }
    }

    Eigen::Vector3d d_center_coord = top_space_coords.row(i);
    Eigen::Vector3d ratio = (d_center_coord - i_center_coord.cast<double>()).array() + 0.5;  // 3x1

    static auto LinearInterp = [](const Eigen::Vector2d &props, double ratio) {
      return props(1) * ratio + props(0) * (1.0 - ratio);
    };
    static auto BilinearInterp = [](const Eigen::Vector4d &props, const Eigen::Vector2d &ratios) {
      Eigen::Vector2d y_props;
      y_props(0) = LinearInterp({props(0), props(1)}, ratios.x());
      y_props(1) = LinearInterp({props(3), props(2)}, ratios.x());
      return LinearInterp(y_props, ratios.y());
    };
    static auto TrilinearInterp = [](const Eigen::Vector<double, 8> &props,
                                     const Eigen::Vector3d &ratios) {
      Eigen::Vector2d xy_props;
      xy_props(0) = BilinearInterp(props.topRows(4), ratios.topRows(2));
      xy_props(1) = BilinearInterp(props.bottomRows(4), ratios.topRows(2));
      return LinearInterp(xy_props, ratios.z());
    };
    rho_each_vertex(i)=TrilinearInterp(rho_around, ratio);
    // assert(rho_each_vertex(i)>=0 && rho_each_vertex(i)<=1);
  }

  return {obj_mesh, rho_each_vertex};
}

}  // namespace top
}  // namespace da::sha