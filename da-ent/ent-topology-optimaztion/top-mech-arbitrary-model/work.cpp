#include <spdlog/spdlog.h>
#include <memory>
#include "config.h"
#include "sha-base-framework/frame.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-topology-optimization-3d/Boundary.h"
#include "sha-topology-optimization-3d/IrregularMesh.h"
#include "sha-topology-optimization-3d/Top3d.h"
#include "sha-topology-optimization-3d/Util.h"
#include "igl/readOBJ.h"
void GenerateDensityFieldByArbitraryModel() {
  using namespace da;
  using namespace da::sha;

  fs_path base_path = WorkingAssetDirectoryPath();
  auto config_path  = base_path / "config_mech_arbitrary.json";
  log::info("Algo is working on path '{}'", base_path.string());

  // load config from json file
  Config config;
  if (!config.loadFromJSON(config_path.string())) {
    spdlog::error("error on reading json file!");
    exit(-1);
  }
  // config.backUpConfig((WorkingResultDirectoryPath() / "config_arbitrary.json").string());

  // set topology parameters
  auto para      = std::make_shared<top::CtrlPara>();
  para->max_loop = config.max_loop;
  para->volfrac  = config.volfrac;
  para->penal    = config.penal;
  para->r_min    = config.r_min;

  // set material parameters
  double E             = config.YM;
  double Poisson_ratio = config.PR;
  auto material        = std::make_shared<top::Material>(E, Poisson_ratio);

  // set mesh(arbitrary)
  auto mesh = std::make_shared<top::IrregularMesh>(WorkingAssetDirectoryPath()/ config.arbitraryModel, WorkingAssetDirectoryPath() / config.chosenModel,config.relativeLengthOfVoxel);

  // initialize Top3d
  top::Top3d top3d(para, material, mesh);

  // auxiliary class(Boundary) help to get boundary coordinates
  //      see the comments in Boundary.h for more information
  top::Boundary bdr(mesh);

  //      add Dirichlet boundary, see the comments in Top3d::AddDBC for more information
  for(auto &dbc:config.v_Dir){
      top3d.AddDBC(bdr.GetBoundaryCoordsInAbsoluteAlignedBox(dbc.box), dbc.dir);
  }
  //      add Neumann boundary, see the comments in Top3d::AddNBC for more information
  for(auto &nbc:config.v_Neu){
    top3d.AddNBC(bdr.GetBoundaryCoordsInAbsoluteAlignedBox(nbc.box),nbc.val);
  }

  // process topology optimization
  top::Tensor3d ten_rho = top3d.TopOptMainLoop();
  // postprocess 
  // extract txt or vtk
  write_tensor3d(da::WorkingResultDirectoryPath().string() + "/field_matrix.txt", ten_rho,mesh->GetOrigin(),mesh->GetOrigin()+mesh->GetLenBox());
  WriteTensorToVtk(da::WorkingResultDirectoryPath() / "field_matrix.vtk", ten_rho, mesh);
  // extract stress field
  auto v_tenosr=top3d.GetTensorOfStress(Eigen::Vector3d{0,1,2});
  write_tensor3d(da::WorkingResultDirectoryPath().string() + "/stressX_field_matrix.txt", v_tenosr[0],mesh->GetOrigin(),mesh->GetOrigin()+mesh->GetLenBox());
  write_tensor3d(da::WorkingResultDirectoryPath().string() + "/stressY_field_matrix.txt", v_tenosr[1],mesh->GetOrigin(),mesh->GetOrigin()+mesh->GetLenBox());
  write_tensor3d(da::WorkingResultDirectoryPath().string() + "/stressZ_field_matrix.txt", v_tenosr[2],mesh->GetOrigin(),mesh->GetOrigin()+mesh->GetLenBox());
  WriteTensorToVtk(da::WorkingResultDirectoryPath() / "stressX_field_matrix.vtk", v_tenosr[0], mesh);
  WriteTensorToVtk(da::WorkingResultDirectoryPath() / "stressY_field_matrix.vtk", v_tenosr[1], mesh);
  WriteTensorToVtk(da::WorkingResultDirectoryPath() / "stressZ_field_matrix.vtk", v_tenosr[2], mesh);
  if(!config.visualObjPath.empty()){
    // visual model with tensor
    //  rho
    {
      auto [mesh_obj,vertex_propty]=top::GetMeshVertexPropty(da::WorkingAssetDirectoryPath() / config.visualObjPath ,top3d.GetRhoFieldOneFilled(), bdr);
      std::string  vtk_path=(da::WorkingResultDirectoryPath()/"arbitrary_with_rho.vtk").string();
      WriteTriVTK(vtk_path, mesh_obj.mat_coordinates, mesh_obj.mat_faces,{}, std::vector<double>(vertex_propty.data(),vertex_propty.data()+vertex_propty.size()));
      spdlog::info("write vtk with rho to: {}",vtk_path);
      top::WriteVectorXd((da::WorkingResultDirectoryPath()/"arbitrary_rho.txt").string(), vertex_propty);
    }
    //  stressX
    {
      auto [mesh_obj,vertex_propty]=top::GetMeshVertexPropty(da::WorkingAssetDirectoryPath()/config.visualObjPath,v_tenosr[0], bdr);
      std::string  vtk_path=(da::WorkingResultDirectoryPath()/"arbitrary_with_stressX.vtk").string();
      WriteTriVTK(vtk_path, mesh_obj.mat_coordinates, mesh_obj.mat_faces,{}, std::vector<double>(vertex_propty.data(),vertex_propty.data()+vertex_propty.size()));
      spdlog::info("write vtk with stressX to: {}",vtk_path);
      top::WriteVectorXd((da::WorkingResultDirectoryPath()/"arbitrary_stressX.txt").string(), vertex_propty);
    }
    //  stressY
    {
      auto [mesh_obj,vertex_propty]=top::GetMeshVertexPropty(da::WorkingAssetDirectoryPath()/config.visualObjPath,v_tenosr[1], bdr);
      std::string  vtk_path=(da::WorkingResultDirectoryPath()/"arbitrary_with_stressY.vtk").string();
      WriteTriVTK(vtk_path, mesh_obj.mat_coordinates, mesh_obj.mat_faces,{}, std::vector<double>(vertex_propty.data(),vertex_propty.data()+vertex_propty.size()));
      spdlog::info("write vtk with stressY to: {}",vtk_path);
      top::WriteVectorXd((da::WorkingResultDirectoryPath()/"arbitrary_stressY.txt").string(), vertex_propty);
    }
    //  stressZ
    {
      auto [mesh_obj,vertex_propty]=top::GetMeshVertexPropty(da::WorkingAssetDirectoryPath()/config.visualObjPath,v_tenosr[2], bdr);
      std::string  vtk_path=(da::WorkingResultDirectoryPath()/"arbitrary_with_stressZ.vtk").string();
      WriteTriVTK(vtk_path, mesh_obj.mat_coordinates, mesh_obj.mat_faces,{}, std::vector<double>(vertex_propty.data(),vertex_propty.data()+vertex_propty.size()));
      spdlog::info("write vtk with stressZ to: {}",vtk_path);
      top::WriteVectorXd((da::WorkingResultDirectoryPath()/"arbitrary_stressZ.txt").string(), vertex_propty);
    }
  }
}
