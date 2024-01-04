

#include "physical_domain.h"

#include <igl/write_triangle_mesh.h>

#include "sha-base-framework/frame.h"
#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"

namespace da::sha {
PhysicalDomain::PhysicalDomain(const MatMesh3 &mesh,
                               const std::vector<Eigen::Matrix<double, 2, 3>> &p_NBCRelBBox,
                               const std::vector<Eigen::Vector3d> &p_NBCVal,
                               const std::vector<Eigen::Matrix<double, 2, 3>> &p_DBCRelBBox,
                               const std::vector<Eigen::Vector3d> &p_DBCVal)
    : NBCRelBBox(p_NBCRelBBox),
      NBCVal(p_NBCVal),
      DBCRelBBox(p_DBCRelBBox),
      DBCVal(p_DBCVal),
      mesh_(mesh) {
  Assert(mesh.mat_coordinates.cols() == 3);
  Assert(mesh.mat_faces.cols() == 3);

  numV = (int)mesh.mat_coordinates.rows();
  numF = (int)mesh.mat_faces.rows();
  V1.resize(numV, 3);

  bbox.row(0) = mesh.mat_coordinates.colwise().minCoeff();
  bbox.row(1) = mesh.mat_coordinates.colwise().maxCoeff();
  lenBBox     = bbox.row(1) - bbox.row(0);

  for (int triI = 0; triI < numF; ++triI) {
    Tri.emplace_back(Triangle(mesh.mat_coordinates(
        {mesh.mat_faces(triI, 0), mesh.mat_faces(triI, 1), mesh.mat_faces(triI, 2)}, Eigen::all)));
  }

  igl::fast_winding_number(mesh.mat_coordinates, mesh.mat_faces, 2, fwn_bvh);

  InitializeBoundaryConditions();

  spdlog::info("physical domain constructed " + std::to_string(numV) + " points, " +
               std::to_string(numF) + " triangles");
}

void PhysicalDomain::Update() {
  // update bbox
  bbox.row(0) = mesh_.mat_coordinates.colwise().minCoeff();
  bbox.row(1) = mesh_.mat_coordinates.colwise().maxCoeff();
  lenBBox     = bbox.row(1) - bbox.row(0);

  // update Triangle
  for (int triI = 0; triI < numF; ++triI) {
    Tri[triI].Update(
        mesh_.mat_coordinates({mesh_.mat_coordinates(triI, 0), mesh_.mat_coordinates(triI, 1),
                               mesh_.mat_coordinates(triI, 2)},
                              Eigen::all));
  }

  // update fwn_bvh
  igl::fast_winding_number(mesh_.mat_coordinates, mesh_.mat_faces, 2, fwn_bvh);
}

void PhysicalDomain::GetDomainID(const Eigen::MatrixXd &Q, Eigen::VectorXi &domainID) const {
  Assert(Q.cols() == 3);
  // winding number bigger than val will be seen inside physical domain, otherwise outside
  double val = 0.5;

  Eigen::VectorXd W;
  igl::fast_winding_number(fwn_bvh, 2, Q, W);
  Assert(W.rows() == Q.rows());

  domainID = (W.array() > val).cast<int>();
}

void PhysicalDomain::InitializeBoundaryConditions() {
  nNBC    = 0;
  int num = (int)NBCRelBBox.size();
  NBCBBox.resize(num);

  for (int i = 0; i < num; ++i) {
    NBCBBox[i].row(0) = bbox.row(0) + (lenBBox.array() * NBCRelBBox[i].row(0).array()).matrix();
    NBCBBox[i].row(1) = bbox.row(0) + (lenBBox.array() * NBCRelBBox[i].row(1).array()).matrix();
  }
  // log::info("hhhhhhh");
  // log::info("num V: {}", numV);
  // log::info("num: {}", num);
  // log::info("mat_coordinates: {} x {}", mesh.mat_coordinates.rows(),
  // mesh.mat_coordinates.cols()); log::info("NBCBBox: {}", NBCBBox.size());
  for (int vI = 0; vI < numV; ++vI) {
    for (int i = 0; i < num; ++i) {
      if (mesh_.mat_coordinates(vI, 0) >= NBCBBox[i](0, 0) &&
          mesh_.mat_coordinates(vI, 0) <= NBCBBox[i](1, 0) &&
          mesh_.mat_coordinates(vI, 1) >= NBCBBox[i](0, 1) &&
          mesh_.mat_coordinates(vI, 1) <= NBCBBox[i](1, 1) &&
          mesh_.mat_coordinates(vI, 2) >= NBCBBox[i](0, 2) &&
          mesh_.mat_coordinates(vI, 2) <= NBCBBox[i](1, 2)) {
        NBC.emplace_back(std::make_pair(mesh_.mat_coordinates.row(vI), NBCVal[i]));
        ++nNBC;
        break;
      }
    }
  }

  nDBC = 0;
  num  = (int)DBCRelBBox.size();
  DBCBBox.resize(num);

  for (int i = 0; i < num; ++i) {
    DBCBBox[i].row(0) = bbox.row(0) + (lenBBox.array() * DBCRelBBox[i].row(0).array()).matrix();
    DBCBBox[i].row(1) = bbox.row(0) + (lenBBox.array() * DBCRelBBox[i].row(1).array()).matrix();
  }

  for (int triI = 0; triI < numF; ++triI) {
    const Eigen::RowVector3d &v1 = mesh_.mat_coordinates.row(mesh_.mat_faces(triI, 0));
    const Eigen::RowVector3d &v2 = mesh_.mat_coordinates.row(mesh_.mat_faces(triI, 1));
    const Eigen::RowVector3d &v3 = mesh_.mat_coordinates.row(mesh_.mat_faces(triI, 2));
    for (int i = 0; i < num; ++i) {
      if (v1(0) >= DBCBBox[i](0, 0) && v1(0) <= DBCBBox[i](1, 0) && v1(1) >= DBCBBox[i](0, 1) &&
          v1(1) <= DBCBBox[i](1, 1) && v1(2) >= DBCBBox[i](0, 2) && v1(2) <= DBCBBox[i](1, 2) &&
          v2(0) >= DBCBBox[i](0, 0) && v2(0) <= DBCBBox[i](1, 0) && v2(1) >= DBCBBox[i](0, 1) &&
          v2(1) <= DBCBBox[i](1, 1) && v2(2) >= DBCBBox[i](0, 2) && v2(2) <= DBCBBox[i](1, 2) &&
          v3(0) >= DBCBBox[i](0, 0) && v3(0) <= DBCBBox[i](1, 0) && v3(1) >= DBCBBox[i](0, 1) &&
          v3(1) <= DBCBBox[i](1, 1) && v3(2) >= DBCBBox[i](0, 2) && v3(2) <= DBCBBox[i](1, 2)) {
        DBC.emplace_back(std::make_pair(triI, DBCVal[i]));
        ++nDBC;
        break;
      }
    }
  }
}

void PhysicalDomain::SetBoundaryConditions(
    int p_nNBC, const std::vector<std::pair<Eigen::RowVector3d, Eigen::Vector3d>> &p_NBC,
    int p_nDBC, const std::vector<std::pair<int, Eigen::Vector3d>> &p_DBC,
    double p_penaltyNitsche) {
  nNBC           = p_nNBC;
  NBC            = p_NBC;
  nDBC           = p_nDBC;
  DBC            = p_DBC;
  use_Nitsche    = true;
  penaltyNitsche = p_penaltyNitsche;
}

void PhysicalDomain::WriteNBCToVtk(const fs_path &path) {
  spdlog::info("number of NBC points: {}", nNBC);
  Eigen::MatrixXd NBC_V(nNBC, 3);
  for (int NBCI = 0; NBCI < nNBC; ++NBCI) {
    NBC_V.row(NBCI) = NBC[NBCI].first;
  }
  sha::WritePointsToVtk(path, NBC_V);
}

void PhysicalDomain::WriteDBCToObj(const fs_path &path) {
  Eigen::MatrixXd Vdebug(numV, 3);
  Eigen::MatrixXi Fdebug(nDBC, 3);
  std::map<int, int> mp;
  int cnt = 0;
  spdlog::info("number of DBC triangles: {}", nDBC);

  for (int DBCI = 0; DBCI < nDBC; ++DBCI) {
    int triI = DBC[DBCI].first;
    int vI   = mesh_.mat_faces(triI, 0);
    int vJ   = mesh_.mat_faces(triI, 1);
    int vK   = mesh_.mat_faces(triI, 2);
    if (!mp.count(vI)) {
      mp[vI]          = cnt;
      Vdebug.row(cnt) = mesh_.mat_coordinates.row(vI);
      ++cnt;
    }
    if (!mp.count(vJ)) {
      mp[vJ]          = cnt;
      Vdebug.row(cnt) = mesh_.mat_coordinates.row(vJ);
      ++cnt;
    }
    if (!mp.count(vK)) {
      mp[vK]          = cnt;
      Vdebug.row(cnt) = mesh_.mat_coordinates.row(vK);
      ++cnt;
    }
    Fdebug.row(DBCI) = Eigen::RowVector3i(mp[vI], mp[vJ], mp[vK]);
  }
  Vdebug.conservativeResize(cnt, 3);
  igl::write_triangle_mesh(path.string(), Vdebug, Fdebug);
}

void PhysicalDomain::WriteV1MeshToObj(const fs_path &path) {
  igl::write_triangle_mesh(path.string(), V1, mesh_.mat_faces);
}
}  // namespace da::sha