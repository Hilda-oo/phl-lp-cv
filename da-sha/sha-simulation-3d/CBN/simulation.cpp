#include "simulation.h"

#include "Eigen/src/Core/Matrix.h"
#include "cpt-linear-solver/cholmod_solver.h"
#include "cpt-linear-solver/eigen_solver.h"
#include "sha-base-framework/frame.h"
#include "sha-io-foundation/data_io.h"
#include "sha-io-foundation/mesh_io.h"
#include "sha-simulation-utils/material_utils.h"
#include "sha-simulation-utils/other_utils.h"
#include "sha-simulation-utils/shape_function_utils.h"

#include "utils.h"

#include <oneapi/tbb.h>
#include <spdlog/spdlog.h>
#include <cassert>
#include <cstdlib>

#define DIM_ 3

namespace da::sha {
CBNSimulator::CBNSimulator(double p_YM1, double p_YM0, double p_PR, double p_penaltyYM,
                           std::shared_ptr<PhysicalDomain> p_physicalDomain,
                           const std::shared_ptr<NestedBackgroundMesh> &nested_background,
                           bool p_handlePhysicalDomain)
    : YM1(p_YM1),
      YM0(p_YM0),
      penaltyYM(p_penaltyYM),
      physicalDomain(std::move(p_physicalDomain)),
      nested_background_(nested_background),
      handlePhysicalDomain(p_handlePhysicalDomain) {
  ComputeElasticMatrix(1.0, p_PR, D_);
  // setup GP and GW for BC
  GetGaussQuadratureCoordinates(integrationOrder_BC, GP_BC);
  GetGaussQuadratureWeights(integrationOrder_BC, GW_BC);

  PreprocessBackgroundMesh();

  // polygon.clear();
  // boost::transform(nested_background->nested_cells_, std::back_inserter(polygon),
  //                  [&](const NestedCell &cell) { return cell.polygons; });

  nDof = nNode * DIM_;
  ComputeFeatures();

  // output fine mesh number
  int numFineMesh = 0;
  for (int macI = 0; macI < nEle; ++macI) {
    numFineMesh += mic_nT[macI];
  }
  // std::cout << "numFineMesh: " << numFineMesh << std::endl;

  linSysSolver = new cpt::CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd, 3>();

  linSysSolver->SetPattern(vNeighbor);
  linSysSolver->AnalyzePattern();

  spdlog::info("number of macro nodes: {}, number of macro dofs: {} in this mesh", nNode, nDof);
  spdlog::info("number of macro polygon elements: {}", nEle);

  ComputePhi();
  Preprocess();
  PrepareMicroIntegration();
}

CBNSimulator::~CBNSimulator() {
  delete linSysSolver;
  linSysSolver = nullptr;
}

void CBNSimulator::PreprocessBackgroundMesh() {
  // get data structure from matlab
  // if (!readMesh(p_meshFile, polygon, node, eDof, nid, dup, micV, micT, bnid, bnid_deDup,
  //               bnid_deDup_ID, bnNode, reorderVec, nEle, nNode, nFacePerPoly, eleDofNum, mic_nV,
  //               mic_nT)) {
  //   log::error("error in reading mesh");
  //   exit(-1);
  // }

  const double tol = 1e-6;
  nEle             = static_cast<int>(nested_background_->nested_cells_.size());
  nFacePerPoly.resize(nEle);
  eleDofNum.resize(nEle);
  mic_nV.resize(nEle);
  mic_nT.resize(nEle);

  for (int macI = 0; macI < nEle; ++macI) {
    const auto &cell   = nested_background_->nested_cells_.at(macI);
    nFacePerPoly[macI] = static_cast<int>(cell.polyhedron_edges.size());
    mic_nV[macI]       = static_cast<int>(cell.tetrahedrons.NumVertices());
    mic_nT[macI]       = static_cast<int>(cell.tetrahedrons.NumTetrahedrons());
  }

  auto SetDiff = [](const Eigen::VectorXi &vecA, const Eigen::VectorXi &vecB) {
    size_t sizeA = vecA.size();
    Eigen::VectorXi diff(sizeA);
    Eigen::VectorXi diffIdx(sizeA);
    int cnt = 0;
    std::set<int> setB(vecB.begin(), vecB.end());
    for (int i = 0; i < sizeA; ++i) {
      if (!setB.count(vecA(i))) {
        diff(cnt)    = vecA(i);
        diffIdx(cnt) = i;
        ++cnt;
      }
    }
    diff.conservativeResize(cnt);
    diffIdx.conservativeResize(cnt);
    return std::make_pair(diff, diffIdx);
  };

  bnid.resize(nEle);
  bnid_deDup.resize(nEle);
  bnid_deDup_ID.resize(nEle);
  bnNode.resize(nEle);
  reorderVec.resize(nEle);

  for (int macI = 0; macI < nEle; ++macI) {
    const auto &cell         = nested_background_->nested_cells_.at(macI);
    const auto &TV_macI      = cell.tetrahedrons.mat_coordinates;
    const auto &polygon_macI = cell.polyhedron_edges;

    int mic_nV_macI = mic_nV[macI];

    bnid[macI].resize(nFacePerPoly[macI]);
    bnid_deDup[macI].resize(nFacePerPoly[macI]);
    bnid_deDup_ID[macI].resize(nFacePerPoly[macI]);
    Eigen::VectorXi bnid_all;
    for (int fI = 0; fI < nFacePerPoly[macI]; ++fI) {
      const auto &segs       = polygon_macI[fI];
      Eigen::Vector3d point  = segs(0, {0, 1, 2});
      Eigen::Vector3d vec1   = segs(0, {3, 4, 5}) - segs(0, {0, 1, 2});
      Eigen::Vector3d vec2   = segs(1, {3, 4, 5}) - segs(1, {0, 1, 2});
      Eigen::Vector3d normal = vec1.cross(vec2);
      normal.normalize();

      Eigen::VectorXi idx(mic_nV_macI);
      int cnt = 0;
      for (int mic_vI = 0; mic_vI < mic_nV_macI; ++mic_vI) {
        Eigen::Vector3d vec = TV_macI.row(mic_vI).transpose() - point;
        if (abs(vec.dot(normal)) < tol) {
          idx(cnt++) = mic_vI;
        }
      }
      idx.conservativeResize(cnt);
      bnid[macI][fI] = idx;

      auto retPair                = SetDiff(idx, bnid_all);
      Eigen::VectorXi idx_deDup   = retPair.first;
      Eigen::VectorXi idx_deDupID = retPair.second;
      int curSize                 = static_cast<int>(bnid_all.size());
      int insSize                 = static_cast<int>(idx_deDup.size());
      bnid_all.conservativeResize(curSize + insSize);
      bnid_all.segment(curSize, insSize) = idx_deDup;
      bnid_deDup[macI][fI]    = Eigen::VectorXi::LinSpaced(insSize, curSize, curSize + insSize - 1);
      bnid_deDup_ID[macI][fI] = idx_deDupID;
    }
    bnNode[macI] = static_cast<int>(bnid_all.size());

    // reorderVec
    Eigen::VectorXi bDofs(bnNode[macI] * 3);
    for (int bI = 0; bI < bnNode[macI]; ++bI) {
      bDofs(bI * 3)     = bnid_all(bI) * 3;
      bDofs(bI * 3 + 1) = bnid_all(bI) * 3 + 1;
      bDofs(bI * 3 + 2) = bnid_all(bI) * 3 + 2;
    }
    Eigen::VectorXi allDofs = Eigen::VectorXi::LinSpaced(mic_nV_macI * 3, 0, mic_nV_macI * 3 - 1);
    auto retPair            = SetDiff(allDofs, bDofs);
    Eigen::VectorXi iDofs   = retPair.first;
    Eigen::VectorXi reorderDofs(mic_nV_macI * 3);
    reorderDofs.segment(0, bDofs.size())            = bDofs;
    reorderDofs.segment(bDofs.size(), iDofs.size()) = iDofs;
    reorderVec[macI].resize(reorderDofs.size());
    for (int i_ = 0; i_ < reorderDofs.size(); ++i_) {
      reorderVec[macI](reorderDofs(i_)) = i_;
    }
  }

  eDof.resize(nEle);
  nid.resize(nEle);
  dup.resize(nEle);

  auto isEqual = [](const Eigen::Vector3d &a, const Eigen::Vector3d &b, const double &tol) {
    return ((a - b).cwiseAbs().array() < tol).all();
  };

  Eigen::MatrixXd nodes0(0, 3);                   // all nodes in polygon mesh, not de-duplicate yet
  std::vector<Eigen::MatrixXd> nodes_macI(nEle);  // nodes in each polygon

  for (int macI = 0; macI < nEle; ++macI) {
    const auto &cell = nested_background_->nested_cells_.at(macI);
    nid[macI].resize(nFacePerPoly[macI]);
    dup[macI].resize(nFacePerPoly[macI]);
    nodes_macI[macI].resize(0, 3);

    for (int fI = 0; fI < nFacePerPoly[macI]; ++fI) {
      int nV;
      Eigen::MatrixXd vertex;  // vertex per face (not de-duplicate yet)
      {                        // calculate vertex
        int nSeg = static_cast<int>(cell.polyhedron_edges[fI].rows());
        Eigen::MatrixXd cornerVertex =
            cell.polyhedron_edges[fI].leftCols<3>();  // corner vertex per face
        nV = nSeg * (nSeg + 1) / 2;
        vertex.resize(nV, 3);
        int cnt = 0;
        for (int i_ = 0; i_ < nSeg; ++i_) {
          for (int j_ = i_; j_ < nSeg; ++j_) {
            vertex.row(cnt++) = 0.5 * (cornerVertex.row(i_) + cornerVertex.row(j_));
          }
        }
      }

      // detect the duplicated nodes and remove them
      Eigen::MatrixXd vertexUni;
      std::vector<Eigen::VectorXi> dup_fI;
      {
        vertexUni.resize(nV, 3);
        dup_fI.reserve(nV);
        Eigen::VectorXi vertexLable = Eigen::VectorXi::Constant(nV, -1);
        int cnt                     = 0;
        for (int vI = 0; vI < nV; ++vI) {
          if (vertexLable(vI) != -1) {
            continue;
          }
          vertexUni.row(cnt) = vertex.row(vI);
          Eigen::VectorXi dup_fI_(nV);
          int cnt_ = 0;
          for (int vJ = vI; vJ < nV; ++vJ) {
            if (isEqual(vertex.row(vJ), vertex.row(vI), tol)) {
              vertexLable(vJ) = cnt;
              dup_fI_(cnt_++) = vJ;
            }
          }
          dup_fI_.conservativeResize(cnt_);
          dup_fI.emplace_back(dup_fI_);
          ++cnt;
        }
        vertexUni.conservativeResize(cnt, 3);
        dup_fI.shrink_to_fit();
      }

      dup[macI][fI] = dup_fI;

      Eigen::VectorXi idxInVertexUni;
      Eigen::VectorXi nid_fI;
      {
        idxInVertexUni.resize(vertexUni.rows());
        nid_fI.resize(vertexUni.rows());
        int cnt     = 0;
        int curSize = static_cast<int>(nodes_macI[macI].rows());
        for (int i_ = 0; i_ < vertexUni.rows(); ++i_) {
          int flag = 0;
          int nid_i;
          for (int j_ = 0; j_ < curSize; ++j_) {
            if (isEqual(nodes_macI[macI].row(j_), vertexUni.row(i_), tol)) {
              flag  = 1;
              nid_i = j_;
              break;
            }
          }
          if (!flag) {
            idxInVertexUni(cnt) = i_;
            nid_i               = curSize + cnt;
            ++cnt;
          }
          nid_fI(i_) = nid_i;
        }
        idxInVertexUni.conservativeResize(cnt);
      }
      nodes_macI[macI].conservativeResize(nodes_macI[macI].rows() + idxInVertexUni.size(), 3);
      nodes_macI[macI].bottomRows(idxInVertexUni.size()) = vertexUni(idxInVertexUni, Eigen::all);
      nid[macI][fI]                                      = nid_fI;
    }

    // append nodes_macI to nodes0
    nodes0.conservativeResize(nodes0.rows() + nodes_macI[macI].rows(), 3);
    nodes0.bottomRows(nodes_macI[macI].rows()) = nodes_macI[macI];
  }

  // compute node by de-duplicate nodes0
  {
    int nNode0                  = static_cast<int>(nodes0.rows());
    Eigen::VectorXi nodes0Lable = Eigen::VectorXi::Constant(nNode0, -1);
    Eigen::VectorXi nodeIdxInNode0(nNode0);
    int cnt = 0;
    for (int nI0 = 0; nI0 < nNode0; ++nI0) {
      if (nodes0Lable(nI0) != -1) {
        continue;
      }
      for (int nJ0 = nI0; nJ0 < nNode0; ++nJ0) {
        if (isEqual(nodes0.row(nJ0), nodes0.row(nI0), tol)) {
          nodes0Lable(nJ0) = cnt;
        }
      }

      nodeIdxInNode0(cnt) = nI0;
      cnt++;
    }
    nodeIdxInNode0.conservativeResize(cnt);
    node  = nodes0(nodeIdxInNode0, Eigen::all);
    nNode = static_cast<int>(node.rows());

    // eDof
    int k = 0;
    for (int macI = 0; macI < nEle; ++macI) {
      int nNode_macI = nodes_macI[macI].rows();
      eDof[macI].resize(nNode_macI * 3);
      eleDofNum[macI] = nNode_macI * 3;
      for (int i_ = 0; i_ < nNode_macI; ++i_) {
        int nI                 = nodes0Lable(k + i_);
        eDof[macI](i_ * 3)     = nI * 3;
        eDof[macI](i_ * 3 + 1) = nI * 3 + 1;
        eDof[macI](i_ * 3 + 2) = nI * 3 + 2;
      }
      k += nNode_macI;
    }
  }
}

void CBNSimulator::ComputeFeatures() {
  // setup eNode
  eNode.resize(nEle);
  oneapi::tbb::parallel_for(0, nEle, 1, [&](int eleI) {
    eNode[eleI] = eDof[eleI](Eigen::seq(0, Eigen::last, 3));
    eNode[eleI] /= 3;
    Assert(eNode[eleI].size() == eleDofNum[eleI] / 3);
  });

  // compute vNeighbor
  vNeighbor.resize(nNode);
  for (int eleI = 0; eleI < nEle; ++eleI) {
    int n = eNode[eleI].size();
    oneapi::tbb::parallel_for(0, n, 1,
                              [&](int i)

                              {
                                int vI = eNode[eleI](i);
                                vNeighbor[vI].insert(eNode[eleI].begin(), eNode[eleI].end());
                              });
  }

  // remove self in vNeighbor
  oneapi::tbb::parallel_for(0, nNode, 1, [&](int vI) { vNeighbor[vI].erase(vI); });
}

void CBNSimulator::ComputePhiOnFace(const Eigen::MatrixXd &macV_f, const Eigen::MatrixXd &bV_f,
                                    Eigen::MatrixXd &Phi_f) {
  int nSeg = macV_f.rows();
  // if(nSeg > 7 && nSeg != 10){
  //     spdlog::error("polygon sides undefined");
  //     exit(-1);
  // }

  // compute generalized barycentric basis w
  const double eps = 1.0e-4;
  int bnV          = bV_f.rows();
  Eigen::MatrixXd w(bnV, nSeg);
  for (int bI = 0; bI < bnV; ++bI) {
    Eigen::RowVector3d bVI = bV_f.row(bI);
    int k                  = -1;
    for (int i_ = 0; i_ < nSeg; ++i_) {
      if (abs(macV_f(i_, 0) - bVI(0)) < eps && abs(macV_f(i_, 1) - bVI(1)) < eps &&
          abs(macV_f(i_, 2) - bVI(2)) < eps) {
        k = i_;
        break;
      }
    }

    if (k == -1) {
      ComputeBarycentric(macV_f, bVI, bI, w);
    } else {
      w.row(bI).setZero();
      w(bI, k) = 1.0;
    }
  }

  // S-Patches
  int nV;  // number of macro nodes in this face
  nV = nSeg * (nSeg + 1) / 2;
  // switch (nSeg) {
  //     case 3:
  //         nV = 6;
  //         break;
  //     case 4:
  //         nV = 10;
  //         break;
  //     case 5:
  //         nV = 15;
  //         break;
  //     case 6:
  //         nV = 21;
  //         break;
  //     case 7:
  //         nV = 28;
  //         break;
  //     case 10:
  //         nV = 55;
  //         break;
  //     default: {
  //         spdlog::error("polygon sides undefined");
  //         exit(-1);
  //     }
  // }

  Phi_f.resize(bnV, nV);
  int cnt = 0;
  for (int i = 0; i < nSeg; ++i) {
    Phi_f.col(cnt++) = w.col(i).array().square();
    for (int j = i + 1; j < nSeg; ++j) {
      Phi_f.col(cnt++) = 2.0 * w.col(i).array() * w.col(j).array();
    }
  }
  Assert(cnt == nV);
}

void CBNSimulator::ComputePhi() {
  spdlog::info("compute Phi");
  Phi.resize(nEle);

  oneapi::tbb::parallel_for(0, nEle, 1, [&](int macI) {
    const auto &cell = nested_background_->nested_cells_.at(macI);
    Phi[macI].setZero(bnNode[macI] * DIM_, eleDofNum[macI]);

    for (int fI = 0; fI < nFacePerPoly[macI]; ++fI) {
      Eigen::MatrixXd macV_f = cell.polyhedron_edges[fI](Eigen::all, Eigen::seq(0, 2));
      Eigen::MatrixXd bV_f   = cell.tetrahedrons.mat_coordinates(bnid[macI][fI], Eigen::all);
      Eigen::MatrixXd Phi_f;
      ComputePhiOnFace(macV_f, bV_f, Phi_f);

      const Eigen::VectorXi &nid_f           = nid[macI][fI];
      const Eigen::VectorXi &bnid_deDup_f    = bnid_deDup[macI][fI];
      const Eigen::VectorXi &bnid_deDup_ID_f = bnid_deDup_ID[macI][fI];

      int sz1 = (int)nid_f.size();
      int sz2 = (int)bnid_deDup_ID_f.size();
      if (!sz2) {
        // std::cout << "skip empty bnid_deDup_f" << std::endl;
        continue;
      }

      Eigen::MatrixXd Phi_deDup_f = Eigen::MatrixXd::Zero(sz2, sz1);
      for (int j = 0; j < sz1; ++j) {
        const Eigen::VectorXi &dup_j = dup[macI][fI][j];
        Assert(dup_j.size() > 0);
        for (const auto &j_ : dup_j) {
          Assert(j_ >= 0 && j_ < Phi_f.cols());
          Phi_deDup_f.col(j) += Phi_f(bnid_deDup_ID_f, j_);
        }
      }

      Eigen::MatrixXd Phi_deDup_f_ = Eigen::MatrixXd::Zero(sz2 * 3, sz1 * 3);
      Phi_deDup_f_(Eigen::seq(0, Eigen::last, 3), Eigen::seq(0, Eigen::last, 3)) = Phi_deDup_f;
      Phi_deDup_f_(Eigen::seq(1, Eigen::last, 3), Eigen::seq(1, Eigen::last, 3)) = Phi_deDup_f;
      Phi_deDup_f_(Eigen::seq(2, Eigen::last, 3), Eigen::seq(2, Eigen::last, 3)) = Phi_deDup_f;

      Eigen::VectorXi index1(sz1 * 3);
      Eigen::VectorXi index2(sz2 * 3);
      index1(Eigen::seq(0, Eigen::last, 3)) = nid_f * 3;
      index1(Eigen::seq(1, Eigen::last, 3)) = (nid_f * 3).array() + 1;
      index1(Eigen::seq(2, Eigen::last, 3)) = (nid_f * 3).array() + 2;
      index2(Eigen::seq(0, Eigen::last, 3)) = bnid_deDup_f * 3;
      index2(Eigen::seq(1, Eigen::last, 3)) = (bnid_deDup_f * 3).array() + 1;
      index2(Eigen::seq(2, Eigen::last, 3)) = (bnid_deDup_f * 3).array() + 2;

      // assert
      for (const auto &item : index1) {
        Assert(item >= 0 && item < eleDofNum[macI]);
      }
      for (const auto &item : index2) {
        Assert(item >= 0 && item < bnNode[macI] * DIM_);
      }

      Phi[macI](index2, index1) = Phi_deDup_f_;
    }
  });
}

void CBNSimulator::Preprocess() {
  spdlog::info("preprocess NBC in FCM manner!");

  const auto &nNBC = physicalDomain->nNBC;
  const auto &NBC  = physicalDomain->NBC;
  NBC_micI.resize(nNBC);
  NBC_micN.resize(nNBC);
  NBC_val.resize(nNBC);
  elementNBC.resize(nEle);
  int cnt = 0;

  for (const auto &NBCI : NBC) {
    const auto &P                    = NBCI.first;
    const Eigen::Vector3d &loadValue = NBCI.second;

    // get macI, micI
    Eigen::Vector3d P_      = P.transpose();
    std::pair<int, int> pii = nested_background_->GetPointLocation(P_);
    int macI                = pii.first;
    int micI                = pii.second;
    Assert(macI >= 0 && macI < nEle);
    Assert(micI >= 0 && micI < mic_nT[macI]);
    if (macI < 0 || macI >= nEle || micI < 0 || micI >= mic_nT[macI]) {
      spdlog::error("error on point query, macI = {}, micI = {}", macI, micI);
      exit(0);
    }
    const auto &cell = nested_background_->nested_cells_.at(macI);
    Eigen::Matrix<double, 4, 3> X =
        cell.tetrahedrons.mat_coordinates(cell.tetrahedrons.mat_tetrahedrons.row(micI), Eigen::all);
    elementNBC[macI].emplace_back(cnt);
    NBC_micI[cnt] = micI;
    ComputeNForTet(P, X, NBC_micN[cnt]);
    NBC_val[cnt] = loadValue;

    ++cnt;
  }

  if (physicalDomain->use_Nitsche) {
    spdlog::info("preprocess DBC in Nitsche method!");

    // preprocess DBC
    const auto &DBC = physicalDomain->DBC;
    int DBCNum      = static_cast<int>(DBC.size());
    int gpNum       = DBCNum * integrationOrder_BC * integrationOrder_BC;
    DBC_macI.resize(gpNum);
    DBC_micI.resize(gpNum);
    DBC_micN.resize(gpNum);
    DBC_micB.resize(gpNum);
    DBC_DT_mul_normal.resize(gpNum);
    DBC_w.resize(gpNum);
    DBC_val.resize(gpNum);

    // tbb::parallel_for(0, DBCNum, 1, [&](int dbc_idx) {
    for (int dbc_idx = 0; dbc_idx < DBCNum; ++dbc_idx) {
      int gpcnt                        = dbc_idx * integrationOrder_BC * integrationOrder_BC;
      auto DBCI                        = DBC[dbc_idx];
      const Triangle &triangle         = physicalDomain->Tri[DBCI.first];
      const Eigen::Vector3d &dispValue = DBCI.second;

      for (int i = 0; i < integrationOrder_BC; ++i) {
        for (int j = 0; j < integrationOrder_BC; ++j) {
          Eigen::RowVector3d localInTri(GP_BC(i), GP_BC(j), 0.0);
          DBC_w[gpcnt] = GW_BC(i) * GW_BC(j) * triangle.GetDetJac(localInTri);

          Eigen::RowVector3d normal = triangle.GetNormal(localInTri);
          DBC_DT_mul_normal[gpcnt].setZero();
          DBC_DT_mul_normal[gpcnt](0, 0) = normal(0);
          DBC_DT_mul_normal[gpcnt](4, 0) = normal(2);
          DBC_DT_mul_normal[gpcnt](5, 0) = normal(1);
          DBC_DT_mul_normal[gpcnt](1, 1) = normal(1);
          DBC_DT_mul_normal[gpcnt](3, 1) = normal(2);
          DBC_DT_mul_normal[gpcnt](5, 1) = normal(0);
          DBC_DT_mul_normal[gpcnt](2, 2) = normal(2);
          DBC_DT_mul_normal[gpcnt](3, 2) = normal(1);
          DBC_DT_mul_normal[gpcnt](4, 2) = normal(0);
          DBC_DT_mul_normal[gpcnt]       = YM1 * D_.transpose() * DBC_DT_mul_normal[gpcnt];

          Eigen::RowVector3d global;
          triangle.MapLocalToGlobal(localInTri, global);

          // get macI, micI
          Eigen::Vector3d global_ = global.transpose();
          // std::pair<int, int> pii = nested_background_->GetPointLocation(global_);
          std::pair<int, int> pii =
              nested_background_->GetPointLocationAlternative(global_, 1e-6, 1e-6);
          int macI = pii.first;
          int micI = pii.second;
          Assert(macI >= 0 && macI < nEle);
          Assert(micI >= 0 && micI < mic_nT[macI]);
          if (macI < 0 || macI >= nEle || micI < 0 || micI >= mic_nT[macI]) {
            spdlog::error("error on point query, macI = {}, micI = {}", macI, micI);
            exit(0);
          }

          const auto &cell              = nested_background_->nested_cells_.at(macI);
          Eigen::Matrix<double, 4, 3> X = cell.tetrahedrons.mat_coordinates(
              cell.tetrahedrons.mat_tetrahedrons.row(micI), Eigen::all);
          DBC_macI[gpcnt] = macI;
          DBC_micI[gpcnt] = micI;
          ComputeNForTet(global, X, DBC_micN[gpcnt]);
          ComputeBForTet(X, DBC_micB[gpcnt]);

          DBC_val[gpcnt] = dispValue;

          ++gpcnt;
        }
      }
    }

    elementDBC.resize(0);
    elementDBC.resize(nEle);
    for (int gpI = 0; gpI < gpNum; ++gpI) {
      elementDBC[DBC_macI[gpI]].emplace_back(gpI);
    }

  } else {
    spdlog::info("preprocess DBC in direct manner!");

    DBCV.resize(nNode, 3);
    int cntD            = 0;
    const auto &DBCBBox = physicalDomain->DBCBBox;
    for (int vI = 0; vI < nNode; ++vI) {
      for (int _i = 0; _i < DBCBBox.size(); ++_i) {
        if (node(vI, 0) > DBCBBox[_i](0, 0) && node(vI, 0) < DBCBBox[_i](1, 0) &&
            node(vI, 1) > DBCBBox[_i](0, 1) && node(vI, 1) < DBCBBox[_i](1, 1) &&
            node(vI, 2) > DBCBBox[_i](0, 2) && node(vI, 2) < DBCBBox[_i](1, 2)) {
          fixeddofs.emplace_back(vI * 3);
          fixeddofs.emplace_back(vI * 3 + 1);
          fixeddofs.emplace_back(vI * 3 + 2);
          DBCV.row(cntD++) = node.row(vI);

          break;
        }
      }
    }
    std::cout << "DBC points: " << cntD << std::endl;
    DBCV.conservativeResize(cntD, 3);
  }
}

void CBNSimulator::PrepareMicroIntegration() {
  spdlog::info("prepare for micro integration");
  mic_Ke.resize(nEle);
  mic_Vol.resize(nEle);
  Vol0 = 0.0;

  for (int macI = 0; macI < nEle; ++macI) {
    const auto &cell_tet_mesh = nested_background_->nested_cells_.at(macI).tetrahedrons;

    mic_Ke[macI].resize(mic_nT[macI]);
    mic_Vol[macI].resize(mic_nT[macI]);
    const Eigen::MatrixXd &micV_ = cell_tet_mesh.mat_coordinates;
    const Eigen::MatrixXi &micT_ = cell_tet_mesh.mat_tetrahedrons;

    // compute mic_Ke, mic_Vol
    oneapi::tbb::parallel_for(0, mic_nT[macI], 1, [&](int tI) {
      ComputeKeForTet(micV_(micT_.row(tI), Eigen::all), D_, mic_Ke[macI][tI], mic_Vol[macI][tI]);
    });
    Vol0 += mic_Vol[macI].sum();
  }

  mic_eDof.resize(nEle);
  nBDof.resize(nEle);
  nIDof.resize(nEle);
  for (int macI = 0; macI < nEle; ++macI) {
    const auto &cell_tet_mesh = nested_background_->nested_cells_.at(macI).tetrahedrons;

    const Eigen::MatrixXi &micT_ = cell_tet_mesh.mat_tetrahedrons;

    // compute mic_eDof
    mic_eDof[macI].resize(mic_nT[macI] * 12);
    oneapi::tbb::parallel_for(0, mic_nT[macI], 1, [&](int tI) {
      int start                                            = 12 * tI;
      Eigen::Array4i tmp                                   = micT_.row(tI) * 3;
      mic_eDof[macI](Eigen::seq(start, start + 11, 3))     = reorderVec[macI](tmp);
      mic_eDof[macI](Eigen::seq(start + 1, start + 11, 3)) = reorderVec[macI](tmp + 1);
      mic_eDof[macI](Eigen::seq(start + 2, start + 11, 3)) = reorderVec[macI](tmp + 2);
    });

    // compute nBDof, nIDof
    nBDof[macI] = bnNode[macI] * 3;
    nIDof[macI] = mic_nV[macI] * 3 - nBDof[macI];
  }
}

void CBNSimulator::ComputeSystem() {
  elementKe.resize(nEle);
  elementLoad.resize(nEle);
  elementM2.resize(nEle);
  elementM.resize(nEle);

  // resize first is needed, otherwise segment fault will occur in GPU (don't find the reason T_T)
  for (int macI = 0; macI < nEle; ++macI) {
    elementKe[macI].resize(eleDofNum[macI], eleDofNum[macI]);
    elementLoad[macI].resize(eleDofNum[macI]);
    elementM2[macI].resize(12 * mic_nT[macI], eleDofNum[macI]);
    elementM[macI].resize(3 * mic_nV[macI], eleDofNum[macI]);
  }

  spdlog::info("compute global system on CPU");

  // time recorder
  double time_microAssemble = 0.0;
  double time_KibPhi        = 0.0;
  double time_microSolver   = 0.0;
  double time_M1            = 0.0;
  double time_M2            = 0.0;
  double time_BDB           = 0.0;
  double time_NBC           = 0.0;
  double time_DBC           = 0.0;

  // iter on macro elements
  tbb::parallel_for(0, nEle, 1, [&](int macI) {
    const int &nEleMi = mic_nT[macI];  // number of micro tets in this macro polygon element
    std::vector<Eigen::Matrix<double, 12, 12>> rho_Ke(nEleMi);

    auto microAssemble_Start = std::chrono::steady_clock::now();

    std::vector<Eigen::Triplet<double>> micK_triLists;
    micK_triLists.reserve(12 * 12 * nEleMi);
    // iter on micro elements
    for (int micI = 0; micI < nEleMi; ++micI) {
      double YM_rho = YM0 + pow(rhos_[macI](micI), penaltyYM) * (YM1 - YM0);
      rho_Ke[micI]  = YM_rho * mic_Ke[macI][micI];
      int start     = micI * 12;
      for (int i_ = 0; i_ < 12; ++i_) {
        for (int j_ = 0; j_ < 12; ++j_) {
          micK_triLists.emplace_back(Eigen::Triplet<double>(
              mic_eDof[macI](start + i_), mic_eDof[macI](start + j_), rho_Ke[micI](i_, j_)));
        }
      }
    }

    Eigen::SparseMatrix<double> micK(mic_nV[macI] * 3, mic_nV[macI] * 3);
    micK.setFromTriplets(micK_triLists.begin(), micK_triLists.end());

    Eigen::MatrixXd Kib = micK.block(nBDof[macI], 0, nIDof[macI], nBDof[macI]);
    micK                = micK.block(nBDof[macI], nBDof[macI], nIDof[macI], nIDof[macI]);

    cpt::LinearSolver<Eigen::VectorXi, Eigen::VectorXd, 3> *microSolver;
    // #ifdef LINSYSSOLVER_USE_CHOLMOD
    //         microSolver = new cpt::CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd, 3>();
    // #else
    microSolver = new cpt::EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd, 3>();
    // #endif
    microSolver->SetPattern(micK);
    microSolver->AnalyzePattern();
    microSolver->Factorize();

    std::chrono::duration<double> microAssemble_(std::chrono::steady_clock::now() -
                                                 microAssemble_Start);
    time_microAssemble += microAssemble_.count();

    auto KibPhi_start   = std::chrono::steady_clock::now();
    Eigen::MatrixXd rhs = -Kib * Phi[macI];
    std::chrono::duration<double> KibPhi_(std::chrono::steady_clock::now() - KibPhi_start);
    time_KibPhi += KibPhi_.count();

    auto microSolver_start = std::chrono::steady_clock::now();

    Eigen::MatrixXd M(nIDof[macI], eleDofNum[macI]);
    for (int r = 0; r < eleDofNum[macI]; ++r) {
      Eigen::VectorXd rhs_r = rhs.col(r);
      Eigen::VectorXd tmp(nIDof[macI]);
      microSolver->Solve(rhs_r, tmp);
      M.col(r) = tmp;
    }
    delete microSolver;

    std::chrono::duration<double> microSolver_(std::chrono::steady_clock::now() -
                                               microSolver_start);
    time_microSolver += microSolver_.count();

    auto M1_start = std::chrono::steady_clock::now();
    auto &M1      = elementM[macI];
    M1.resize(mic_nV[macI] * 3, eleDofNum[macI]);
    M1(Eigen::seq(0, nBDof[macI] - 1), Eigen::all)       = Phi[macI];
    M1(Eigen::seq(nBDof[macI], Eigen::last), Eigen::all) = M;
    std::chrono::duration<double> M1_(std::chrono::steady_clock::now() - M1_start);
    time_M1 += M1_.count();

    auto M2_start       = std::chrono::steady_clock::now();
    Eigen::MatrixXd &M2 = elementM2[macI];
    M2                  = M1(mic_eDof[macI], Eigen::all);
    Assert(M2.rows() == 12 * nEleMi && M2.cols() == eleDofNum[macI]);
    std::chrono::duration<double> M2_(std::chrono::steady_clock::now() - M2_start);
    time_M2 += M2_.count();

    // compute macro element stiffness matrix
    auto BDB_start = std::chrono::steady_clock::now();
    elementKe[macI].setZero(eleDofNum[macI], eleDofNum[macI]);
    for (int micI = 0; micI < nEleMi; ++micI) {
      const Eigen::MatrixXd &mic_M2 = M2(Eigen::seq(micI * 12, (micI + 1) * 12 - 1), Eigen::all);
      elementKe[macI].noalias() += mic_M2.transpose() * rho_Ke[micI] * mic_M2;
    }
    // debug
    // Utils::writeMatrixXd("../output/ke_c.txt", elementKe[macI]);
    std::chrono::duration<double> BDB_(std::chrono::steady_clock::now() - BDB_start);
    time_BDB += BDB_.count();

    // apply Neumann boundary condition: compute macro element load vector
    auto NBC_start = std::chrono::steady_clock::now();

    elementLoad[macI].setZero(eleDofNum[macI]);
    for (const auto &NBC_i : elementNBC[macI]) {
      int micI = NBC_micI[NBC_i];
      // double rho_NBC = rho[macI](micI);
      double rho_NBC = 1.0;
      Eigen::MatrixXd macN =
          NBC_micN[NBC_i] * M2(Eigen::seq(micI * 12, (micI + 1) * 12 - 1), Eigen::all);
      elementLoad[macI].noalias() += rho_NBC * macN.transpose() * NBC_val[NBC_i];
    }

    std::chrono::duration<double> NBC_(std::chrono::steady_clock::now() - NBC_start);
    time_NBC += NBC_.count();

    if (physicalDomain->use_Nitsche) {
      for (const auto &gpI : elementDBC[macI]) {
        int micI = DBC_micI[gpI];
        Eigen::MatrixXd macN =
            DBC_micN[gpI] * M2(Eigen::seq(micI * 12, (micI + 1) * 12 - 1), Eigen::all);
        Eigen::MatrixXd macB =
            DBC_micB[gpI] * M2(Eigen::seq(micI * 12, (micI + 1) * 12 - 1), Eigen::all);

        // add penalty term to elementKe
        elementKe[macI].noalias() +=
            DBC_w[gpI] * physicalDomain->penaltyNitsche * macN.transpose() * macN;
        // add penalty term to elementLoad
        elementLoad[macI].noalias() +=
            DBC_w[gpI] * physicalDomain->penaltyNitsche * macN.transpose() * DBC_val[gpI];

        // add -(G+G') to elementKe
        Eigen::MatrixXd traction = macB.transpose() * DBC_DT_mul_normal[gpI];  // B' * D' * normal
        Eigen::MatrixXd G        = DBC_w[gpI] * traction * macN;
        elementKe[macI].noalias() -= (G + G.transpose());
        // add -g to elementLoad
        elementLoad[macI].noalias() -= DBC_w[gpI] * traction * DBC_val[gpI];
      }
    }
  });

  linSysSolver->SetZero();
  for (int macI = 0; macI < nEle; ++macI) {
    for (int i_ = 0; i_ < eleDofNum[macI]; ++i_) {
      for (int j_ = 0; j_ < eleDofNum[macI]; ++j_) {
        linSysSolver->AddCoeff(eDof[macI](i_), eDof[macI](j_), elementKe[macI](i_, j_));
      }
    }
  }
  // add eps to diagonal of K
  for (int i = 0; i < nDof; ++i) {
    linSysSolver->AddCoeff(i, i, 1.0e-5);
  }

  // assemble macro load vector
  if (!NBC_flag) {
    load.setZero(nDof);
    for (int macI = 0; macI < nEle; ++macI) {
      load(eDof[macI]) += elementLoad[macI];
    }
    NBC_flag = true;
  }

  if (!physicalDomain->use_Nitsche) {
    // auto DBC_start = std::chrono::steady_clock::now();
    for (const auto &dofI : fixeddofs) {
      linSysSolver->SetZeroCol(dofI);
      linSysSolver->SetUnitRow(dofI);
    }
    // std::chrono::duration<double> DBC_(std::chrono::steady_clock::now() - DBC_start);
    // std::cout << "direct DBC cost: " << DBC_.count() << std::endl;
  }
}

void CBNSimulator::Solve() {
  spdlog::info("solve the linear system");
  linSysSolver->Factorize();
  linSysSolver->Solve(load, U);

  // debug
  // Utils::writeMatrixXd(workspace + "U.txt", U);
  // Utils::writeMatrixXd(workspace + "load.txt", load);

  // scatter global solution to element solution
  elementU.resize(nEle);
  oneapi::tbb::parallel_for(0, nEle, 1, [&](int eleI) {
    elementU[eleI] = U(eDof[eleI]);
    Assert(elementU[eleI].rows() == eleDofNum[eleI]);
  });
}

auto CBNSimulator::Simulate(const std::vector<Eigen::VectorXd> &rhos) -> Eigen::VectorXd {
  // check size
  rhos_ = rhos;
  Assert(rhos_.size() == nEle);
  for (int macI = 0; macI < nEle; ++macI) {
    Assert(rhos_[macI].size() == mic_nT[macI]);
  }
  ComputeSystem();
  Solve();
  PostprocessFineMesh();

  if (handlePhysicalDomain) {
    UpdateVInPD();
  }

  return U;
}

void CBNSimulator::PostprocessFineMesh() {
  spdlog::info("Postprocess fine mesh");

  // resize
  fine_tet_displacement.resize(nEle);
  fine_tet_stress.resize(nEle);
  for (int macI = 0; macI < nEle; ++macI) {
    fine_tet_displacement[macI].resize(mic_nT[macI]);
    fine_tet_stress[macI].resize(mic_nT[macI]);
  }

  tbb::parallel_for(0, nEle, 1, [&](int macI) {
    // for (int macI = 0; macI < nEle; ++macI) {
    Eigen::VectorXd mic_displacement = elementM[macI] * elementU[macI];
    const auto &cell                 = nested_background_->nested_cells_.at(macI);
    for (int micI = 0; micI < mic_nT[macI]; ++micI) {
      fine_tet_displacement[macI][micI] =
          mic_displacement(mic_eDof[macI](Eigen::seqN(micI * 12, 12)));

      Eigen::Matrix<double, 4, 3> X = cell.tetrahedrons.mat_coordinates(
          cell.tetrahedrons.mat_tetrahedrons.row(micI), Eigen::all);
      Eigen::Matrix<double, 6, 12> micB;
      ComputeBForTet(X, micB);

      double YM_rho                       = YM0 + pow(rhos_[macI](micI), penaltyYM) * (YM1 - YM0);
      Eigen::Vector<double, 6> stress_vec = YM_rho * D_ * micB * fine_tet_displacement[macI][micI];
      fine_tet_stress[macI][micI](0, 0)   = stress_vec(0);
      fine_tet_stress[macI][micI](1, 1)   = stress_vec(1);
      fine_tet_stress[macI][micI](2, 2)   = stress_vec(2);
      fine_tet_stress[macI][micI](0, 1)   = stress_vec(3);
      fine_tet_stress[macI][micI](1, 0)   = stress_vec(3);
      fine_tet_stress[macI][micI](1, 2)   = stress_vec(4);
      fine_tet_stress[macI][micI](2, 1)   = stress_vec(4);
      fine_tet_stress[macI][micI](0, 2)   = stress_vec(5);
      fine_tet_stress[macI][micI](2, 0)   = stress_vec(5);
    }
  });
}

void CBNSimulator::QueryResults(const Eigen::MatrixXd &query_points,
                                std::vector<Eigen::Vector3d> &query_displacement,
                                std::vector<Eigen::Matrix3d> &query_stress) {
  // locate query points
  int query_num = static_cast<int>(query_points.rows());
  Eigen::VectorXi query_flag(query_num);
  Eigen::VectorXi query_mac_index(query_num);
  Eigen::VectorXi query_mic_index(query_num);
  // TODO: can't parallelize, due to 'GetPointLocationAlternative'
  for (int qI = 0; qI < query_num; ++qI) {
    Eigen::Vector3d query_pointI = query_points.row(qI);
    std::pair<int, int> pii =
        nested_background_->GetPointLocationAlternative(query_pointI, 1e-6, 1e-6);
    int macI = pii.first;
    int micI = pii.second;
    if (macI < 0 || macI >= nEle || micI < 0 || micI >= mic_nT[macI]) {
      spdlog::warn("query point outside mesh, macI = {}, micI = {}", macI, micI);
      query_flag[qI] = 0;
    } else {
      query_flag[qI]      = 1;
      query_mac_index[qI] = macI;
      query_mic_index[qI] = micI;
    }
  }
  // sha::WriteVectorToFile(WorkingResultDirectoryPath() / "query_flag.txt", query_flag);
  // sha::WriteVectorToFile(WorkingResultDirectoryPath() / "query_macI.txt", query_mac_index);
  // sha::WriteVectorToFile(WorkingResultDirectoryPath() / "query_micI.txt", query_mic_index);

  QueryResultsWithLocation(query_points, query_flag, query_mac_index, query_mic_index,
                           query_displacement, query_stress);
}

void CBNSimulator::QueryResultsWithLocation(const Eigen::MatrixXd &query_points,
                                            const Eigen::VectorXi &query_flag,
                                            const Eigen::VectorXi &query_mac_index,
                                            const Eigen::VectorXi &query_mic_index,
                                            std::vector<Eigen::Vector3d> &query_displacement,
                                            std::vector<Eigen::Matrix3d> &query_stress) {
  int query_num = static_cast<int>(query_points.rows());
  query_displacement.resize(query_num);
  query_stress.resize(query_num);
  tbb::parallel_for(0, query_num, 1, [&](int qI) {
    // for (int qI = 0; qI < query_num; ++qI) {
    if (query_flag[qI]) {  // inside mesh
      Eigen::Vector3d query_pointI = query_points.row(qI);
      int macI                     = query_mac_index[qI];
      int micI                     = query_mic_index[qI];
      const auto &cell_tet_mesh    = nested_background_->nested_cells_.at(macI).tetrahedrons;

      Eigen::Matrix<double, 4, 3> X =
          cell_tet_mesh.mat_coordinates(cell_tet_mesh.mat_tetrahedrons.row(micI), Eigen::all);
      Eigen::Matrix<double, 3, 12> micN;
      ComputeNForTet(query_pointI, X, micN);

      query_displacement[qI] = micN * fine_tet_displacement[macI][micI];
      query_stress[qI]       = fine_tet_stress[macI][micI];
    } else {  // outside mesh
      query_displacement[qI].setZero();
      query_stress[qI].setIdentity();
    }
  });
}

void CBNSimulator::UpdateVInPD() {
  spdlog::info("update V in triangle mesh (physical domain)");

  int numV = physicalDomain->numV;
  std::vector<Eigen::Vector3d> pd_displacement;
  std::vector<Eigen::Matrix3d> pd_stress;
  QueryResults(physicalDomain->mesh_.mat_coordinates, pd_displacement, pd_stress);

  oneapi::tbb::parallel_for(0, numV, 1, [&](int vI) {
    physicalDomain->V1.row(vI) =
        physicalDomain->mesh_.mat_coordinates.row(vI) + pd_displacement[vI].transpose();
  });
}

}  // namespace da::sha