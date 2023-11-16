#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>
#include <mshio/mshio.h>

#include "io_utils.h"

#include "sha-base-framework/frame.h"

namespace da::sha {

void WriteTriVTK(const std::string &path, const Eigen::MatrixXd &V, const Eigen::MatrixXi &T,
                        const std::vector<double> &cell_data,  const std::vector<double> &v_data){
  std::ofstream out(path);
  out << "# vtk DataFile Version 3.0\n"
         "Volume Mesh\n"
         "ASCII\n"
         "DATASET UNSTRUCTURED_GRID" << std::endl;
  out << "POINTS " << V.rows() << " double" << std::endl;
  for (int i = 0; i < V.rows(); ++i) {
    out << std::setprecision(18) << V.row(i).x() << " " << V.row(i).y() << " " << V.row(i).z() << std::endl;
  }
  out << "CELLS " << T.rows() << " " << T.rows() * (3 + 1) << std::endl;
  for (int i = 0; i < T.rows(); ++i) {
    out << "3 " << T.row(i).x() << " " << T.row(i).y() << " " << T.row(i).z() << std::endl;
  }
  out << "CELL_TYPES " << T.rows() << std::endl;
  for (int i = 0; i < T.rows(); ++i) {
    out << 5 << std::endl;
  }

  if(!cell_data.empty()){
    out << "CELL_DATA " << cell_data.size() << "\n" <<
        "SCALARS cell_scalars double 1\n" <<
        "LOOKUP_TABLE default" << std::endl;
    for (auto &d: cell_data) {
      out << d << std::endl;
    }
  }

  if(!v_data.empty()){
    out << "POINT_DATA " << v_data.size() << "\n" <<
        "SCALARS point_scalars double 1\n" <<
        "LOOKUP_TABLE default" << std::endl;
    for (auto &d: v_data) {
      out << d << std::endl;
    }
  }
}

void WriteTetVTK(const std::string &path, const Eigen::MatrixXd &V, const Eigen::MatrixXi &T,
                        const std::vector<double> &cell_data,  const std::vector<double> &v_data){
  std::ofstream out(path);
  out << "# vtk DataFile Version 3.0\n"
         "Volume Mesh\n"
         "ASCII\n"
         "DATASET UNSTRUCTURED_GRID" << std::endl;
  out << "POINTS " << V.rows() << " double" << std::endl;
  for (int i = 0; i < V.rows(); ++i) {
    out << std::setprecision(18) << V.row(i).x() << " " << V.row(i).y() << " " << V.row(i).z() << std::endl;
  }
  out << "CELLS " << T.rows() << " " << T.rows() * (4 + 1) << std::endl;
  for (int i = 0; i < T.rows(); ++i) {
    out << "4 " << T.row(i).x() << " " << T.row(i).y() << " " << T.row(i).z() << " " << T.row(i).w() << std::endl;
  }
  out << "CELL_TYPES " << T.rows() << std::endl;
  for (int i = 0; i < T.rows(); ++i) {
    out << 10 << std::endl;
  }

  if(!cell_data.empty()){
    out << "CELL_DATA " << cell_data.size() << "\n" <<
        "SCALARS cell_scalars double 1\n" <<
        "LOOKUP_TABLE default" << std::endl;
    for (auto &d: cell_data) {
      out << d << std::endl;
    }
  }

  if(!v_data.empty()){
    out << "POINT_DATA " << v_data.size() << "\n" <<
        "SCALARS point_scalars double 1\n" <<
        "LOOKUP_TABLE default" << std::endl;
    for (auto &d: v_data) {
      out << d << std::endl;
    }
  }
}

void WritePntVTK(const std::string &path, const Eigen::MatrixXd &V) {
  std::ofstream out(path);
  out << "# vtk DataFile Version 3.0\n"
         "Volume Mesh\n"
         "ASCII\n"
         "DATASET UNSTRUCTURED_GRID" << std::endl;
  out << "POINTS " << V.rows() << " float" << std::endl;
  for (int i = 0; i < V.rows(); ++i) {
    out << std::setprecision(4) << V.row(i).x() << " " << V.row(i).y() << " " << V.row(i).z() << std::endl;
  }
  out << "CELLS " << V.rows() << " " << V.rows() * (1 + 1) << std::endl;
  for (int i = 0; i < V.rows(); ++i) {
    out << "1 " << i << std::endl;
  }
  out << "CELL_TYPES " << V.rows() << std::endl;
  for (int i = 0; i < V.rows(); ++i) {
    out << 1 << std::endl;
  }
}

void FindSurfTriForTet(const Eigen::MatrixXi& TT, Eigen::MatrixXi& SF) {
  std::map<std::tuple<int, int, int>, int> tri2Tet;
  for (int elemI = 0; elemI < TT.rows(); elemI++) {
    const Eigen::RowVector4i& elemVInd = TT.row(elemI);
    tri2Tet[std::make_tuple(elemVInd[0], elemVInd[2], elemVInd[1])] = elemI;
    tri2Tet[std::make_tuple(elemVInd[0], elemVInd[3], elemVInd[2])] = elemI;
    tri2Tet[std::make_tuple(elemVInd[0], elemVInd[1], elemVInd[3])] = elemI;
    tri2Tet[std::make_tuple(elemVInd[1], elemVInd[2], elemVInd[3])] = elemI;
  }

  //TODO: parallelize
  std::vector<Eigen::RowVector3i> tmpF;
  for (const auto& triI : tri2Tet) {
    const auto& triVInd = triI.first;
    // find dual triangle with reversed indices:
    bool isSurfaceTriangle = //
        tri2Tet.find(std::make_tuple(std::get<2>(triVInd), std::get<1>(triVInd), std::get<0>(triVInd))) == tri2Tet.end()
        && tri2Tet.find(std::make_tuple(std::get<1>(triVInd), std::get<0>(triVInd), std::get<2>(triVInd))) == tri2Tet.end()
        && tri2Tet.find(std::make_tuple(std::get<0>(triVInd), std::get<2>(triVInd), std::get<1>(triVInd))) == tri2Tet.end();
    if (isSurfaceTriangle) {
      tmpF.emplace_back(std::get<0>(triVInd), std::get<1>(triVInd), std::get<2>(triVInd));
    }
  }

  SF.resize(tmpF.size(), 3);
  for (int i = 0; i < SF.rows(); i++) {
    SF.row(i) = tmpF[i];
  }
}

bool ReadTetMesh(const std::string& filePath, Eigen::MatrixXd& TV, Eigen::MatrixXi& TT,
                        Eigen::MatrixXi& SF) {
  using namespace boost::filesystem;
  if (!exists(path(filePath))) {
    return false;
  }

  mshio::MshSpec spec;
  try {
    spec = mshio::load_msh(filePath);
  }
  catch (...) {
    spdlog::error("MshIO only supports MSH 2.2 and 4.1");
    exit(-1);
  }

  const auto& nodes = spec.nodes;
  const auto& els = spec.elements;
  const int vAmt = nodes.num_nodes;
  int elemAmt = 0;
  for (const auto& e : els.entity_blocks) {
    assert(e.entity_dim == 3);
    assert(e.element_type == 4); // linear tet
    elemAmt += e.num_elements_in_block;
  }

  TV.resize(vAmt, 3);
  int index = 0;
  for (const auto& n : nodes.entity_blocks) {
    for (int i = 0; i < n.num_nodes_in_block * 3; i += 3) {
      TV.row(index) << n.data[i], n.data[i + 1], n.data[i + 2];
      ++index;
    }
  }

  TT.resize(elemAmt, 4);
  int elm_index = 0;
  for (const auto& e : els.entity_blocks) {
    for (int i = 0; i < e.data.size(); i += 5) {
      index = 0;
      for (int j = i + 1; j <= i + 4; ++j) {
        TT(elm_index, index++) = e.data[j] - 1;
      }
      ++elm_index;
    }
  }

  // finding the surface because $Surface is not supported by MshIO
  spdlog::info("Finding the surface triangle mesh for {:s}", filePath);
  FindSurfTriForTet(TT, SF);

  spdlog::info("tet mesh loaded with {:d} particles, {:d} tets, and {:d} surface triangles.",
               TV.rows(), TT.rows(), SF.rows());

  return true;
}

void WriteOBJ(const std::string &path, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
  std::ofstream out(path);
  for (int i = 0; i < V.rows(); ++i) {
    out << std::setprecision(18) << "v " << V.row(i).x() << " " << V.row(i).y() << " " << V.row(i).z() << std::endl;
  }

  int cols = static_cast<int>(F.cols());
  char type;
  switch (cols) {
    case 1:
      type = 'p';
      break;
    case 2:
      type = 'l';
      break;
    case 3:
      type = 'f';
      break;
    default:
      std::cout << "function WriteOBJ: fail to write obj file, F.cols must in {1, 2, 3}" << std::endl;
      return;
  }
  for (int i = 0; i < F.rows(); ++i) {
    out << type;
    for (int j = 0; j < cols; ++j) {
      out << " " << F(i, j) + 1; // index start from 1 in obj file
    }
    out << std::endl;
  }
}

}  // namespace da::sha
