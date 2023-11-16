#include "mesh_io.h"

#include <CGAL/Surface_mesh/IO.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/readSTL.h>
#include <igl/writeOBJ.h>

#include <cstdio>
#include <fstream>
#include "sha-base-framework/frame.h"

namespace da {
namespace sha {
SurfaceMesh3 ReadSurfaceMesh3FromOFF(const fs_path &file_path) {
  SurfaceMesh3 mesh;
  CGAL::IO::read_OFF(file_path.string(), mesh);
  return mesh;
}

SurfaceMesh3 ReadSurfaceMesh3FromOBJ(const fs_path &file_path) {
  SurfaceMesh3 mesh;
  CGAL::IO::read_OBJ(file_path.string(), mesh);
  return mesh;
}

MatMesh3 ReadMatMeshFromOBJ(const fs_path &file_path) {
  MatMesh3 matmesh;
  igl::readOBJ(file_path.string(), matmesh.mat_coordinates, matmesh.mat_faces);
  return matmesh;
}

MatMesh3 ReadMatMeshFromSTL(const fs_path &file_path) {
  MatMesh3 matmesh;
  Eigen::MatrixXd mat_normals;
  std::ifstream mesh_instream(file_path.string());
  igl::readSTL(mesh_instream, matmesh.mat_coordinates, matmesh.mat_faces, mat_normals);
  return matmesh;
}

MatMesh3 ReadMatMeshFromPLY(const fs_path &file_path) {
  MatMesh3 matmesh;
  std::ifstream mesh_instream(file_path.string());
  Eigen::MatrixXd mat_normals;

  Eigen::MatrixXd mat_uv, mat_vertex_data, mat_face_data, mesh_edge_data;
  Eigen::MatrixXi mat_edges;
  std::vector<std::string> vtx_data_headers, edge_data_headers, face_data_headers, comments;

  igl::readPLY(mesh_instream, matmesh.mat_coordinates, matmesh.mat_faces, mat_edges, mat_normals,
               mat_uv, mat_vertex_data, vtx_data_headers, mat_face_data, face_data_headers,
               mesh_edge_data, edge_data_headers, comments);

  return matmesh;
}

void WriteMatMesh3ToObj(const fs_path &file_path, const MatMesh3 &matmesh,
                        const Eigen::MatrixXd &mat_vtx_colors) {
  if (mat_vtx_colors.rows() == 0) {
    igl::writeOBJ(file_path.string(), matmesh.mat_coordinates, matmesh.mat_faces);
  } else {  // with colors
    Assert(matmesh.NumVertices() == mat_vtx_colors.rows(),
           "The number of colors and vertices should be equal");
    Assert(mat_vtx_colors.cols() == 3, "Cols of colors must be 3");
    std::ofstream obj_out_stream(file_path.string());
    if (not obj_out_stream.is_open()) {
      Terminate("File " + file_path.string() + " can not be opened.");
    }
    for (index_t vtx_idx = 0; vtx_idx < matmesh.NumVertices(); ++vtx_idx) {
      obj_out_stream << "v " << matmesh.mat_coordinates(vtx_idx, 0) << " "
                     << matmesh.mat_coordinates(vtx_idx, 1) << " "
                     << matmesh.mat_coordinates(vtx_idx, 2) << " ";
      obj_out_stream << mat_vtx_colors(vtx_idx, 0) << " " << mat_vtx_colors(vtx_idx, 1) << " "
                     << mat_vtx_colors(vtx_idx, 2) << std::endl;
    }

    for (index_t face_idx = 0; face_idx < matmesh.NumFaces(); ++face_idx) {
      obj_out_stream << "f " << matmesh.mat_faces(face_idx, 0) + 1 << " "
                     << matmesh.mat_faces(face_idx, 1) + 1 << " "
                     << matmesh.mat_faces(face_idx, 2) + 1 << std::endl;
    }
    obj_out_stream.close();
  }
}

void WriteToVtk(const fs_path &file_path, const Eigen::MatrixXd &mat_coordinates,
                const Eigen::MatrixXi &mat_cells, const std::vector<double> &point_data,
                const std::vector<double> &cell_data, int cell_type) {
  std::ofstream vtk_out_stream(file_path.string());
  if (not vtk_out_stream.is_open()) {
    Terminate("File " + file_path.string() + " can not be opened.");
  }
  vtk_out_stream << "# vtk DataFile Version 3.0\n"
                    "Volume Mesh\n"
                    "ASCII\n"
                    "DATASET UNSTRUCTURED_GRID"
                 << std::endl;
  vtk_out_stream << "POINTS " << mat_coordinates.rows() << " float" << std::endl;
  for (index_t vertex_idx = 0; vertex_idx < mat_coordinates.rows(); ++vertex_idx) {
    vtk_out_stream << mat_coordinates.row(vertex_idx).x() << " "
                   << mat_coordinates.row(vertex_idx).y() << " "
                   << mat_coordinates.row(vertex_idx).z() << std::endl;
  }
  vtk_out_stream << "CELLS " << mat_cells.rows() << " " << mat_cells.rows() * (mat_cells.cols() + 1)
                 << std::endl;
  for (index_t face_idx = 0; face_idx < mat_cells.rows(); ++face_idx) {
    vtk_out_stream << mat_cells.cols();
    for (index_t col_idx = 0; col_idx < mat_cells.cols(); ++col_idx) {
      vtk_out_stream << " " << mat_cells(face_idx, col_idx);
    }
    vtk_out_stream << std::endl;
  }
  vtk_out_stream << "CELL_TYPES " << mat_cells.rows() << std::endl;
  for (index_t face_idx = 0; face_idx < mat_cells.rows(); ++face_idx) {
    vtk_out_stream << cell_type << std::endl;
  }

  if (!point_data.empty()) {
    vtk_out_stream << "POINT_DATA " << point_data.size() << "\n"
                   << "SCALARS point_scalars double 1\n"
                   << "LOOKUP_TABLE default" << std::endl;
    for (auto &data : point_data) {
      vtk_out_stream << data << std::endl;
    }
  }

  if (!cell_data.empty()) {
    vtk_out_stream << "CELL_DATA " << cell_data.size() << "\n"
                   << "SCALARS cell_scalars double 1\n"
                   << "LOOKUP_TABLE default" << std::endl;

    for (auto &data : cell_data) {
      vtk_out_stream << data << std::endl;
    }
  }
}

void ReadFromVtk(const fs_path &file_path, Eigen::MatrixXd &mat_coordinates,
                 Eigen::MatrixXi &mat_cells, const size_t cell_size) {
  std::ifstream infile_stream(file_path.string());
  std::string line;
  std::string string_temp;
  size_t num_vertices;
  size_t num_cells;
  int inttmp;
  for (index_t line_idx = 0; line_idx < 4; ++line_idx) {
    std::getline(infile_stream, line);
  }
  std::getline(infile_stream, line);
  std::stringstream line_stream(line);
  line_stream >> string_temp >> num_vertices >> string_temp;
  mat_coordinates.resize(num_vertices, 3);
  for (index_t vtx_idx = 0; vtx_idx < num_vertices; ++vtx_idx) {
    std::getline(infile_stream, line);
    std::stringstream ss(line);
    ss >> mat_coordinates(vtx_idx, 0) >> mat_coordinates(vtx_idx, 1) >> mat_coordinates(vtx_idx, 2);
  }
  std::getline(infile_stream, line);
  line_stream = std::stringstream(line);
  line_stream >> string_temp >> num_cells >> string_temp;
  mat_cells.resize(num_cells, cell_size);
  for (int cell_idx = 0; cell_idx < num_cells; ++cell_idx) {
    std::getline(infile_stream, line);
    std::stringstream cell_stream(line);
    cell_stream >> inttmp;
    for (int j = 0; j < cell_size; j++) {
      cell_stream >> mat_cells(cell_idx, j);
    }
  }
}

void WriteMatMesh3ToVtk(const fs_path &file_path, const MatMesh3 &matmesh,
                        const std::vector<double> &point_data,
                        const std::vector<double> &cell_data) {
  WriteToVtk(file_path, matmesh.mat_coordinates, matmesh.mat_faces, point_data, cell_data, 5);
}

void WriteMatMesh2ToVtk(const fs_path &file_path, const MatMesh2 &matmesh,
                        const std::vector<double> &point_data,
                        const std::vector<double> &cell_data) {
  WriteToVtk(file_path, matmesh.mat_coordinates, matmesh.mat_beams, point_data, cell_data, 3);
}

void WritePointsToVtk(const fs_path &file_path, const Eigen::MatrixXd &mat_points,
                      const std::vector<double> &point_data) {
  const size_t num_points = mat_points.rows();
  Eigen::MatrixXi mat_cells(num_points, 1);
  mat_cells.col(0).setLinSpaced(0, num_points - 1);
  WriteToVtk(file_path, mat_points, mat_cells, point_data, {}, 1);
}

void WriteTetrahedralMatmeshToVtk(const fs_path &file_path, const TetrahedralMatMesh &matmesh,
                                  const std::vector<double> &point_data,
                                  const std::vector<double> &cell_data) {
  WriteToVtk(file_path, matmesh.mat_coordinates, matmesh.mat_tetrahedrons, point_data, cell_data,
             10);
}

void WriteHexahedralMatmeshToVtk(const fs_path &file_path, const HexahedralMatMesh &matmesh,
                                 const std::vector<double> &point_data,
                                 const std::vector<double> &cell_data) {
  WriteToVtk(file_path, matmesh.mat_coordinates, matmesh.mat_hexahedrons, point_data, cell_data,
             12);
}

TetrahedralMatMesh ReadTetrahedralMatMeshFromVtk(const fs_path &file_path) {
  TetrahedralMatMesh mesh;
  ReadFromVtk(file_path, mesh.mat_coordinates, mesh.mat_tetrahedrons, 4);
  return mesh;
}
}  // namespace sha
}  // namespace da
