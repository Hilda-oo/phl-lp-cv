#pragma once
#include <string>
#include <vector>

#include "sha-base-framework/declarations.h"
#include "sha-base-framework/frame.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-surface-mesh/mesh3.h"
#include "sha-volume-mesh/matmesh.h"

namespace da {
namespace sha {
SurfaceMesh3 ReadSurfaceMesh3FromOFF(const fs_path &file_path);

SurfaceMesh3 ReadSurfaceMesh3FromOBJ(const fs_path &file_path);

MatMesh3 ReadMatMeshFromOBJ(const fs_path &file_path);

MatMesh3 ReadMatMeshFromSTL(const fs_path &file_path);

MatMesh3 ReadMatMeshFromPLY(const fs_path &file_path);

void WriteMatMesh3ToObj(const fs_path &file_path, const MatMesh3 &matmesh,
                        const Eigen::MatrixXd &mat_vtx_colors);

void WriteToVtk(const fs_path &file_path, const Eigen::MatrixXd &mat_coordinates,
                const Eigen::MatrixXi &mat_cells, const std::vector<double> &point_data,
                const std::vector<double> &cell_data, int cell_type);

void WriteMatMesh3ToVtk(const fs_path &file_path, const MatMesh3 &matmesh,
                        const std::vector<double> &point_data = {},
                        const std::vector<double> &cell_data  = {});

void WriteMatMesh2ToVtk(const fs_path &file_path, const MatMesh2 &matmesh,
                        const std::vector<double> &point_data = {},
                        const std::vector<double> &cell_data  = {});

void WritePointsToVtk(const fs_path &file_path, const Eigen::MatrixXd &mat_points,
                      const std::vector<double> &point_data = {});

void WriteTetrahedralMatmeshToVtk(const fs_path &file_path, const TetrahedralMatMesh &matmesh,
                                  const std::vector<double> &point_data = {},
                                  const std::vector<double> &cell_data  = {});

void WriteHexahedralMatmeshToVtk(const fs_path &file_path, const HexahedralMatMesh &matmesh,
                                 const std::vector<double> &point_data = {},
                                 const std::vector<double> &cell_data  = {});

void ReadFromVtk(const fs_path &file_path, Eigen::MatrixXd &mat_coordinates,
                 Eigen::MatrixXi &mat_cells, const size_t cell_size);

TetrahedralMatMesh ReadTetrahedralMatMeshFromVtk(const fs_path &file_path);
}  // namespace sha
}  // namespace da