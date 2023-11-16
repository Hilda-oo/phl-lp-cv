#include "utility.h"

#include "sha-io-foundation/data_io.h"

namespace da {
namespace sha {
double LinearInterpolateFor1D(double v1, double v2, double x1, double x2, double x) {
  double t = (x - x1) / (x2 - x1);
  return v1 * (1 - t) + v2 * t;
}

double LinearInterpolateFor2D(double v1, double v2, double v3, double v4, double x1, double x2,
                              double x, double y1, double y2, double y) {
  double s = LinearInterpolateFor1D(v1, v2, x1, x2, x);
  double t = LinearInterpolateFor1D(v3, v4, x1, x2, x);
  return LinearInterpolateFor1D(s, t, y1, y2, y);
}

double LinearInterpolateFor3D(double v1, double v2, double v3, double v4, double v5, double v6,
                              double v7, double v8, double x1, double x2, double x, double y1,
                              double y2, double y, double z1, double z2, double z) {
  double s = LinearInterpolateFor2D(v1, v2, v3, v4, x1, x2, x, y1, y2, y);
  double t = LinearInterpolateFor2D(v5, v6, v7, v8, x1, x2, x, y1, y2, y);
  return LinearInterpolateFor1D(s, t, z1, z2, z);
}

MatMesh2 LoadStructureVF(index_t struct_type, const fs_path &microstructure_base_path) {
  MatMesh2 microstructure_beam_mesh;

  std::string typeName = "Type" + std::to_string(struct_type);
  fs_path basePath(microstructure_base_path);
  auto typeBasePath = basePath / typeName;
  auto typeVPath    = typeBasePath / (typeName + "_Lattice_Vertex.txt");
  auto typeFPath    = typeBasePath / (typeName + "_Lattice_BarList.txt");

  microstructure_beam_mesh.mat_coordinates = sha::ReadDoubleMatrixFromFile(typeVPath.string());
  microstructure_beam_mesh.mat_beams       = sha::ReadIntMatrixFromFile(typeFPath.string());
  microstructure_beam_mesh.mat_beams.array() -= 1;
  return microstructure_beam_mesh;
}

auto ReadTrianglePatchFromMicrostructureBase(const std::vector<index_t> &types,
                                             const fs_path &microstructure_base_path)
    -> std::map<index_t, MatMesh3> {
  std::map<index_t, MatMesh3> map_type_idx_to_patch_mesh;
  std::set<int> type_set(types.begin(), types.end());

  for (const auto &type_idx : type_set) {
    std::string type_name = std::to_string(type_idx);
    fs_path vertex_path, face_path;
    vertex_path = microstructure_base_path / ("Type" + type_name) /
                  (("Type" + type_name) + "_Patch_Vertex.txt");
    face_path = microstructure_base_path / ("Type" + type_name) /
                (("Type" + type_name) + "_Patch_TriList.txt");

    Assert(boost::filesystem::exists(vertex_path),
           (vertex_path.string() + " does not exist").c_str());
    Assert(boost::filesystem::exists(face_path), (face_path.string() + " does not exist").c_str());
    MatMesh3 microstructure_patch_mesh;
    microstructure_patch_mesh.mat_coordinates = sha::ReadDoubleMatrixFromFile(vertex_path.string());
    microstructure_patch_mesh.mat_faces       = sha::ReadIntMatrixFromFile(face_path.string());
    map_type_idx_to_patch_mesh[type_idx]      = microstructure_patch_mesh;
  }
  return map_type_idx_to_patch_mesh;
}

}  // namespace sha
}  // namespace da
