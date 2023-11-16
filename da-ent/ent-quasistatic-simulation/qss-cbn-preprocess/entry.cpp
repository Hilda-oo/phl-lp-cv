#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"

#include "sha-surface-mesh/matmesh.h"

// ************* Input ***************
std::string &fine_mesh_path_arg       = *(new std::string);
std::string &output_dir_path_arg      = *(new std::string);
std::vector<int> &level_num_faces_arg = *(new std::vector<int>);
std::string &tetgen_macro_switch_arg  = *(new std::string);
std::string &tetgen_micro_switch_arg  = *(new std::string);
// ***********************************

// ************* Getter ***************
std::string FineMeshPath() { return fine_mesh_path_arg; }
std::string OutputDirectoryPath() { return output_dir_path_arg; }
std::string TetgenMacroSwitches() { return tetgen_macro_switch_arg; }
std::string TetgenMicroSwitches() { return tetgen_micro_switch_arg; }
std::vector<int> LevelNumFaces() { return level_num_faces_arg; }
// ***********************************

// ************* Bridge ***************
size_t GenerateCBNBackgroundMesh(const std::string &fine_mesh_path,
                                 const std::string &output_dir_path,
                                 const std::vector<int> &level_num_faces,
                                 const std::string tetgen_macro_switch,
                                 const std::string tetgen_micro_switch);

// ************************************

// **************** Command Line *******************
#ifdef DA_CMD
namespace std {
std::ostream &operator<<(std::ostream &os, const std::vector<int> &vec) {
  for (auto item : vec) {
    os << item << " ";
  }
  return os;
}
}  // namespace std
int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("lad-cbn-preprocess");

  entry.AddCmdOption()("help,h", "Print help");
  entry.AddCmdOption()(
      "input,i", po::value<std::string>(&fine_mesh_path_arg)
                     ->default_value((da::WorkingAssetDirectoryPath() / "model.obj").string()));
  entry.AddCmdOption()("output,o", po::value<std::string>(&output_dir_path_arg)
                                       ->default_value(da::WorkingResultDirectoryPath().string()));
  entry.AddCmdOption()("level,l", po::value<std::vector<int>>(&level_num_faces_arg)
                                      ->multitoken()
                                      ->default_value(std::vector<int>{500, 200, 100}));
  entry.AddCmdOption()("tetgen_macro",
                       po::value<std::string>(&tetgen_macro_switch_arg)->default_value("pQq1"));
  entry.AddCmdOption()("tetgen_micro",
                       po::value<std::string>(&tetgen_micro_switch_arg)->default_value("pQa0.01"));
  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    GenerateCBNBackgroundMesh(FineMeshPath(), OutputDirectoryPath(), LevelNumFaces(),
                              TetgenMacroSwitches(), TetgenMicroSwitches());
  });
  return 0;
}
#endif
// ************************************************

// **************** Python Lib ********************
#ifdef DA_PY
namespace py = pybind11;
PYBIND11_MODULE(dapy_qss_cbn_preprocess, m) {
  m.doc() = "qss-cbn-preprocess";

  m.def("GenerateCBNBackgroundMesh", &GenerateCBNBackgroundMesh,
        py::call_guard<py::gil_scoped_release>(), "GenerateCBNBackgroundMesh",
        py::arg("fine_mesh_path"), py::arg("output_dir_path"), py::arg("level_num_faces"),
        py::arg("tetgen_macro_switch"), py::arg("tetgen_micro_switch"));
}
#endif
// ************************************************
