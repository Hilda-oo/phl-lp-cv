#include <iostream>
#include <tuple>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"

#include "sha-io-foundation/mesh_io.h"

// ************* Input ***************
std::string mesh_path_arg;
std::string mesh_property_path_arg;
// ***********************************

// ************* Getter ***************
std::string MeshPath() { return mesh_path_arg; }
std::string MeshPropertyPath() { return mesh_property_path_arg; }
// ***********************************

// ************* Bridge ***************
auto DensityFieldDisplay(const std::string &mesh_path, const std::string &mesh_property_path)
    -> std::tuple<da::MatMesh3, Eigen::VectorXd>;
// ************************************

// **************** Command Line *******************
#ifdef DA_CMD
int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("top-density-field-display");
  entry.AddCmdOption()("help,h", "Print help");
  entry.AddCmdOption()("mesh_path,m",
                       po::value<std::string>(&mesh_path_arg)
                           ->default_value((WorkingAssetDirectoryPath() / "mesh.obj").string()));
  entry.AddCmdOption()("mesh_property_path,p",
                       po::value<std::string>(&mesh_property_path_arg)
                           ->default_value((WorkingAssetDirectoryPath() / "mesh_property.txt").string()));

  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    DensityFieldDisplay(mesh_path_arg, mesh_property_path_arg);
  });
  return 0;
}
#endif
// ************************************************

// **************** Python Lib ********************
#ifdef DA_PY
namespace py = pybind11;
PYBIND11_MODULE(dapy_top_density_field_display, m) {
  m.doc() = "top-density-field-display";
  py::class_<da::MatMesh3>(m, "MatMesh3", py::module_local())
      .def(py::init<>())
      .def_readwrite("mat_coordinates", &da::MatMesh3::mat_coordinates)
      .def_readwrite("mat_faces", &da::MatMesh3::mat_faces);

  m.def("DensityFieldDisplay", &DensityFieldDisplay,
        py::call_guard<py::gil_scoped_release>(), "DensityFieldDisplay",
        py::arg("mesh_path"), py::arg("mesh_property_path"));
}
#endif
// ************************************************