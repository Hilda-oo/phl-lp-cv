#include <iostream>
#include <tuple>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"

#include "sha-io-foundation/mesh_io.h"

// ************* Input ***************
std::string working_directory_arg;
// ***********************************

// ************* Getter ***************
std::string WorkingDirectory() { return working_directory_arg; }
// ***********************************

// ************* Bridge ***************
auto QuasistaticSimulationByFEMTetrahedral()
    -> std::tuple<da::MatMesh3, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>;
// ************************************

// **************** Command Line *******************
#ifdef DA_CMD
int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("qss-fem-tetrahedral");
  entry.AddCmdOption()("help,h", "Print help");
  entry.AddCmdOption()("working_directory,w",
                       po::value<std::string>(&working_directory_arg)
                           ->default_value(WorkingAssetDirectoryPath().string()));

  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    QuasistaticSimulationByFEMTetrahedral();
  });
  return 0;
}
#endif
// ************************************************

// **************** Python Lib ********************
#ifdef DA_PY
namespace py = pybind11;
PYBIND11_MODULE(dapy_qss_fem_tetrahedral, m) {
  m.doc() = "lad-tpms-structure";
  py::class_<da::MatMesh3>(m, "MatMesh3", py::module_local())
      .def(py::init<>())
      .def_readwrite("mat_coordinates", &da::MatMesh3::mat_coordinates)
      .def_readwrite("mat_faces", &da::MatMesh3::mat_faces);

  m.def("QuasistaticSimulationByFEMTetrahedral", &QuasistaticSimulationByFEMTetrahedral,
        py::call_guard<py::gil_scoped_release>(), "QuasistaticSimulationByFEMTetrahedral");
}
#endif
// ************************************************