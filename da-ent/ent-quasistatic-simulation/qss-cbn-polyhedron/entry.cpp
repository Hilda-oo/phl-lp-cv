#include <iostream>
#include <tuple>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"

#include "sha-io-foundation/mesh_io.h"

// ************* Input ***************
std::string config_file_arg;
double YM_arg;
double PR_arg;
double density_arg;
// ***********************************

// ************* Getter ***************
std::string ConfigFile() { return config_file_arg; }
double YoungsModulus() { return YM_arg; }
double PoissionRatio() { return PR_arg; }
double Density() { return density_arg; }
// ***********************************

// ************* Bridge ***************
auto QuasistaticSimulationByCBN(std::string config_file, double YM, double PR, double density)
    -> std::tuple<da::MatMesh3, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>;
// ************************************

// **************** Command Line *******************
#ifdef DA_CMD
int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("qss-cbn-polyhedron");
  entry.AddCmdOption()("help,h", "Print help");
  entry.AddCmdOption()("config_file,c",
                       po::value<std::string>(&config_file_arg)
                           ->default_value((WorkingAssetDirectoryPath() / "config.json").string()));
  entry.AddCmdOption()("YM", po::value<double>(&YM_arg)->default_value(1e5));
  entry.AddCmdOption()("PR", po::value<double>(&PR_arg)->default_value(0.3));
  entry.AddCmdOption()("density", po::value<double>(&density_arg)->default_value(1e3));

  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    QuasistaticSimulationByCBN(config_file_arg, YoungsModulus(), PoissionRatio(), Density());
  });
  return 0;
}
#endif
// ************************************************

// **************** Python Lib ********************
#ifdef DA_PY
namespace py = pybind11;
PYBIND11_MODULE(dapy_qss_cbn_polyhedron, m) {
  m.doc() = "qss-cbn-polyhedron";
  py::class_<da::MatMesh3>(m, "MatMesh3", py::module_local())
      .def(py::init<>())
      .def_readwrite("mat_coordinates", &da::MatMesh3::mat_coordinates)
      .def_readwrite("mat_faces", &da::MatMesh3::mat_faces);

  m.def("QuasistaticSimulationByCBN", &QuasistaticSimulationByCBN,
        py::call_guard<py::gil_scoped_release>(), "QuasistaticSimulationByCBN",
        py::arg("config_file"), py::arg("YM"), py::arg("PR"), py::arg("density"));
}
#endif
// ************************************************