#include <spdlog/spdlog.h>
#include <cstddef>
#include <iostream>

#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"

#include "sha-io-foundation/mesh_io.h"

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

// ************* Input ***************
std::string working_directory_arg;
// ***********************************

// ************* Getter ***************
std::string WorkingDirectory() { return working_directory_arg; }
// ***********************************

// ************* Bridge ***************
void GenerateDensityFieldByRegularModel();
// ************************************

// **************** Command Line *******************
#ifdef DA_CMD
int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("top-heat-regular-model");
  entry.AddCmdOption()("help,h", "Print help");
  entry.AddCmdOption()("working_directory,w",
                       po::value<std::string>(&working_directory_arg)
                           ->default_value(WorkingAssetDirectoryPath().string()));

  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    GenerateDensityFieldByRegularModel();
  });
  return 0;
}
#endif
// ************************************************

// **************** Python Lib ********************
#ifdef DA_PY
namespace py = pybind11;
PYBIND11_MODULE(dapy_top_regular_model, m) {
  m.doc() = "top-heat-regular-model";
  py::class_<da::MatMesh3>(m, "MatMesh3", py::module_local())
      .def(py::init<>())
      .def_readwrite("mat_coordinates", &da::MatMesh3::mat_coordinates)
      .def_readwrite("mat_faces", &da::MatMesh3::mat_faces);

  m.def("GenerateDensityFieldByRegularModel", &GenerateDensityFieldByRegularModel,
        py::call_guard<py::gil_scoped_release>(), "GenerateDensityFieldByRegularModel");
}
#endif
// ************************************************