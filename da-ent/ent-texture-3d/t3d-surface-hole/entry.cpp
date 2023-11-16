#include <spdlog/spdlog.h>
#include <cstddef>
#include <iostream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"

#include "parameterization.h"
#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/sample.h"

// ************* Input ***************
size_t num_samples_arg;
double hole_height_arg;
double hole_radius_arg;
bool curvature_field_arg;
// ***********************************

// ************* Getter ***************
size_t NumberOfSamples() { return num_samples_arg; }
double HoleHeight() { return hole_height_arg; }
double HoleRadius() { return hole_radius_arg; }
bool CurvatureField() { return curvature_field_arg; }
// ***********************************

// ************* Bridge ***************
auto GenerateSurfaceHoles(size_t expected_num_samples, double hole_height, double hole_radius,
                          bool use_curvature_field_flag) -> da::MatMesh3;
// ************************************

// **************** Command Line *******************
#ifdef DA_CMD
int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("t3d-surface-hole");
  entry.AddCmdOption()("num_samples", po::value<size_t>(&num_samples_arg)->default_value(15));
  entry.AddCmdOption()("hole_height", po::value<double>(&hole_height_arg)->default_value(0.08));
  entry.AddCmdOption()("hole_radius", po::value<double>(&hole_radius_arg)->default_value(0.03));
  entry.AddCmdOption()("curvature_field",
                       po::value<bool>(&curvature_field_arg)->default_value(false));
  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    GenerateSurfaceHoles(NumberOfSamples(), HoleHeight(), HoleRadius(), CurvatureField());
  });
  return 0;
}
#endif
// ************************************************

// **************** Python Lib ********************
#ifdef DA_PY
namespace py = pybind11;
PYBIND11_MODULE(dapy_t3d_surface_hole, m) {
  m.doc() = "t3d-surface-hole";

  py::class_<da::MatMesh3>(m, "MatMesh3", py::module_local())
      .def(py::init<>())
      .def_readwrite("mat_coordinates", &da::MatMesh3::mat_coordinates)
      .def_readwrite("mat_faces", &da::MatMesh3::mat_faces);

  m.def("GenerateSurfaceHoles", &GenerateSurfaceHoles, py::call_guard<py::gil_scoped_release>(),
        "GenerateSurfaceHoles", py::arg("expected_num_samples"), py::arg("hole_height"),
        py::arg("hole_radius"), py::arg("use_curvature_field_flag"));
}
#endif
// ************************************************
