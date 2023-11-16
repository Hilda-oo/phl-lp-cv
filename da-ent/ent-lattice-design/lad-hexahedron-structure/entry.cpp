#include <spdlog/spdlog.h>
#include <cstddef>
#include <iostream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"

#include "sha-io-foundation/mesh_io.h"
#include "sha-surface-mesh/sample.h"

// ************* Input ***************
double alpha_arg;
double beta_arg;
size_t num_iterations_arg;
std::string &tetgen_switches = *(new std::string);
double radius_of_lattice_frame_arg;
size_t lattice_structure_type_arg;
size_t num_samples_on_longest_side_arg;
size_t num_cells_along_x_axis_arg;
size_t num_cells_along_y_axis_arg;
size_t num_cells_along_z_axis_arg;
// ***********************************

// ************* Getter ***************
double Alpha() { return alpha_arg; }
double Beta() { return beta_arg; }
size_t NumberOfIterations() { return num_iterations_arg; }
std::string TetGenSwitches() { return tetgen_switches; }
double RadiusOfLatticeFrame() { return radius_of_lattice_frame_arg; }
size_t LatticeStructureType() { return lattice_structure_type_arg; }
size_t NumberOfSamplesAlongLongestSide() { return num_samples_on_longest_side_arg; }
Eigen::Vector3i NumCellsAlongAxes() {
  return Eigen::Vector3i(num_cells_along_x_axis_arg, num_cells_along_y_axis_arg,
                         num_cells_along_z_axis_arg);
}
// ***********************************

// ************* Bridge ***************
auto GenerateHexahedronLatticeStructure(size_t num_samples_on_longest_side, double alpha,
                                        double beta, size_t num_iterations,
                                        Eigen::Vector3i num_cells, double radius,
                                        size_t structure_type, const std::string &tetgen_switches)
    -> da::MatMesh3;
// ************************************

// **************** Command Line *******************
#ifdef DA_CMD
int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("lad-hexahedron-structure");
  entry.AddCmdOption()("help,h", "Print help");
  entry.AddCmdOption()("alpha", po::value<double>(&alpha_arg)->default_value(0.1));
  entry.AddCmdOption()("beta", po::value<double>(&beta_arg)->default_value(1));
  entry.AddCmdOption()("iterations", po::value<size_t>(&num_iterations_arg)->default_value(20));
  entry.AddCmdOption()("tetgen_switches",
                       po::value<std::string>(&tetgen_switches)->default_value("pQq1"));
  entry.AddCmdOption()("lattice_radius",
                       po::value<double>(&radius_of_lattice_frame_arg)->default_value(0.06));
  entry.AddCmdOption()("type", po::value<size_t>(&lattice_structure_type_arg)->default_value(2));
  entry.AddCmdOption()("num_samples",
                       po::value<size_t>(&num_samples_on_longest_side_arg)->default_value(100),
                       "Number of samples along longest side");
  entry.AddCmdOption()("cell_x", po::value<size_t>(&num_cells_along_x_axis_arg)->default_value(6));
  entry.AddCmdOption()("cell_y", po::value<size_t>(&num_cells_along_y_axis_arg)->default_value(6));
  entry.AddCmdOption()("cell_z", po::value<size_t>(&num_cells_along_z_axis_arg)->default_value(6));
  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    GenerateHexahedronLatticeStructure(
        NumberOfSamplesAlongLongestSide(), Alpha(), Beta(), NumberOfIterations(),
        NumCellsAlongAxes(), RadiusOfLatticeFrame(), LatticeStructureType(), TetGenSwitches());
  });
  return 0;
}
#endif
// ************************************************

// **************** Python Lib ********************
#ifdef DA_PY
namespace py = pybind11;
PYBIND11_MODULE(dapy_lad_hexahedron_structure, m) {
  m.doc() = "lad-hexahedron-structure";

  py::class_<da::MatMesh3>(m, "MatMesh3", py::module_local())
      .def(py::init<>())
      .def_readwrite("mat_coordinates", &da::MatMesh3::mat_coordinates)
      .def_readwrite("mat_faces", &da::MatMesh3::mat_faces);

  m.def("GenerateHexahedronLatticeStructure", &GenerateHexahedronLatticeStructure,
        py::call_guard<py::gil_scoped_release>(), "GenerateHexahedronLatticeStructure",
        py::arg("num_samples_on_longest_side"), py::arg("alpha"), py::arg("beta"),
        py::arg("num_iterations"), py::arg("num_cells"), py::arg("radius"),
        py::arg("structure_type"), py::arg("tetgen_switches"));
}
#endif