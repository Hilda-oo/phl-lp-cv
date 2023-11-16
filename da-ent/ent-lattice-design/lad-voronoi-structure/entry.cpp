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
size_t num_samples_on_longest_side_arg;
size_t num_voronoi_seeds_arg;
double radius_of_lattice_frame_arg;
double sharp_angle_arg;
// ***********************************

// ************* Getter ***************
size_t NumSamplesAlongLongestSide() { return num_samples_on_longest_side_arg; }
size_t NumVoronoiSeeds() { return num_voronoi_seeds_arg; }
double RadiusOfLatticeFrame() { return radius_of_lattice_frame_arg; }
double SharpAngle() { return sharp_angle_arg; }
// ***********************************

// ************* Bridge ***************
auto GenerateVoronoiLatticeStructure(size_t num_samples_on_longest_side, size_t num_voronoi_seeds,
                                     double radius_of_lattice_frame, double sharp_angle)
    -> da::MatMesh3;
// ************************************

// **************** Command Line *******************
#ifdef DA_CMD
int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("lad-voronoi-structure");
  entry.AddCmdOption()("lattice_radius",
                       po::value<double>(&radius_of_lattice_frame_arg)->default_value(0.1));
  entry.AddCmdOption()("sharp", po::value<double>(&sharp_angle_arg)->default_value(70));
  entry.AddCmdOption()("num_seeds", po::value<size_t>(&num_voronoi_seeds_arg)->default_value(100));
  entry.AddCmdOption()("num_samples",
                       po::value<size_t>(&num_samples_on_longest_side_arg)->default_value(100),
                       "Number of samples along longest side");
  entry.AddCmdOption()("help,h", "Print help");
  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }

    GenerateVoronoiLatticeStructure(NumSamplesAlongLongestSide(), NumVoronoiSeeds(),
                                    RadiusOfLatticeFrame(), SharpAngle());
  });
  return 0;
}
#endif
// ************************************************

// **************** Python Lib ********************
#ifdef DA_PY
namespace py = pybind11;
PYBIND11_MODULE(dapy_lad_voronoi_structure, m) {
  m.doc() = "lad-new-voronoi";

  py::class_<da::MatMesh3>(m, "MatMesh3", py::module_local())
      .def(py::init<>())
      .def_readwrite("mat_coordinates", &da::MatMesh3::mat_coordinates)
      .def_readwrite("mat_faces", &da::MatMesh3::mat_faces);

  m.def("GenerateVoronoiLatticeStructure", &GenerateVoronoiLatticeStructure,
        py::call_guard<py::gil_scoped_release>(), "GenerateVoronoiLatticeStructure",
        py::arg("num_samples_on_longest_side"), py::arg("num_voronoi_seeds"),
        py::arg("radius_of_lattice_frame"), py::arg("sharp_angle"));
}
#endif
// ************************************************
