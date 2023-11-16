#include <cstddef>
#include <iostream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"

#include "sha-surface-mesh/matmesh.h"

// ************* Input ***************
size_t num_samples_on_longest_side_arg;
size_t num_cells_along_x_axis_arg;
size_t num_cells_along_y_axis_arg;
size_t num_cells_along_z_axis_arg;
std::string &field_type_arg = *(new std::string);
double field_coeff_arg;
double radius_of_lattice_frame_arg;
double radius_of_shell_frame_arg;
double sharp_angle_arg;
bool reuse_mesh_sdf_flag_arg;
size_t lattice_structure_type_arg;
// ***********************************

// ************* Getter ***************
size_t NumSamplesAlongLongestSide() { return num_samples_on_longest_side_arg; }
size_t NumCellsAlongXAxis() { return num_cells_along_x_axis_arg; }
size_t NumCellsAlongYAxis() { return num_cells_along_y_axis_arg; }
size_t NumCellsAlongZAxis() { return num_cells_along_z_axis_arg; }
size_t LatticeStructureType() { return lattice_structure_type_arg; }
std::string FieldType() { return field_type_arg; }
double FieldCoeff() { return field_coeff_arg; }
double RadiusOfLatticeFrame() { return radius_of_lattice_frame_arg; }
double RadiusOfShellFrame() { return radius_of_shell_frame_arg; }
double SharpAngle() { return sharp_angle_arg; }
bool ReuseMeshSdfFlag() { return reuse_mesh_sdf_flag_arg; }
// ***********************************

// ************* Bridge ***************
da::MatMesh3 GenerateVoxelLatticeStructureByField(size_t num_samples_on_longest_side,
                                           size_t num_cells_along_x_axis,
                                           size_t num_cells_along_y_axis,
                                           size_t num_cells_along_z_axis,
                                           size_t lattice_structure_type,
                                           const std::string &field_type,
                                           double field_coeff, double radius_of_lattice_frame,
                                           double radius_of_shell_frame, double sharp_angle,
                                           bool reuse_mesh_sdf_flag);

// ************************************

// **************** Command Line *******************
#ifdef DA_CMD
int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("lad-field-driven-voxel");
  entry.AddCmdOption()("field,f", po::value<std::string>(&field_type_arg)->default_value("Matrix"),
                       "[Matrix,NoField, F1, F2, F3]");
  entry.AddCmdOption()("field_coeff,c", po::value<double>(&field_coeff_arg)->default_value(1));
  entry.AddCmdOption()("lattice_radius",
                       po::value<double>(&radius_of_lattice_frame_arg)->default_value(1));
  //------------add by zcp-----------
  entry.AddCmdOption()("shell_radius",
                       po::value<double>(&radius_of_shell_frame_arg)->default_value(1));
  entry.AddCmdOption()("type", po::value<size_t>(&lattice_structure_type_arg)->default_value(0));
  entry.AddCmdOption()("sharp", po::value<double>(&sharp_angle_arg)->default_value(0.5));
  entry.AddCmdOption()("num_samples",
                       po::value<size_t>(&num_samples_on_longest_side_arg)->default_value(80),
                       "Number of samples along longest side");
  entry.AddCmdOption()("cell_x", po::value<size_t>(&num_cells_along_x_axis_arg)->default_value(20));
  entry.AddCmdOption()("cell_y", po::value<size_t>(&num_cells_along_y_axis_arg)->default_value(6));
  entry.AddCmdOption()("cell_z", po::value<size_t>(&num_cells_along_z_axis_arg)->default_value(6));
  entry.AddCmdOption()("reuse_sdf",
                       po::value<bool>(&reuse_mesh_sdf_flag_arg)->default_value(false));

  entry.AddCmdOption()("help,h", "Print help");
  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    GenerateVoxelLatticeStructureByField(NumSamplesAlongLongestSide(), NumCellsAlongXAxis(),
                                         NumCellsAlongYAxis(), NumCellsAlongZAxis(),
                                         LatticeStructureType(), FieldType(), FieldCoeff(),
                                         RadiusOfLatticeFrame(), RadiusOfShellFrame(),
                                         SharpAngle(), ReuseMeshSdfFlag());
  });
  return 0;
}
#endif
// ************************************************

// **************** Python Lib ********************
#ifdef DA_PY
namespace py = pybind11;
PYBIND11_MODULE(dapy_lad_field_driven_voxel, m) {
  m.doc() = "lad-field-driven-voxel";

  py::class_<da::MatMesh3>(m, "MatMesh3", py::module_local())
      .def(py::init<>())
      .def_readwrite("mat_coordinates", &da::MatMesh3::mat_coordinates)
      .def_readwrite("mat_faces", &da::MatMesh3::mat_faces);

  m.def("GenerateVoxelLatticeStructureByField", &GenerateVoxelLatticeStructureByField,
        py::call_guard<py::gil_scoped_release>(), "GenerateVoxelLatticeStructureByField",
        py::arg("num_samples_on_longest_side"), py::arg("num_cells_along_x_axis"),
        py::arg("num_cells_along_y_axis"), py::arg("num_cells_along_z_axis"),
        py::arg("lattice_structure_type"), py::arg("field_type"), py::arg("field_coeff"),
        py::arg("radius_of_lattice_frame"), py::arg("radius_of_shell_frame"),
        py::arg("sharp_angle"), py::arg("reuse_mesh_sdf_flag"));
}
#endif
// ************************************************