#include <cstddef>
#include <iostream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"

#include "sha-surface-mesh/matmesh.h"

// ************* Input ***************
size_t num_samples_on_longest_side_arg;
std::string &tpms_type_arg = *(new std::string);
double tpms_coeff_arg;
double offset_arg;
std::string &field_type_arg = *(new std::string);
double field_coeff_arg;
bool reuse_mesh_sdf_flag_arg;
// ***********************************

// ************* Getter ***************
size_t NumSamplesAlongLongestSide() { return num_samples_on_longest_side_arg; }
std::string TPMSType() { return tpms_type_arg; }
double TPMSCoeff() { return tpms_coeff_arg; }
double Offset() { return offset_arg; }
std::string FieldType() { return field_type_arg; }
double FieldCoeff() { return field_coeff_arg; }
bool ReuseMeshSdfFlag() { return reuse_mesh_sdf_flag_arg; }
// ***********************************

// ************* Bridge ***************
auto GenerateTPMSLatticeStructureByField(size_t num_samples_on_longest_side, 
                                         const std::string &tpms_type,double tpms_coeff, 
                                         double offset, const std::string &field_type,
                                         double field_coeff, 
                                         bool reuse_mesh_sdf_flag) -> da::MatMesh3;
// ************************************

// **************** Command Line *******************
#ifdef DA_CMD
int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("lad-field-driven-tpms");
  entry.AddCmdOption()("tpms,s", po::value<std::string>(&tpms_type_arg)->default_value("G"),
                       "[G, Schwarzp, DoubleP, Schwarzd, DoubleD]");
  entry.AddCmdOption()("tpms_coeff,a", po::value<double>(&tpms_coeff_arg)->default_value(0.6));
  entry.AddCmdOption()("offset,t", po::value<double>(&offset_arg)->default_value(0));
  entry.AddCmdOption()("field,f", po::value<std::string>(&field_type_arg)->default_value("Matrix"),
                       "[NoField, F1, F2, F3]");
  entry.AddCmdOption()("field_coeff,b", po::value<double>(&field_coeff_arg)->default_value(0.2));
  entry.AddCmdOption()("num_samples",
                       po::value<size_t>(&num_samples_on_longest_side_arg)->default_value(300),
                       "Number of samples along longest side");
  entry.AddCmdOption()("reuse_sdf",
                       po::value<bool>(&reuse_mesh_sdf_flag_arg)->default_value(false));

  entry.AddCmdOption()("help,h", "Print help");
  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    GenerateTPMSLatticeStructureByField(NumSamplesAlongLongestSide(), TPMSType(),
                                        TPMSCoeff(), Offset(), FieldType(),
                                        FieldCoeff(), ReuseMeshSdfFlag());
  });
  return 0;
}
#endif
// ************************************************

// **************** Python Lib ********************
#ifdef DA_PY
namespace py = pybind11;
PYBIND11_MODULE(dapy_lad_field_driven_tpms, m) {
  m.doc() = "lad-field-driven-tpms";

  py::class_<da::MatMesh3>(m, "MatMesh3", py::module_local())
      .def(py::init<>())
      .def_readwrite("mat_coordinates", &da::MatMesh3::mat_coordinates)
      .def_readwrite("mat_faces", &da::MatMesh3::mat_faces);

  m.def("GenerateTPMSLatticeStructureByField", &GenerateTPMSLatticeStructureByField,
        py::call_guard<py::gil_scoped_release>(), "GenerateTPMSLatticeStructureByField",
        py::arg("num_samples_on_longest_side"), py::arg("tpms_type"), py::arg("tpms_coeff"),
        py::arg("offset"), py::arg("field_type"), py::arg("field_coeff"),
        py::arg("reuse_mesh_sdf_flag"));
}
#endif
// ************************************************