#include <cstddef>
#include <iostream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"

#include "sha-surface-mesh/matmesh.h"

// ************* Input ***************
size_t num_samples_on_longest_side_arg;
bool reuse_mesh_sdf_flag_arg;
std::string &tpms_type_arg = *(new std::string);
double coeffient_arg;
double offset_arg;
// ***********************************

// ************* Getter ***************
size_t NumSamplesAlongLongestSide() { return num_samples_on_longest_side_arg; }
std::string TPMSType() { return tpms_type_arg; }
double Coeffient() { return coeffient_arg; }
double Offset() { return offset_arg; }
bool ReuseMeshSdfFlag() { return reuse_mesh_sdf_flag_arg; }
// ***********************************

// ************* Bridge ***************
auto GenerateTPMSLatticeStructure(size_t num_samples_on_longest_side, double coeffient,
                                  double offset, const std::string &tpms_type,
                                  bool reuse_mesh_sdf_flag) -> da::MatMesh3;
// ************************************

// **************** Command Line *******************
#ifdef DA_CMD
int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("lad-tpms-structure");
  entry.AddCmdOption()("coeffient,a", po::value<double>(&coeffient_arg)->default_value(10));
  entry.AddCmdOption()("offset,t", po::value<double>(&offset_arg)->default_value(0));
  entry.AddCmdOption()("tpms,s", po::value<std::string>(&tpms_type_arg)->default_value("Schwarzp"),
                       "[Schwarzp, DoubleP, Schwarzd, DoubleD]");
  entry.AddCmdOption()("num_samples",
                       po::value<size_t>(&num_samples_on_longest_side_arg)->default_value(80),
                       "Number of samples along longest side");
  entry.AddCmdOption()("reuse_sdf",
                       po::value<bool>(&reuse_mesh_sdf_flag_arg)->default_value(false));

  entry.AddCmdOption()("help,h", "Print help");
  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    GenerateTPMSLatticeStructure(NumSamplesAlongLongestSide(), Coeffient(), Offset(), TPMSType(),
                                 ReuseMeshSdfFlag());
  });
  return 0;
}
#endif
// ************************************************

// **************** Python Lib ********************
#ifdef DA_PY
namespace py = pybind11;
PYBIND11_MODULE(dapy_lad_tpms_structure, m) {
  m.doc() = "lad-tpms-structure";

  py::class_<da::MatMesh3>(m, "MatMesh3", py::module_local())
      .def(py::init<>())
      .def_readwrite("mat_coordinates", &da::MatMesh3::mat_coordinates)
      .def_readwrite("mat_faces", &da::MatMesh3::mat_faces);

  m.def("GenerateTPMSLatticeStructure", &GenerateTPMSLatticeStructure,
        py::call_guard<py::gil_scoped_release>(), "GenerateTPMSLatticeStructure",
        py::arg("num_samples_on_longest_side"), py::arg("coeffient"), py::arg("offset"),
        py::arg("tpms_type"), py::arg("reuse_mesh_sdf_flag"));
}
#endif
// ************************************************
