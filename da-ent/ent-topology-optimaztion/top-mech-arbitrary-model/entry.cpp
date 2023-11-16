/*
 * @Author: lab pc yjxkwp@foxmail.com
 * @Date: 2023-05-04 12:57:27
 * @LastEditors: lab pc yjxkwp@foxmail.com
 * @LastEditTime: 2023-05-04 17:46:23
 * @FilePath: /designauto/da-ent/ent-topology-optimaztion/top-arbitrary-model/entry.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <spdlog/spdlog.h>
#include <cstddef>
#include <iostream>

#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"

#include "sha-io-foundation/mesh_io.h"

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <pybind11/pybind11.h>

// ************* Input ***************
std::string working_directory_arg;
// ***********************************

// ************* Getter ***************
std::string WorkingDirectory() { return working_directory_arg; }
// ***********************************

// ************* Bridge ***************
void GenerateDensityFieldByArbitraryModel();
// ************************************

// **************** Command Line *******************
#ifdef DA_CMD
int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("top-mech_arbitrary-model");
  entry.AddCmdOption()("help,h", "Print help");
  entry.AddCmdOption()("working_directory,w",
                       po::value<std::string>(&working_directory_arg)
                           ->default_value(WorkingAssetDirectoryPath().string()));

  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    GenerateDensityFieldByArbitraryModel();
  });
  return 0;
}
#endif
// ************************************************

// **************** Python Lib ********************
#ifdef DA_PY
namespace py = pybind11;
PYBIND11_MODULE(dapy_top_arbitrary_model, m) {
  m.doc() = "top-mech_arbitrary-model";
  py::class_<da::MatMesh3>(m, "MatMesh3", py::module_local())
      .def(py::init<>())
      .def_readwrite("mat_coordinates", &da::MatMesh3::mat_coordinates)
      .def_readwrite("mat_faces", &da::MatMesh3::mat_faces);

  m.def("GenerateDensityFieldByArbitraryModel", &GenerateDensityFieldByArbitraryModel,
        py::call_guard<py::gil_scoped_release>(), "GenerateDensityFieldByArbitraryModel");
}
#endif
// ************************************************