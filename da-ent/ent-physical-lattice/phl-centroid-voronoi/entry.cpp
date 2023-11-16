#include <spdlog/spdlog.h>
#include <cstddef>
#include <iostream>

#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"

#include "sha-io-foundation/mesh_io.h"

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

std::string working_directory_arg;

std::string WorkingDirectory() { return working_directory_arg; }

void GeneratePhysicalVoronoiLatticeStructure();

int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("phl-centroid-voronoi");
  entry.AddCmdOption()("help,h", "Print help");
  entry.AddCmdOption()("working_directory,w",
                       po::value<std::string>(&working_directory_arg)
                           ->default_value(WorkingAssetDirectoryPath().string()));

  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    GeneratePhysicalVoronoiLatticeStructure();
  });
  return 0;
}
