#include <boost/filesystem/operations.hpp>
#include <iostream>
#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"


std::string working_directory_arg;

int seeds_num_arg;

int mode_arg;

std::string WorkingDirectory() { return working_directory_arg; }

void GeneratePhysicalLpNormVoronoiLatticeStructure(int mode);

int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("phl-lp-cv-thermo-mech");
  entry.AddCmdOption()("help,h", "Print help");
  entry.AddCmdOption()("working_directory,w",
                       po::value<std::string>(&working_directory_arg)
                           ->default_value(ProjectSourcePath().string()));
  entry.AddCmdOption()("mode,m", po::value<int>(&mode_arg)->default_value(0));

  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    GeneratePhysicalLpNormVoronoiLatticeStructure(mode_arg);
  });
  return 0;
}
