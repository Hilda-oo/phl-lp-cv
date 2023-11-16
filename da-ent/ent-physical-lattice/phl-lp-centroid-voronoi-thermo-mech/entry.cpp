#include <spdlog/spdlog.h>
#include <boost/program_options/value_semantic.hpp>
#include <cstddef>
#include <iostream>

#include "sha-base-framework/frame.h"
#include "sha-entry-framework/frame.h"

#include "sha-io-foundation/mesh_io.h"

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

std::string working_directory_arg;

int seeds_num_arg;

int mode_arg;
int seed_num_arg;
std::string model_file_name_arg;
std::string model_name_arg;

std::string WorkingDirectory() { return working_directory_arg; }

void GeneratePhysicalLpNormVoronoiLatticeStructure(std::string model_file_name, std::string model_name, int seed_num, int mode);

void GenerateSampleSeed(int num_seed, std::string model_file, std::string model_name);

int main(int argc, char **argv) {
  using namespace da;  // NOLINT
  EntryProgram entry("phl-lp-centroid-voronoi");
  entry.AddCmdOption()("help,h", "Print help");
  entry.AddCmdOption()("generate,g", po::value<int>(&seeds_num_arg)->default_value(-1));
  entry.AddCmdOption()("working_directory,w",
                       po::value<std::string>(&working_directory_arg)
                           ->default_value(WorkingAssetDirectoryPath().string()));
  entry.AddCmdOption()("model_file,f", po::value<std::string>(&model_file_name_arg)->default_value("femur"));                         
  entry.AddCmdOption()("model_name,n", po::value<std::string>(&model_name_arg)->default_value("femur.obj"));                         
  entry.AddCmdOption()("mode,m", po::value<int>(&mode_arg)->default_value(0));                         
  entry.AddCmdOption()("seed_num,s", po::value<int>(&seed_num_arg)->default_value(100));           

  entry.Run(argc, argv, [&](auto &variables_map, auto &description) {
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return;
    }
    if (variables_map.count("generate") && seeds_num_arg > 0) {
      GenerateSampleSeed(seeds_num_arg, model_file_name_arg, model_name_arg);
    } else {
      GeneratePhysicalLpNormVoronoiLatticeStructure(model_file_name_arg, model_name_arg, seed_num_arg, mode_arg);
    }
  });
  return 0;
}
