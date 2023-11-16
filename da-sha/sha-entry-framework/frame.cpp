#include "frame.h"

#include <exception>
#include <iostream>
#include "sha-base-framework/frame.h"

namespace da {
auto EntryProgram::AddCmdOption() -> boost::program_options::options_description_easy_init {
  return description_.add_options();
}

void EntryProgram::Run(int argc, char **argv, const ProcessingFunction &Processing) {
  po::variables_map variables_map;
  po::store(boost::program_options::parse_command_line(argc, argv, description_), variables_map);
  po::notify(variables_map);
  if (Processing != nullptr) {
    Processing(variables_map, description_);
//    try {
//      Processing(variables_map, description_);
//    } catch (const std::exception e) {
//      std::cout << e.what() << std::endl;
//      Terminate("Uncatched exception");
//    }
  }
}
}  // namespace da
