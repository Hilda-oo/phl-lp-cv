#pragma once

#include <boost/program_options.hpp>
#include <functional>

namespace da {
namespace po = boost::program_options;
class EntryProgram {
  using ProcessingFunction = std::function<void(po::variables_map & , po::options_description & )>;
 public:
  explicit EntryProgram(const std::string &caption) : description_(caption) {}

  virtual auto AddCmdOption() -> po::options_description_easy_init;

  void Run(int argc, char **argv, const ProcessingFunction &Processing);

 protected:
  po::options_description description_;
};
} // namespace da