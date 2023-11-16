#include "frame.h"

#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <exception>
#include <iostream>
#include <string>

#include <boost/stacktrace.hpp>

namespace da {
void Terminate(const std::string &message) {
  std::cout << "Terminate: " << message << std::endl;
  std::cout << "Stack Tracing: " << std::endl;
  std::cout << boost::stacktrace::stacktrace() << std::endl;
  exit(1);
}

void Assert(bool condition, const std::string &message) {
  if (condition) return;
  if (message.empty()) {
    std::cout << "Assertion Failed" << std::endl;
  } else {
    std::cout << "Assertion Failed: " << message << std::endl;
  }
  std::cout << "Stack Tracing: " << std::endl;
  std::cout << boost::stacktrace::stacktrace() << std::endl;
  exit(2);
}

auto WorkplacePath() -> fs_path {
  return boost::filesystem::absolute(boost::filesystem::current_path().parent_path().parent_path() /
                                     "exe" / "Workplace");
}
auto WorkingAssetDirectoryPath() -> fs_path { return WorkplacePath() / "assets"; }
auto WorkingResultDirectoryPath() -> fs_path { return WorkplacePath() / "results"; }
auto ProjectSourcePath() -> fs_path {
  return boost::filesystem::absolute(boost::filesystem::current_path().parent_path().parent_path());
}
auto ProjectAssetDirectoryPath() -> fs_path { return ProjectSourcePath() / "assets"; }
}  //  namespace da
