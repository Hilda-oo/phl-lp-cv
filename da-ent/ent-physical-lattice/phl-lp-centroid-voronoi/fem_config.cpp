#include <spdlog/spdlog.h>
#include <fstream>
#include <iostream>
#include <json.hpp>
using json = nlohmann::json;

#include "fem_config.h"

namespace da {

void hasObject(const json &data, const std::string &key) {
  if (!data.count(key)) {
    spdlog::error("no \"{}\" in json file", key);
    exit(-1);
  }
}

bool FEMConfig::loadFromJSON(const std::string &p_filePath) {
  filePath = p_filePath;
  std::ifstream f(filePath);
  if (!f) {
    return false;
  }

  json data = json::parse(f);

  if (data.count("outputPath")) {  // outputPath
    outputPath = data["outputPath"];
  }

  if (data.count("material")) {  // material
    YM      = data["material"][0];
    PR      = data["material"][1];
    density = data["material"][2];
  }

  hasObject(data, "mshFilePath");
  mshFilePath = data["mshFilePath"];

  if (data.count("DBC")) {  // DBC
    int DBCNum = data["DBC"].size();
    for (int _i = 0; _i < DBCNum; ++_i) {
      const auto &DBCI = data["DBC"][_i];
      hasObject(DBCI, "min");
      Eigen::Vector3d minBBox(DBCI["min"][0], DBCI["min"][1], DBCI["min"][2]);
      hasObject(DBCI, "max");
      Eigen::Vector3d maxBBox(DBCI["max"][0], DBCI["max"][1], DBCI["max"][2]);
      std::array<double, 2> timeRange = {0.0,
                                         std::numeric_limits<double>::infinity()};  // default value
      if (DBCI.count("timeRange")) {
        timeRange[0] = DBCI["timeRange"][0];
        timeRange[1] = DBCI["timeRange"][1];
      }
      DirichletBCs.emplace_back(sha::DirichletBC(minBBox, maxBBox, timeRange));
    }
  }

  if (data.count("NBC")) {
    int NBCNum = data["NBC"].size();
    for (int _i = 0; _i < NBCNum; ++_i) {
      const auto &NBCI = data["NBC"][_i];
      hasObject(NBCI, "min");
      Eigen::Vector3d minBBox(NBCI["min"][0], NBCI["min"][1], NBCI["min"][2]);
      hasObject(NBCI, "max");
      Eigen::Vector3d maxBBox(NBCI["max"][0], NBCI["max"][1], NBCI["max"][2]);
      hasObject(NBCI, "force");
      Eigen::Vector3d force(NBCI["force"][0], NBCI["force"][1], NBCI["force"][2]);
      std::array<double, 2> timeRange = {0.0,
                                         std::numeric_limits<double>::infinity()};  // default value
      if (NBCI.count("timeRange")) {
        timeRange[0] = NBCI["timeRange"][0];
        timeRange[1] = NBCI["timeRange"][1];
      }
      NeumannBCs.emplace_back(sha::NeumannBC(minBBox, maxBBox, force, timeRange));
    }
  }

  return true;
}

void FEMConfig::backUpConfig(const std::string &p_filePath) {
  std::ifstream src(filePath, std::ios::binary);
  std::ofstream dst(p_filePath, std::ios::binary);
  dst << src.rdbuf();
}

}  // namespace da
