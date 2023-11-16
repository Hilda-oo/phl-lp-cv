#include <spdlog/spdlog.h>
#include <fstream>
#include <iostream>
#include "../../nlohmann/json.hpp"
using json = nlohmann::json;

#include "config.h"

namespace da {

bool hasObject(const json &data, const std::string &key) {
  if (!data.count(key)) {
    spdlog::error("no \"{}\" in json file", key);
    exit(-1);
  }
  return true;
}

bool Config::loadFromJSON(const std::string &p_filePath) {
  filePath = p_filePath;
  std::ifstream f(filePath);
  if (!f) {
    return false;
  }

  json data = json::parse(f);

  if (data.count("material")) {  // material
    hasObject(data["material"], "thermal_conductivity");
    thermal_conductivity = data["material"]["thermal_conductivity"];
  }

  if (data.count("topology")) {
    max_loop = data["topology"]["max_loop"];
    volfrac  = data["topology"]["volfrac"];
    penal    = data["topology"]["penal"];
    r_min    = data["topology"]["r_min"];
  }

  if (hasObject(data, "model")) {
    auto &model = data["model"];
    if (hasObject(model, "relativeLengthOfVoxel")) {
      relativeLengthOfVoxel = model["relativeLengthOfVoxel"];
    }
    if (hasObject(model, "arbitraryModel")) {
      arbitraryModel = model["arbitraryModel"];
    }
    if (model.count("chosenModel")) {
      chosenModel = model["chosenModel"];
    } else {
      spdlog::info("no chosenModel in json file, use \"{}\" instead.", arbitraryModel);
      chosenModel = arbitraryModel;
    }
    if (model.count("visualObjModel")) {
      visualObjPath = model["visualObjModel"];
    }
  }

  if (data.count("absDBC")) {
    int DBCNum = data["absDBC"].size();
    for (int _i = 0; _i < DBCNum; ++_i) {
      const auto &DBCI = data["absDBC"][_i];
      hasObject(DBCI, "min");
      Eigen::Vector3d minBBox(DBCI["min"][0], DBCI["min"][1], DBCI["min"][2]);
      hasObject(DBCI, "max");
      Eigen::Vector3d maxBBox(DBCI["max"][0], DBCI["max"][1], DBCI["max"][2]);
      // hasObject(DBCI, "dir");
      // Eigen::Vector3i dir(DBCI["dir"][0], DBCI["dir"][1], DBCI["dir"][2]);
      v_Dir.emplace_back(Dir(minBBox, maxBBox, {1, 0, 0}));
    }
  }

  if (data.count("absNBC")) {
    int NBCNum = data["absNBC"].size();
    for (int _i = 0; _i < NBCNum; ++_i) {
      const auto &NBCI = data["absNBC"][_i];
      hasObject(NBCI, "min");
      Eigen::Vector3d minBBox(NBCI["min"][0], NBCI["min"][1], NBCI["min"][2]);
      hasObject(NBCI, "max");
      Eigen::Vector3d maxBBox(NBCI["max"][0], NBCI["max"][1], NBCI["max"][2]);
      hasObject(NBCI, "val");
      Eigen::Vector3d val(NBCI["val"], 0, 0);
      v_Neu.emplace_back(Neu(minBBox, maxBBox, val));
    }
  }
  return true;
}

void Config::backUpConfig(const std::string &p_filePath) {
  std::ifstream src(filePath, std::ios::binary);
  std::ofstream dst(p_filePath, std::ios::binary);
  dst << src.rdbuf();
}

}  // namespace da
