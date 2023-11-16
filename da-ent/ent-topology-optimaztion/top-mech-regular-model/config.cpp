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
    YM = data["material"][0];
    PR = data["material"][1];
    // density = data["material"][2];
  }

  if (data.count("topology")) {
    max_loop = data["topology"]["max_loop"];
    volfrac  = data["topology"]["volfrac"];
    penal    = data["topology"]["penal"];
    r_min    = data["topology"]["r_min"];
  }

  if (hasObject(data, "model")) {
    auto &model = data["model"];
    if (hasObject(model, "regularModel")) {
      len_x = model["regularModel"][0];
      len_y = model["regularModel"][1];
      len_z = model["regularModel"][2];
    }
    if (model.count("visualObjModel")) {
      visualObjPath = model["visualObjModel"];
    }
  }

  if (data.count("DBC")) {
    int DBCNum = data["DBC"].size();
    for (int _i = 0; _i < DBCNum; ++_i) {
      const auto &DBCI = data["DBC"][_i];
      hasObject(DBCI, "min");
      Eigen::Vector3d minBBox(DBCI["min"][0], DBCI["min"][1], DBCI["min"][2]);
      hasObject(DBCI, "max");
      Eigen::Vector3d maxBBox(DBCI["max"][0], DBCI["max"][1], DBCI["max"][2]);
      hasObject(DBCI, "dir");
      Eigen::Vector3i dir(DBCI["dir"][0], DBCI["dir"][1], DBCI["dir"][2]);
      v_Dir.emplace_back(Dir(minBBox, maxBBox, dir));
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
      hasObject(NBCI, "val");
      Eigen::Vector3d val(NBCI["val"][0], NBCI["val"][1], NBCI["val"][2]);
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
